// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
    GPUDirect Storage functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_gds_handle.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include "deepspeed_gds_op.h"

using namespace std;

int deepspeed_gds_handle_t::s_cuFile_init = 0;
std::mutex deepspeed_gds_handle_t::s_cuFile_mutex;
std::string deepspeed_gds_handle_t::s_cuFile_config_path;
std::string deepspeed_gds_handle_t::s_previous_cuFile_config_path;
bool deepspeed_gds_handle_t::s_had_previous_cuFile_config_path = false;

namespace {

std::string create_cufile_config(const std::string& contents)
{
    char path_template[] = "/tmp/deepspeed_cufile_XXXXXX";
    const int fd = mkstemp(path_template);
    if (fd == -1) {
        throw std::runtime_error(std::string("Unable to create temporary cuFile config: ") +
                                 std::strerror(errno));
    }

    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t written = write(fd, contents.data() + offset, contents.size() - offset);
        if (written == -1 && errno == EINTR) { continue; }
        if (written <= 0) {
            const int error_code = errno;
            close(fd);
            unlink(path_template);
            throw std::runtime_error(std::string("Unable to write temporary cuFile config: ") +
                                     std::strerror(error_code));
        }
        offset += static_cast<size_t>(written);
    }

    if (close(fd) == -1) {
        const int error_code = errno;
        unlink(path_template);
        throw std::runtime_error(std::string("Unable to close temporary cuFile config: ") +
                                 std::strerror(error_code));
    }
    return path_template;
}

void remove_cufile_config(std::string& config_path,
                          std::string& previous_config_path,
                          bool& had_previous_config_path)
{
    if (config_path.empty()) { return; }

    const char* configured_path = std::getenv("CUFILE_ENV_PATH_JSON");
    if (configured_path != nullptr && config_path == configured_path) {
        if (had_previous_config_path) {
            setenv("CUFILE_ENV_PATH_JSON", previous_config_path.c_str(), 1);
        } else {
            unsetenv("CUFILE_ENV_PATH_JSON");
        }
    }
    unlink(config_path.c_str());
    config_path.clear();
    previous_config_path.clear();
    had_previous_config_path = false;
}

}  // namespace

deepspeed_gds_handle_t::deepspeed_gds_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const int intra_op_parallelism)
    : deepspeed_io_handle_t(
          block_size, queue_depth, single_submit, overlap_events, intra_op_parallelism),
      _intra_gds_op_parallelism(intra_op_parallelism)
{
    _init_cuFile(block_size, queue_depth);
}

deepspeed_gds_handle_t::~deepspeed_gds_handle_t() { _close_cuFile(); }

const int deepspeed_gds_handle_t::get_intra_op_parallelism() const
{
    return _intra_gds_op_parallelism;
}

void deepspeed_gds_handle_t::_init_cuFile(const int block_size, const int queue_depth)
{
    std::lock_guard<std::mutex> lock(deepspeed_gds_handle_t::s_cuFile_mutex);
    if (deepspeed_gds_handle_t::s_cuFile_init == 0) {
        std::string depthStr = std::to_string(queue_depth);
        std::string threadsStr = std::to_string(_intra_gds_op_parallelism);
        const char* pinnedMemEnv = std::getenv("DEEPSPEED_GDS_MAX_DEVICE_PINNED_MEM_KB");
        const char* cacheMemEnv = std::getenv("DEEPSPEED_GDS_MAX_DEVICE_CACHE_KB");
        std::string pinnedMemKb = pinnedMemEnv != nullptr ? std::string(pinnedMemEnv) : "134217728";
        std::string cacheMemKb = cacheMemEnv != nullptr ? std::string(cacheMemEnv) : "1048576";
        std::string json1 = R"({"execution": {"max_io_queue_depth": )" + depthStr + ", ";
        std::string json2 = R"("max_request_parallelism": )" + threadsStr + ", ";
        std::string json3 = R"("max_io_threads": )" + threadsStr + ", ";
        std::string json4 = R"("parallel_io": true, "min_io_threshold_size_kb": 8192}, )";
        std::string json5 = R"("properties": {"max_device_cache_size_kb": )" + cacheMemKb + ", ";
        std::string json6 = R"("per_buffer_cache_size_kb": 1024, "max_device_pinned_mem_size_kb": )" + pinnedMemKb + ", ";
        std::string json7 = R"("allow_compat_mode": false, "use_poll_mode": false}})";
        const std::string config_contents = json1 + json2 + json3 + json4 + json5 + json6 + json7;
        const char* previous_config_path = std::getenv("CUFILE_ENV_PATH_JSON");
        deepspeed_gds_handle_t::s_had_previous_cuFile_config_path = previous_config_path != nullptr;
        deepspeed_gds_handle_t::s_previous_cuFile_config_path =
            previous_config_path != nullptr ? previous_config_path : "";
        deepspeed_gds_handle_t::s_cuFile_config_path = create_cufile_config(config_contents);
        if (setenv("CUFILE_ENV_PATH_JSON", deepspeed_gds_handle_t::s_cuFile_config_path.c_str(), 1) != 0) {
            const int error_code = errno;
            remove_cufile_config(deepspeed_gds_handle_t::s_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_previous_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_had_previous_cuFile_config_path);
            throw std::runtime_error(std::string("Unable to set CUFILE_ENV_PATH_JSON: ") +
                                     std::strerror(error_code));
        }

        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            remove_cufile_config(deepspeed_gds_handle_t::s_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_previous_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_had_previous_cuFile_config_path);
            throw std::runtime_error(std::string("cuFileDriverOpen failed: ") + cuFileGetErrorString(status));
        }
        size_t direct_io_size = (size_t)block_size / 1024;
        status = cuFileDriverSetMaxDirectIOSize(direct_io_size);
        if (status.err != CU_FILE_SUCCESS) {
            cuFileDriverClose();
            remove_cufile_config(deepspeed_gds_handle_t::s_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_previous_cuFile_config_path,
                                 deepspeed_gds_handle_t::s_had_previous_cuFile_config_path);
            throw std::runtime_error(std::string("cuFileDriverSetMaxDirectIOSize failed: ") +
                                     cuFileGetErrorString(status));
        }
    }
    deepspeed_gds_handle_t::s_cuFile_init++;
}

void deepspeed_gds_handle_t::_close_cuFile()
{
    std::lock_guard<std::mutex> lock(deepspeed_gds_handle_t::s_cuFile_mutex);
    if (deepspeed_gds_handle_t::s_cuFile_init <= 0) { return; }
    deepspeed_gds_handle_t::s_cuFile_init--;
    if (deepspeed_gds_handle_t::s_cuFile_init == 0) {
        cuFileDriverClose();
        remove_cufile_config(deepspeed_gds_handle_t::s_cuFile_config_path,
                             deepspeed_gds_handle_t::s_previous_cuFile_config_path,
                             deepspeed_gds_handle_t::s_had_previous_cuFile_config_path);
    }
}

torch::Tensor deepspeed_gds_handle_t::new_pinned_device_tensor(const size_t num_elem,
                                                               const torch::Tensor& example_tensor)
{
    auto options = torch::TensorOptions().dtype(example_tensor.scalar_type()).device(torch::kCUDA);
    auto dev_tensor = torch::empty(num_elem, options);
    pin_device_tensor(dev_tensor);
    return dev_tensor;
}

bool deepspeed_gds_handle_t::free_pinned_device_tensor(torch::Tensor& buffer)
{
    unpin_device_tensor(buffer);
    return true;
}

bool deepspeed_gds_handle_t::pin_device_tensor(const torch::Tensor& buffer)
{
    gds_op_desc_t::add_buffer_to_registry(buffer);
    return true;
}

bool deepspeed_gds_handle_t::unpin_device_tensor(const torch::Tensor& buffer)
{
    gds_op_desc_t::remove_buffer_from_registry(buffer);
    return true;
}

std::shared_ptr<struct io_op_desc_t> deepspeed_gds_handle_t::_create_io_op_desc(
    const bool read_op,
    const torch::Tensor& buffer,
    const int fd,
    const char* filename,
    const bool validate,
    const int64_t file_offset)
{
    if (buffer.is_cuda()) {
        return std::make_shared<gds_op_desc_t>(
            read_op, buffer, fd, filename, _intra_op_parallelism, validate, file_offset);
    }
    return deepspeed_io_handle_t::_create_io_op_desc(
        read_op, buffer, fd, filename, validate, file_offset);
}
