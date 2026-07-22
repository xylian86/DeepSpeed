import torch

from deepspeed.runtime.swap_tensor.utils import PosixIOHandle, swap_in_tensors, swap_out_tensors


def test_posix_io_handle_roundtrip_bfloat16_with_offset(tmp_path):
    src = torch.arange(17, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    swap_path = tmp_path / "swap.bin"
    swap_offset = 4096

    handle = PosixIOHandle()
    assert swap_out_tensors(handle, [src], [swap_path], [swap_offset]) == 1
    assert handle.wait() == 1

    assert swap_in_tensors(handle, [dst], [swap_path], [swap_offset]) == 1
    assert handle.wait() == 1
    assert torch.equal(src, dst)
