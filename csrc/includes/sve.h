#pragma once

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

template <typename T>
inline T readAs(const void* src)
{
    T res;
    std::memcpy(&res, src, sizeof(T));
    return res;
}

template <typename T>
inline void writeAs(void* dst, const T& val)
{
    std::memcpy(dst, &val, sizeof(T));
}

#define ROUND_DOWN(size, step) ((size) & ~((step) - 1))
#define TILE (512 * 1024)

// SVE SIMD macros
#define SIMD_WIDTH 4
#define SIMD_PRED svptrue_pat_b32(SV_VL4)  // Predicate for exactly 4 elements
#define SIMD_STORE(dst, src) svst1_f32(SIMD_PRED, dst, src)
#define SIMD_LOAD(x) svld1_f32(svptrue_b32(), x)
#define SIMD_SET(x) svdup_n_f32(x)
#define SIMD_ADD(x, y) svadd_f32_x(svptrue_b32(), x, y)
#define SIMD_MUL(x, y) svmul_f32_x(SIMD_PRED, x, y)
#define SIMD_FMA(x, y, c) svmla_f32_x(SIMD_PRED, c, x, y)
#define SIMD_SQRT(x) svsqrt_f32_x(SIMD_PRED, x)
#define SIMD_DIV(x, y) svdiv_f32_x(SIMD_PRED, x, y)

// Optimized SVE Data structure for 128-bit vectors
union SVE_Data {
    float data_f[4] __attribute__((aligned(16))); // 16-byte alignment for 128-bit vectors
};

// Fixed predicate for exactly 4 elements
inline svbool_t get_sve_pred_4() {
    return svptrue_pat_b32(SV_VL4);
}

// Optimized load for 4-wide vectors
template <typename T>
inline void simd_load_4(SVE_Data* dst, T* src, size_t offset) {

    if constexpr (std::is_same_v<T, float>) {
        svfloat32_t data = svld1_f32(SIMD_PRED, src + offset);
        svst1_f32(SIMD_PRED, dst->data_f, data);
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        // Load 4 half precision values and convert to float
        svfloat16_t half_data = svld1_f16(svptrue_pat_b16(SV_VL4), (float16_t*)(src + offset));
        svfloat32_t float_data = svcvt_f32_f16_x(SIMD_PRED, half_data);
        svst1_f32(SIMD_PRED, dst->data_f, float_data);
    }
}

// Optimized store for 4-wide vectors
template <typename T>
inline void simd_store_4(T* dst, SVE_Data* src, size_t offset) {
    svfloat32_t data = svld1_f32(SIMD_PRED, src->data_f);
    
    if constexpr (std::is_same_v<T, float>) {
        svst1_f32(SIMD_PRED, dst + offset, data);
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        svfloat16_t half_data = svcvt_f16_f32_x(SIMD_PRED, data);
        svst1_f16(svptrue_pat_b16(SV_VL4), (float16_t*)(dst + offset), half_data);
    }
}

template <int span>
inline void simd_fma_4(SVE_Data* dst, SVE_Data* src1, SVE_Data* src2, SVE_Data* acc) {
    
    #pragma unroll 4
    for (int i = 0; i < span; i++) {
        svfloat32_t v1 = svld1_f32(SIMD_PRED, src1[i].data_f);
        svfloat32_t v2 = svld1_f32(SIMD_PRED, src2[i].data_f);
        svfloat32_t va = svld1_f32(SIMD_PRED, acc[i].data_f);
        
        // Optimized FMA for 4-wide vectors
        svfloat32_t result = svmad_f32_x(SIMD_PRED, v1, v2, va);
        svst1_f32(SIMD_PRED, dst[i].data_f, result);
    }
}

// FP16 conversion functions
static svfloat32_t load_sve_fp16_as_f32(const void* data)
{
    svfloat16_t fp16_data = svld1_f16(svptrue_b16(), (float16_t*)data);
    return svcvt_f32_f16_x(svptrue_b32(), fp16_data);
}

static void store_sve_fp16_from_f32(svfloat32_t v, void* data)
{
    svfloat16_t fp16_data = svcvt_f16_f32_x(svptrue_b16(), v);
    svst1_f16(svptrue_b16(), (float16_t*)data, fp16_data);
}

#define SIMD_LOAD_FP16(x) load_sve_fp16_as_f32(x)
#define SIMD_STORE_FP16(x, d) store_sve_fp16_from_f32(d, x)


// Helper functions to load/store SVE vectors
inline svfloat32_t load_sve_data(const SVE_Data& d)
{
    return svld1_f32(SIMD_PRED, d.data_f);
}

inline void store_sve_data(SVE_Data& d, svfloat32_t vec)
{
    svst1_f32(SIMD_PRED, d.data_f, vec);
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::Half>, void> simd_store(T* dst,
                                                                                SVE_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        // Load float32 data
        svfloat32_t f32_data = load_sve_data(src[i]);
        // Convert float32 to float16
        svfloat16_t f16_data = svcvt_f16_f32_x(SIMD_PRED, f32_data);
        // Store float16 data
        svst1_f16(SIMD_PRED, reinterpret_cast<float16_t*>(dst + width * i), f16_data);
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_store(T* dst, SVE_Data* src)
{
    size_t width = SIMD_WIDTH;
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec = load_sve_data(src[i]);
        SIMD_STORE(dst + width * i, vec);
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::Half>, void> simd_load(SVE_Data* dst,
                                                                               T* src)
{
    size_t width = SIMD_WIDTH;  // 128-bit = 4 floats
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        // Convert FP16 to FP32 and load
        svfloat32_t tmp = svcvt_f32_f16_x(SIMD_PRED, svld1_f16(SIMD_PRED, (float16_t*)(src + width * i)));
        store_sve_data(dst[i], tmp);
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_load(SVE_Data* dst, T* src)
{
    size_t width = SIMD_WIDTH;  // 128-bit = 4 floats
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        // Load directly as FP32
        svfloat32_t tmp = svld1_f32(SIMD_PRED, src + width * i);
        store_sve_data(dst[i], tmp);
    }
}

template <int span>
inline void simd_fma(SVE_Data* dst, SVE_Data* src_m_l, SVE_Data src_m_r, SVE_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_m_l[i]);
        svfloat32_t vec_r = load_sve_data(src_m_r);
        svfloat32_t vec_a = load_sve_data(src_a[i]);
        svfloat32_t result = SIMD_FMA(vec_l, vec_r, vec_a);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_fma(SVE_Data* dst, SVE_Data* src_m_l, SVE_Data src_m_r, SVE_Data src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_m_l[i]);
        svfloat32_t vec_r = load_sve_data(src_m_r);
        svfloat32_t vec_a = load_sve_data(src_a);
        svfloat32_t result = SIMD_FMA(vec_l, vec_r, vec_a);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_fma(SVE_Data* dst, SVE_Data* src_m_l, SVE_Data* src_m_r, SVE_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_m_l[i]);
        svfloat32_t vec_r = load_sve_data(src_m_r[i]);
        svfloat32_t vec_a = load_sve_data(src_a[i]);
        svfloat32_t result = SIMD_FMA(vec_l, vec_r, vec_a);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_mul(SVE_Data* dst, SVE_Data* src_a_l, SVE_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_a_l[i]);
        svfloat32_t vec_r = load_sve_data(src_a_r);
        svfloat32_t result = SIMD_MUL(vec_l, vec_r);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_mul(SVE_Data* dst, SVE_Data* src_a_l, SVE_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_a_l[i]);
        svfloat32_t vec_r = load_sve_data(src_a_r[i]);
        svfloat32_t result = SIMD_MUL(vec_l, vec_r);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_sqrt(SVE_Data* dst, SVE_Data* src)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec = load_sve_data(src[i]);
        svfloat32_t result = SIMD_SQRT(vec);
        store_sve_data(dst[i], result);
    }
}

template <int span>
inline void simd_div(SVE_Data* dst, SVE_Data* src_a_l, SVE_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        svfloat32_t vec_l = load_sve_data(src_a_l[i]);
        svfloat32_t vec_r = load_sve_data(src_a_r[i]);
        svfloat32_t result = SIMD_DIV(vec_l, vec_r);
        store_sve_data(dst[i], result);
    }
}

#endif
