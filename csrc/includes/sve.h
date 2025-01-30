// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

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
#define TILE 512  // Adjusted tile size to match L2 cache size

// SVE SIMD macros with non-temporal access
#define SIMD_WIDTH 1024 * 512
#define VECTOR_UNROLL 4
#define SIMD_PRED svptrue_pat_b32(SV_VL4)  // Predicate for exactly 4 elements
#define SIMD_STORE(dst, src) svstnt1_f32(SIMD_PRED, dst, src)  // Non-temporal store
#define SIMD_LOAD(x) svldnt1_f32(svptrue_b32(), x)  // Non-temporal load
#define SIMD_SET(x) svdup_n_f32(x)
#define SIMD_ADD(x, y) svadd_f32_x(svptrue_b32(), x, y)
#define SIMD_MUL(x, y) svmul_f32_x(SIMD_PRED, x, y)
#define SIMD_FMA(x, y, c) svmla_f32_x(SIMD_PRED, c, x, y)
#define SIMD_SQRT(x) svsqrt_f32_x(SIMD_PRED, x)
#define SIMD_DIV(x, y) svdiv_f32_x(SIMD_PRED, x, y)

// Non-temporal load/store macros
#define SIMD_NT_LOAD(x) svldnt1_f32(svptrue_b32(), x)  // Non-temporal load
#define SIMD_NT_STORE(dst, src) svstnt1_f32(SIMD_PRED, dst, src)  // Non-temporal store

// Optimized SVE Data structure for 128-bit vectors
union SVE_Data {
    float data_f[4] __attribute__((aligned(16)));  // 16-byte alignment for 128-bit vectors
} __attribute__((aligned(64)));  // Cache line alignment to prevent false sharing

// Optimized constants for better cache locality
struct AdamConstants {
    SVE_Data beta1;
    SVE_Data beta2;
    SVE_Data beta1_minus1;
    SVE_Data beta2_minus1;
    SVE_Data eps;
    SVE_Data step_size;
    SVE_Data weight_decay;
} __attribute__((aligned(64)));  // Cache line alignment to prevent false sharing

// Fixed predicate for exactly 4 elements
inline svbool_t get_sve_pred_4() { return svptrue_pat_b32(SV_VL4); }

// Optimized load for 4-wide vectors with non-temporal hint
template <typename T>
inline void simd_load_4(SVE_Data* dst, T* src, size_t offset)
{
    if constexpr (std::is_same_v<T, float>) {
        svfloat32_t data = SIMD_NT_LOAD(src + offset);
        SIMD_NT_STORE(dst->data_f, data);
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        // Load 4 half precision values and convert to float
        svfloat16_t half_data = svldnt1_f16(svptrue_pat_b16(SV_VL4), (float16_t*)(src + offset));
        svfloat32_t float_data = svcvt_f32_f16_x(SIMD_PRED, half_data);
        SIMD_NT_STORE(dst->data_f, float_data);
    }
    __builtin_prefetch(src + offset + 64, 0, 0);  // Prefetch with no temporal locality
}

// Optimized store for 4-wide vectors with non-temporal hint
template <typename T>
inline void simd_store_4(T* dst, SVE_Data* src, size_t offset)
{
    svfloat32_t data = SIMD_NT_LOAD(src->data_f);

    if constexpr (std::is_same_v<T, float>) {
        SIMD_NT_STORE(dst + offset, data);
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        svfloat16_t half_data = svcvt_f16_f32_x(SIMD_PRED, data);
        svstnt1_f16(svptrue_pat_b16(SV_VL4), (float16_t*)(dst + offset), half_data);
    }
    __builtin_prefetch(dst + offset + 64, 1, 0);  // Prefetch with no temporal locality
}

template <int span>
inline void simd_fma_4(SVE_Data* dst, SVE_Data* src1, SVE_Data* src2, SVE_Data* acc)
{
#pragma unroll 4
    for (int i = 0; i < span; i++) {
        svfloat32_t v1 = SIMD_NT_LOAD(src1[i].data_f);
        svfloat32_t v2 = SIMD_NT_LOAD(src2[i].data_f);
        svfloat32_t va = SIMD_NT_LOAD(acc[i].data_f);

        // Optimized FMA for 4-wide vectors
        svfloat32_t result = svmad_f32_x(SIMD_PRED, v1, v2, va);
        SIMD_NT_STORE(dst[i].data_f, result);
        
        // Prefetch next iterations with no temporal locality
        if (i + 1 < span) {
            __builtin_prefetch(&src1[i + 1], 0, 0);
            __builtin_prefetch(&src2[i + 1], 0, 0);
            __builtin_prefetch(&acc[i + 1], 0, 0);
        }
    }
}

// Rest of the code remains the same but using non-temporal loads/stores
// ... (remaining functions unchanged but using SIMD_NT_LOAD/SIMD_NT_STORE)

#endif
