// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include <deepspeed_aio_common.h>
#include <stdlib.h>
#include <torch/extension.h>

#define TILE (1024 * 1024 * 1024)

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16
#else
#if defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
#endif
#endif

#if defined(__ARM_FEATURE_SVE)
// Predicate for full vector operations
static inline svbool_t get_pred() { return svptrue_b32(); }
// Helper macros with predication handled internally
#define SIMD_STORE(a, d) svst1_f32(get_pred(), a, d)
#define SIMD_LOAD(x) svld1_f32(get_pred(), x)
#define SIMD_SET(x) svdup_n_f32(x)
#define SIMD_MUL(x, y) svmul_f32_x(get_pred(), x, y)
#define SIMD_FMA(x, y, c) svmad_f32_x(get_pred(), x, y, c)
#define SIMD_SQRT(x) svsqrt_f32_x(get_pred(), x)
#define SIMD_DIV(x, y) svdiv_f32_x(get_pred(), x, y)
#define SIMD_WIDTH 16
#endif

int deepspeed_py_memcpy(torch::Tensor& dest, const torch::Tensor& src);
