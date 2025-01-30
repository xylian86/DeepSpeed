// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <torch/extension.h>
#include <cassert>
#include "simd.h"
#include "sve.h"

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

// Add prefetch hints to avoid cache pollution
// Use PREFETCH_HINT_NTA (0) to bypass L3 cache on Grace CPU
#define PREFETCH_HINT_T0 0  // Changed from 3 to 0 to bypass L3
#define PREFETCH_HINT_NTA 0

#define STEP(SPAN)                                                           \
    template <typename ds_params_precision_t, typename ds_state_precision_t> \
    void Step_##SPAN(ds_params_precision_t* _params,                         \
                     ds_params_precision_t* grads,                         \
                     ds_state_precision_t* _exp_avg,                         \
                     ds_state_precision_t* _exp_avg_sq,                      \
                     size_t _param_size);

class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0,
                   bool adamw_mode = true)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _step(0),
          _adamw_mode(adamw_mode)
    {
        // Prefetch constants using non-temporal hint to bypass L3 cache
        __builtin_prefetch(&_alpha, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(&_betta1, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(&_betta2, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(&_eps, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(&_weight_decay, 0, PREFETCH_HINT_NTA);
    }
    ~Adam_Optimizer() {}
#if defined(__ARM_FEATURE_SVE)
    template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
    void Step_SVE(size_t* rounded_size,
                  ds_params_precision_t* _params,
                  ds_params_precision_t* grads,
                  ds_state_precision_t* _exp_avg,
                  ds_state_precision_t* _exp_avg_sq,
                  size_t param_size);
#endif
#if defined(__AVX512__) or defined(__AVX256__)
    template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
    void Step_AVX(size_t* rounded_size,
                  ds_params_precision_t* _params,
                  ds_params_precision_t* grads,
                  ds_state_precision_t* _exp_avg,
                  ds_state_precision_t* _exp_avg_sq,
                  size_t param_size);
#endif
    STEP(1)
    STEP(4)
    STEP(8)
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;
            _betta1 = beta1;
            _betta2 = beta2;
            _betta1_t = std::pow(_betta1, step);
            _betta2_t = std::pow(_betta2, step);
        } else {
            _step++;
            if (_step != step) {
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
            } else {
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            }
        }
    }
    inline void update_state(float lr, float epsilon, float weight_decay, bool bias_correction)
    {
        _alpha = lr;
        _eps = epsilon;
        _weight_decay = weight_decay;

        _bias_correction1 = 1.0f;
        _bias_correction2 = 1.0f;
        if (bias_correction == 1) {
            _bias_correction1 = 1 - _betta1_t;
            _bias_correction2 = 1 / sqrt(1 - _betta2_t);
        }
    }

private:
    // Align data to cache line boundary to avoid false sharing
    alignas(64) float _alpha;
    alignas(64) float _betta1;
    alignas(64) float _betta2;
    alignas(64) float _eps;
    alignas(64) float _weight_decay;

    alignas(64) float _betta1_t;
    alignas(64) float _betta2_t;
    alignas(64) size_t _step;

    alignas(64) float _bias_correction1;
    alignas(64) float _bias_correction2;

    alignas(64) bool _adamw_mode;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              ds_params_precision_t* _params,
                              ds_params_precision_t* grads,
                              ds_state_precision_t* _exp_avg,
                              ds_state_precision_t* _exp_avg_sq,
                              size_t _param_size)
{
#if !defined(__AVX512__)
    if (std::is_same_v<ds_params_precision_t, c10::BFloat16> ||
        std::is_same_v<ds_state_precision_t, c10::BFloat16>) {
        return;
    }
#endif
    size_t new_rounded_size = 0;

    // Prefetch data using non-temporal hint to bypass L3 cache
    for (size_t i = 0; i < _param_size; i += 64) {
        __builtin_prefetch(_params + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(grads + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(_exp_avg + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(_exp_avg_sq + i, 0, PREFETCH_HINT_NTA);
    }

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + i);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i);

            AVX_Data variance_4[span];
            simd_load<span>(variance_4, _exp_avg_sq + i);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + i);

            if (_weight_decay > 0 && !_adamw_mode) {
                simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
            }

            simd_mul<span>(momentum_4, momentum_4, betta1_4);
            simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
            simd_mul<span>(variance_4, variance_4, betta2_4);
            simd_mul<span>(grad_4, grad_4, grad_4);
            simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
            simd_sqrt<span>(grad_4, variance_4);
            simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);

            if (_weight_decay > 0 && _adamw_mode) {
                simd_fma<span>(param_4, param_4, weight_decay4, param_4);
            }

            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            // Use non-temporal stores to bypass cache
            simd_store<span>(_params + i, param_4);
            simd_store<span>(_exp_avg + i, momentum_4);
            simd_store<span>(_exp_avg_sq + i, variance_4);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif

#if defined(__ARM_FEATURE_SVE)
template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_SVE(size_t* rounded_size,
                              ds_params_precision_t* _params,
                              ds_params_precision_t* grads,
                              ds_state_precision_t* _exp_avg,
                              ds_state_precision_t* _exp_avg_sq,
                              size_t _param_size)
{
    if (!std::is_same_v<ds_params_precision_t, float> ||
        !std::is_same_v<ds_state_precision_t, float>) {
        return;
    }
    printf("Step_SVE\n");
    *rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);

    // Prefetch data using non-temporal hint to bypass L3 cache
    for (size_t i = 0; i < _param_size; i += 64) {
        __builtin_prefetch(_params + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(grads + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(_exp_avg + i, 0, PREFETCH_HINT_NTA);
        __builtin_prefetch(_exp_avg_sq + i, 0, PREFETCH_HINT_NTA);
    }

    AdamConstants constants __attribute__((aligned(64)));

    // Initialize constants using vectorized operations
    svfloat32_t beta1_vec = svdup_n_f32(_betta1);
    svfloat32_t beta2_vec = svdup_n_f32(_betta2);
    svfloat32_t beta1_m1_vec = svdup_n_f32(1.0f - _betta1);
    svfloat32_t beta2_m1_vec = svdup_n_f32(1.0f - _betta2);
    svfloat32_t eps_vec = svdup_n_f32(_eps);
    svfloat32_t step_vec = svdup_n_f32(-_alpha / _bias_correction1);

    svst1_f32(SIMD_PRED, constants.beta1.data_f, beta1_vec);
    svst1_f32(SIMD_PRED, constants.beta2.data_f, beta2_vec);
    svst1_f32(SIMD_PRED, constants.beta1_minus1.data_f, beta1_m1_vec);
    svst1_f32(SIMD_PRED, constants.beta2_minus1.data_f, beta2_m1_vec);
    svst1_f32(SIMD_PRED, constants.eps.data_f, eps_vec);
    svst1_f32(SIMD_PRED, constants.step_size.data_f, step_vec);

    if (_weight_decay > 0) {
        float w_decay = _adamw_mode ? -1 * _alpha * _weight_decay : _weight_decay;
        svst1_f32(SIMD_PRED, constants.weight_decay.data_f, svdup_n_f32(w_decay));
    }

    const float bias_correction2_float = _bias_correction2;

#pragma omp parallel for
    for (size_t t = 0; t < *rounded_size; t += TILE) {
        const size_t current_size = ((*rounded_size - t) < TILE) ? (*rounded_size - t) : TILE;

#pragma unroll VECTOR_UNROLL
        for (size_t i = t; i < t + current_size; i += SIMD_WIDTH * VECTOR_UNROLL) {
            SVE_Data params_data[VECTOR_UNROLL], grads_data[VECTOR_UNROLL];
            SVE_Data exp_avg_data[VECTOR_UNROLL], exp_avg_sq_data[VECTOR_UNROLL];

#pragma unroll VECTOR_UNROLL
            for (int j = 0; j < VECTOR_UNROLL; j++) {
                const size_t offset = i + j * SIMD_WIDTH;
                if (offset < t + current_size) {
                    // Use non-temporal loads to bypass cache
                    svfloat32_t params_vec = SIMD_NT_LOAD(_params + offset);
                    svfloat32_t grads_vec = SIMD_NT_LOAD(grads + offset);
                    svfloat32_t exp_avg_vec = SIMD_NT_LOAD(_exp_avg + offset);
                    svfloat32_t exp_avg_sq_vec = SIMD_NT_LOAD(_exp_avg_sq + offset);
                    
                    // Use streaming stores to bypass all cache levels and write directly to memory
                    SIMD_NT_STORE(params_data[j].data_f, params_vec);
                    SIMD_NT_STORE(grads_data[j].data_f, grads_vec);
                    SIMD_NT_STORE(exp_avg_data[j].data_f, exp_avg_vec);
                    SIMD_NT_STORE(exp_avg_sq_data[j].data_f, exp_avg_sq_vec);
                }
            }

#pragma unroll VECTOR_UNROLL
            for (int j = 0; j < VECTOR_UNROLL; j++) {
                const size_t offset = i + j * SIMD_WIDTH;
                if (offset >= t + current_size) continue;

                // Load vectors from local data using non-temporal loads
                svfloat32_t params_vec = SIMD_NT_LOAD(params_data[j].data_f);
                svfloat32_t grads_vec = SIMD_NT_LOAD(grads_data[j].data_f); 
                svfloat32_t momentum = SIMD_NT_LOAD(exp_avg_data[j].data_f);
                svfloat32_t variance = SIMD_NT_LOAD(exp_avg_sq_data[j].data_f);

                // Apply weight decay if needed (fused operation)
                if (_weight_decay > 0 && !_adamw_mode) {
                    svfloat32_t weight_decay_vec = SIMD_NT_LOAD(constants.weight_decay.data_f);
                    grads_vec = svmad_f32_x(SIMD_PRED, params_vec, weight_decay_vec, grads_vec);
                }

                // Update momentum (fused multiply-add)
                svfloat32_t beta1_vec = SIMD_NT_LOAD(constants.beta1.data_f);
                svfloat32_t beta1_m1_vec = SIMD_NT_LOAD(constants.beta1_minus1.data_f);
                momentum = svmad_f32_x(SIMD_PRED, momentum, beta1_vec,
                                     svmul_f32_x(SIMD_PRED, grads_vec, beta1_m1_vec));

                // Update variance (fused multiply-add)
                svfloat32_t beta2_vec = SIMD_NT_LOAD(constants.beta2.data_f);
                svfloat32_t beta2_m1_vec = SIMD_NT_LOAD(constants.beta2_minus1.data_f);
                svfloat32_t grad_sq = svmul_f32_x(SIMD_PRED, grads_vec, grads_vec);
                variance = svmad_f32_x(SIMD_PRED, variance, beta2_vec,
                                     svmul_f32_x(SIMD_PRED, grad_sq, beta2_m1_vec));

                // Compute update
                svfloat32_t sqrt_variance = svsqrt_f32_x(SIMD_PRED, variance);
                svfloat32_t denom = svadd_f32_x(SIMD_PRED,
                    svmul_f32_x(SIMD_PRED, sqrt_variance, svdup_n_f32(bias_correction2_float)),
                    SIMD_NT_LOAD(constants.eps.data_f));

                svfloat32_t update = svdiv_f32_x(SIMD_PRED, momentum, denom);

                // Apply weight decay in ADAMW mode
                if (_weight_decay > 0 && _adamw_mode) {
                    svfloat32_t weight_decay_vec = SIMD_NT_LOAD(constants.weight_decay.data_f);
                    params_vec = svmad_f32_x(SIMD_PRED, params_vec, weight_decay_vec, params_vec);
                }

                // Update parameters
                svfloat32_t step_vec = SIMD_NT_LOAD(constants.step_size.data_f);
                params_vec = svmad_f32_x(SIMD_PRED, update, step_vec, params_vec);

                // Store results back using non-temporal stores
                SIMD_NT_STORE(params_data[j].data_f, params_vec);
                SIMD_NT_STORE(exp_avg_data[j].data_f, momentum);
                SIMD_NT_STORE(exp_avg_sq_data[j].data_f, variance);
            }

#pragma unroll VECTOR_UNROLL
            for (int j = 0; j < VECTOR_UNROLL; j++) {
                const size_t offset = i + j * SIMD_WIDTH;
                if (offset < t + current_size) {
                    // Use non-temporal stores to bypass cache
                    SIMD_NT_STORE(_params + offset, svld1_f32(SIMD_PRED, params_data[j].data_f));
                    SIMD_NT_STORE(_exp_avg + offset, svld1_f32(SIMD_PRED, exp_avg_data[j].data_f));
                    SIMD_NT_STORE(_exp_avg_sq + offset, svld1_f32(SIMD_PRED, exp_avg_sq_data[j].data_f));
                }
            }
        }
    }
}
#endif

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false);

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq);

int destroy_adam_optimizer(int optimizer_id);
