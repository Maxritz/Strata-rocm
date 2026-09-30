// include/strata/kernels/ds2_moe.hpp - the deepseek2 / GLM gating, which differs from qwen4exp's router.
//
// From `build_moe_ffn` (vendored llama.cpp llama-graph.cpp) with `expert_gating_func = SIGMOID` (the GLM GGUF's
// `deepseek2.expert_gating_func = 2`):
//
//     probs           = sigmoid(logits)
//     selection_probs = probs + exp_probs_b          (the DeepSeek-V3 selection bias; probs stay unbiased)
//     ids             = top-k(selection_probs), stable descending, ties by ascending index
//     weights         = probs[ids]
//     weights        /= max(sum(weights), 2^-14)     (expert_weights_norm)
//     weights        *= scale                        (expert_weights_scale, 1.8 for GLM)
//
// qwen4exp's router is softmax over all experts with no bias, so this is a separate entry point rather than a flag
// on it.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// One block per token.  `logits` is (n_tokens, n_expert) f32; `bias` is n_expert f32 or null; `ids`/`weights`
/// are (n_tokens, k).  `n_expert` <= 4096.  `stream` may be null (the call synchronizes).
void ds2_router(const float* logits, const float* bias, int n_tokens, int n_expert, int k, float scale, bool norm,
                int32_t* ids, float* weights, void* stream);

/// `out[i] = silu(gate[i]) * up[i]`, one thread per element.  `out` may equal `gate`.
void swiglu_mul(const float* gate, const float* up, float* out, int64_t n, void* stream);

}  // namespace strata::kernels
