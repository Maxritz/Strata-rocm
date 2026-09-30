// include/strata/kernels/mla.hpp - the MLA attention core (deepseek2 / GLM-4.7-Flash).
//
// The projections and the RoPE are compositions of kernels that already exist (`native_mmvq`, `rms_norm_weighted`);
// these three are the pieces that do not: splitting the q projection into its nope|rope halves, rotating the pe
// halves, writing the compressed KV cell, and the softmax attention over the absorbed latent.
//
// THE FORWARD IS `docs/DEEPSEEK.md` section 5, transcribed from the vendored llama.cpp `deepseek2.cpp`.  Only the
// ABSORBED form is compatible with the packed weights: `attn_k_b` is stored `[nope, kv_lora, n_head]` with nope
// contiguous, so `native_mmvq` can only contract it in the `nope -> kv_lora` direction (the q absorption).  The
// unabsorbed `k = latent @ wk_b` would need a transpose the raw GGUF blocks do not admit without a canonicalization
// pass, which `docs/DEEPSEEK.md` section 6 rules out.
//
// All pointers are device pointers.  `pos` is one int32 (the sequence position) so the kernel argument stays a
// pointer and a captured graph replays the position it is given rather than baking the first token in.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Split the raw `wq_b` output `[n_head, nope + n_rot]` into the nope half (per head) and the pe half (per head).
void mla_split_q(const float* q, float* q_nope, float* q_pe, int64_t n_head, int64_t nope, int64_t n_rot,
                 void* stream);

/// NEOX partial RoPE over `rows` rows of `n_rot` channels, `n_rot` even.  `pos` is ONE device int32 shared by
/// every row (the compressed-KV rope tail has one head and the q rope half shares the token's position).
/// In-place (`out == x`) is allowed.  `freq_base` is the model's `rope.freq_base`.
void mla_rope(const float* x, float* out, int64_t rows, int64_t n_rot, float freq_base, const int32_t* pos,
              void* stream);

/// Append the current token's compressed KV cell: `kv[pos] = latent | k_pe` (fp16) and `v[pos] = latent` (fp16).
void mla_write_kv(uint16_t* kv, uint16_t* v, const float* latent, const float* k_pe, int64_t pos, int64_t n_lora_kv,
                  int64_t n_rot, void* stream);

/// Softmax attention over the absorbed latent.  `q_abs` is `[n_head, n_lora_kv]`, `q_pe` is `[n_head, n_rot]`,
/// `kv` is `[n_cells, n_lora_kv + n_rot]` fp16 and `v` is `[n_cells, n_lora_kv]` fp16.  `scores` is caller scratch
/// of `n_head * n_cells` floats.  The output `attn` is `[n_head, n_lora_kv]` f32.
void mla_attention(const float* q_abs, const float* q_pe, const uint16_t* kv, const uint16_t* v, int64_t n_cells,
                   int64_t n_head, int64_t n_lora_kv, int64_t n_rot, float scale, float* scores, float* attn,
                   void* stream);

}  // namespace strata::kernels
