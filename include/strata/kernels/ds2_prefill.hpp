// include/strata/kernels/ds2_prefill.hpp - the batched-prefill companions of the MLA core kernels.
//
// `mla.cu` is built for ONE query: split, rope, write one KV cell, and score one row against the cache.  A
// prompt chunk needs the same operations over T rows at once, which is what lives here.  Every kernel is a
// direct row-wise generalisation of its single-token namesake, so a T = 1 call reproduces the single-token
// result to the bit (the parity test `ds2_prefill_parity` checks exactly that).
#pragma once

#include <cstdint>

namespace strata::kernels::ds2pf {

/// Rotate the `n_rot`-wide slice at `rot_off` of each of `rows` rows.  Row `r` belongs to token
/// `r / heads_per_pos` and uses position `pos0 + r / heads_per_pos`.  `in_stride`/`out_stride` are in floats
/// between rows, so the caller can rotate a slice of a wider row (the `q_pe` half of `wq_b`'s output: rows =
/// T*n_head, in_stride = n_head*head_dim, rot_off = nope) and a whole row (k_pe: rows = T, in_stride =
/// n_lora_kv + n_rot, rot_off = n_lora_kv).  NeoX pairs, `freq_base` the same as `mla_rope`.
void rope_slice(const float* x, float* out, int64_t rows, int64_t in_stride, int64_t out_stride, int64_t rot_off,
                int64_t n_rot, float freq_base, int64_t pos0, int64_t heads_per_pos, void* stream);

/// Gather the `nope`-wide no-rope half of each head of `wq_b`'s output into head-major order: `qnope[h][t][:]`
/// from `q[t][h*head_dim + :nope]`.  Head-major makes each head's T activation rows contiguous, which is what
/// `mmq::quantize`/`Product` want for the per-head absorption.
void gather_qnope(const float* q, float* qnope, int64_t T, int64_t n_head, int64_t nope, int64_t head_dim,
                  void* stream);

/// Row-wise RMSNorm with a row stride: `out[r][c] = in[r*stride + c] * w[c] / rms`.  The MLA latent is the
/// first `cols` of the `n_lora_kv + n_rot`-wide `wkv_a` output, so it is not a contiguous row.
void rms_strided(const float* in, const float* w, float* out, int64_t rows, int64_t cols, int64_t stride,
                 float eps, void* stream);

/// Write the T KV cells `[pos0, pos0+T)` from the layer's latent and rotated rope: `kv = [latent | k_pe]`,
/// `v = latent`, fp16, the same layout `mla_write_kv` writes one cell of.
void write_kv(uint16_t* kv, uint16_t* v, const float* latent, const float* k_pe, int64_t pos0, int64_t T,
              int64_t n_lora_kv, int64_t n_rot, void* stream);

/// Causal MQA attention over a chunk: query `t` attends keys `[0, pos0 + t]`.  `q_abs[t][h]` is `n_lora_kv`
/// wide, `q_pe[t][h]` is `n_rot` wide, the K/V cache is shared across heads (`kv` cells `[n_lora_kv | n_rot]`,
/// `v` cells `n_lora_kv`).  One block per (head, query).  Writes `attn[t][h][:]` (`n_lora_kv` wide).
void attention(const float* q_abs, const float* q_pe, const uint16_t* kv, const uint16_t* v, int64_t pos0,
               int64_t T, int64_t n_head, int64_t n_lora_kv, int64_t n_rot, float scale, float* attn,
               void* stream);

/// `out[t][d] = sum_j weights[t*k + j] * parts[t*k + j][d] + shared[t][d]` (the shared expert added plain, as
/// `moe_combine` does for one token).  `parts` is `(T*k, n_embd)`.
void combine(const float* parts, const float* weights, const float* shared, float* out, int64_t T, int64_t k,
             int64_t n_embd, void* stream);

/// `y = W x` in float for a tiny W (the router's BF16 projection is separate); used to copy a strided source.
void gather_rows(const float* x, float* out, int64_t rows, int64_t cols, int64_t in_stride, void* stream);

/// Bytes of the scratch `attention` needs: one fp32 score row of `max_cells` per (head, query) is avoided by
/// using dynamic shared memory, so this is zero; kept as an explicit contract.
int64_t attention_scratch_bytes();
}  // namespace strata::kernels::ds2pf
