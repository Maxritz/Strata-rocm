# DeepSeek-V4 / deepseek2 — port plan (concrete first steps)

Companion to `ROCM_PORTING.md` §14 (the staged plan) and `MODEL_SUPPORT.md` (the family map). This file is
the "start here" so a fresh session does not re-derive it.

---

## 1. Where we are

`strata --model-info` **recognises** the family and prints its geometry:

| GGUF (in `G:\More-models`) | arch | geometry |
|---|---|---|
| `DeepSeek-V4-Flash-0731-K160-IQ2XXS-…-imatrix.gguf` | `deepseek4` | 43L · embd 4096 · MLA 64/1 · kv 512 · q_lora 1024 · out_lora 1024 ×8 · MoE 6 × 256 (used 6, shared 1, ff 2048) · hyper-connections 4 · engram 3 |
| `DeepSeek-V4-Flash-0731-K160-REAP-MTP-…-ROCm.gguf` | `deepseek4-dspark` | same base + MTP/REAP |
| `GLM-4.7-Flash-*.gguf` | `deepseek2` | 47L · embd 2048 · MoE 4 × 64 |

**Not runnable yet** — this build runs `qwen4exp`. The whole attention tower is what's missing.

**Reused as-is from `qwen4exp`:** the hyper-connection block (count 4), the MoE plumbing (routing → grouped
expert matmul → combine), the sampler, and the batching/prefill engine. **New:** MLA attention, the
compressed/sparse (DSA indexer) selection, and the Engram hash layers.

---

## 2. The reference implementation (this is why it is a *port*, not a write)

[`antirez/ds4`](https://github.com/antirez/ds4) — DwarfStar 4 — is a model-specific DeepSeek-V4-Flash runtime.
It is **not in this workspace**; fetch it first:

```sh
git clone https://github.com/antirez/ds4            # CUDA runtime + parity vectors
git clone https://github.com/Anemll/ds4-ssd         # SSD-streaming fork (sidecar / slot bank)
```

What each gives us:

| ds4 path | use |
|---|---|
| `ds4_cuda.cu` | the DS4 kernels in CUDA — port to HIP on gfx1201 the same way `src/kernels/cuda` was ported |
| `tests/test-vectors/official.vec` + `*.official.json` | **the parity gate** — DS4's own correctness vectors, so each stage is verifiable without hand-building a reference |
| `gguf-tools/deepseek4-quantize.c` + `quants.[ch]` | the DS4 quantizer (q8_0 / q8_K / q4_K / q2_K / iq2_xxs) |
| `docs/SIDECAR.md`, `--moe-slot-bank`, `--ssd-cache` | the lazy expert paging through a bounded resident slot bank |
| `gguf-tools/qwen4_exp_convert.py` (+ `qwen4_iq2.py`) | converts to the `qwen4exp` schema DS4 loads (MTP as `blk.<n>.nextn.*`) — useful for the pack step |

---

## 3. Stage order (each has a hard gate; correctness before speed)

1. **S1 — recognition.** *Done* (`--model-info`).
2. **S2 — the parity harness.** Clone ds4, build its test, run `official.vec`. Then produce our pack from
   the GGUF and the metadata mapping (our `ModelGeometry` ← `deepseek4.*` keys). **Gate: the harness runs and
   the GGUF's tensors all map to a named slot** — nothing numerical yet.
3. **S3 — the layer graph.** A `Ds4Layer` wiring: RMSNorm → MLA projections (with the q/kv LoRA ranks) →
   (S5 attention) → MLA output (8 groups, out_lora 1024) → hyper-connection residual → sigmoid-gated MoE
   (256/6/1, ff 2048, weights_norm, scale 1.5, swiglu_clamp_exp 10) → Engram (first 3). **Gate: per-layer
   logits vs `official.vec`.**
4. **S4 — the MLA attention kernel.** Start *naive and correct* (materialise absorbed `q` and the latent KV,
   standard softmax) — no absorption, no flash. Verify. Then add partial RoPE (64 dims, yarn ×16) and the
   compressed-KV cache for the `compress_ratio 128` layers.
5. **S5 — sparse / compressed attention.** `compress_ratios [0,0,4,128,4,128,…]`, sliding window 128, the
   DSA indexer (head_count 64, key_length 128, top_k 512). This is what makes 1M context tractable.
6. **S6 — Engram + MTP + tuning.** Engram hash layers (first 3) reuse the PLE/n-gram ingestion already in
   `src/kernels/ngram.cpp`; the MTP head reuses `--mtp`; then gfx1201 tuning (MMA shape, LDS 64 KB/block,
   the PCIe budget, residency).

**Effort:** S3 is a session; S4-S5 are the project (MLA + sparse attention is a kernel suite); S6 follows.
`deepseek2` (GLM) shares the MoE half and is a smaller sibling.

---

## 4. What NOT to re-derive

- The DSA/MLA **math** is in ds4 — read it, don't invent it.
- The **parity vectors** already exist (`official.vec`) — use them as the gate, not a hand-built reference.
- The **hyper-connection** block is already in this engine (`qwen4exp`) — reuse it, don't re-port it.

---

## 5. deepseek2 / GLM-4.7-Flash MLA — the EXACT forward (verified, not invented)

Source of record: the vendored llama.cpp at `build_gfx1201/_deps/strata_llamacpp-src/src/models/deepseek2.cpp`
(shape creation lines 99-119, the attention graph lines 246-347). This is the same math ds4 ports, so a fresh
session implements from this instead of fetching ds4 first.

**Shapes** (from `deepseek2.cpp:105-117` and the GGUF, which match):
`q_lora_rank = 768`, `kv_lora_rank = 512`, `n_head = 20`, `n_embd_head_k = 256`, `n_rot = 64`
⇒ `nope = 192`, `v = 256`. Tensors: `wq_a [2048,768]`, `wq_b [768,5120]=[768,20*256]`,
`wkv_a_mqa [2048,576]=[2048,512+64]`, `wk_b [192,512,20]`, `wv_b [512,256,20]`, `wo [5120,2048]`.

**Per layer (single token; the graph is over n_tokens):**
1. `q  = cur @ wq_a` → `rmsnorm(q, attn_q_a_norm)` → `q = q @ wq_b` → `[20, 256]`, split `[nope 192 | rope 64]`.
2. `kv_pe = cur @ wkv_a_mqa` → `[latent 512 | rope 64]`; `kv_cmpr = rmsnorm(latent, attn_kv_a_norm)`.
3. `q_pe = rope(q[...,192:256], pos)`, `k_pe = rope(kv_pe, pos)` — NeoX pairs, `freq_base 1e7`.
4. **absorption:** `q_nope_absorbed = wk_b @ q_nope` (192 → 512, per head).
5. `Qcur = concat(q_nope_absorbed[512], q_pe[64])` per head; `Kcur = concat(kv_cmpr[512], k_pe[64])`;
   `Vcur = kv_cmpr[512]`.
6. `attn = softmax(Qcur·Kcur / sqrt(n_embd_head_k)) @ Vcur`, then V is **up-projected by `wv_b`** (512→256/head).
7. `out = attn @ wo`.

**MoE** (`deepseek2.cpp:363-392`): `build_moe_ffn(gate_inp, up_exps, gate_exps, down_exps, exp_probs_b, 64, 4,
SILU, weights_norm, scale, gating_func)` — the router carries the **`exp_probs_b` bias** and a `gating_func`
(softmax/sigmoid); then **+ the shared expert** `ffn_{up,gate,down}_shexp` (SILU).

**GGUF → pack:** already handled by `tools/iq_pack.py` (arch-generic since this session): `blk.1..46.*_exps`,
64 experts, `n_embd 2048`, `n_ff 1536`; layer 0 is dense.  `attn_k_b`/`attn_v_b` are 3-D and served natively.

---

## 6. The MLA projections' weight format — RESOLVED

`mla_layer`'s projections can use **`native_mmvq`** (`include/strata/kernels/native_mmvq.hpp`), which already
decodes the raw GGUF blocks this fork needs: **Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210, IQ4_XS=136 bytes per 256
elements** (pinned to llama.cpp `3cf0325`), plus Q2_0/Q4_0/Q5_0/Q8_0/IQ4_NL.  So the MLA projections are a
composition, not a new kernel: `native_mmvq` for `wq_a`/`wq_b`/`wkv_a_mqa`/`wk_b`/`wv_b`/`wo`,
`native_rope_apply` (head_dim 256, n_rot 64) for the partial RoPE, `rmsnorm` for `attn_q_a_norm`/
`attn_kv_a_norm`, and a standard attention for the (naive, non-absorbed) core.  GGUF order is n_in contiguous /
n_out rows, which matches the projection shapes in §5.

The same holds for every new family's K-quant projections (M1 qwen35moe Q4_K, M3 k2-horizon Q4_K, laguna
IQ4_XS): no canonicalization pass is required.

---

## 7. Progress

- **M2 step 4 — the MLA forward is implemented and parity-green (2026-10-01).**  `mla_layer`
  (`src/core/layer.cpp`, declared in `include/strata/core/layer.hpp`) composes the §5 ABSORBED forward from
  `native_mmvq`, `rms_norm_weighted` and three new core kernels (`src/kernels/cuda/mla.cu`:
  `mla_split_q`/`mla_rope`/`mla_write_kv`/`mla_attention`).  The ABSORBED form is the only one the packed
  weights admit: `attn_k_b` is stored `[nope, kv_lora, n_head]` with `nope` contiguous, so `native_mmvq` can
  contract it in the `nope -> kv_lora` direction and no other; the unabsorbed `k = latent @ wk_b` would need a
  transpose the raw GGUF blocks do not admit without a canonicalization pass (§6 rules that out).
- **Evidence.**  `build_gfx1201\mla_parity.exe --selftest` (gfx1201): core kernels bit-exact for the split and
  the KV write, `mla_rope` 3.9e-6 and `mla_attention` 2.7e-7 (both vs a double reference, term-relative); the
  WHOLE `mla_layer` over 3 positions vs a double reference that models the Q8_0 weights and the Q8_1 activation
  quantization: **worst L1 8.6e-4** (the contract's G-COH bar is 1e-3).  The residual is the chained projections
  each re-quantizing their own input, not the core math.
- **M2 step 5 — the GLM pack LOADS (2026-10-01).**  `NativeDense` now serves the MLA projections and the dense
  layer-0 FFN (`eligible`), and a 3-D quantized tensor is uploaded whole with the per-head stride
  `native_mmvq_weight_bytes(type, ne0, ne1)` (`attn_k_b` is `[192, 512, 20]`); `check_architecture` accepts
  `deepseek2` with its own required keys.  Evidence: `build_gfx1201\native_dense_3d_test.exe` (gfx1201) ->
  **PASS**, 423 native weights / 1310.4 MiB, `attn_k_b` Q8_0 `[192,512]`, `attn_v_b` Q6_K `[512,256]`.
- **Still OPEN for M2:** the deepseek2 MoE variant (sigmoid gating + `exp_probs_b` selection bias, top-4,
  weights-norm + scale 1.8, the ungated shared expert), the dense layer-0 FFN composition, and the
  decode/prefill wiring from `generate.cpp`/`session.cpp`.



