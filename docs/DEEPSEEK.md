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
- **M2 step 6 — the deepseek2 MoE arithmetic is implemented and parity-green (2026-10-01).**  `ds2_moe.cu` adds
  the SIGMOID router with the `exp_probs_b` selection bias and the `expert_weights_norm`/`scale` epilogue
  (`ds2_router`) and `swiglu_mul`; `layer.cpp` composes the ungated shared expert and the dense layer-0 FFN
  (`moe_route_ds2`/`moe_shared_ds2`/`moe_finish_ds2`/`dense_ffn_ds2`) from `native_mmvq` + `swiglu_mul` +
  `native_quantize_q8_1`.  Evidence: `build_gfx1201\ds2_moe_parity.exe --selftest` -> **PASS**; ids exact and
  weights exact over 8 seeds (scale 1.0/1.8, norm on/off), the bias is shown to reorder the selection, the
  weight is the unbiased sigmoid, and `swiglu_mul` matches.
- **M2 step 7 — the deepseek2 model RUNS (2026-10-01).**  `ds2_token` (`src/core/session.cpp`) composes one
  token's `x += MLA(rms(x))` / `x += FFN(rms(x))` over all 47 layers; `session_bytes`/`session_init`/
  `session_zero` allocate the per-layer `MlaState` compressed-KV caches (plus the MLA scratch, the MoE buffers and
  four n_embd intermediates) behind `g.mla != 0`, so the qwen4exp path is byte-for-byte unchanged.
  `mla_layer_weights` (layer.cpp) resolves the six native projections by name; the MoE half is the layer-0 dense
  FFN (`dense_ffn_ds2`) or the routed+shared sigmoid MoE (`moe_route_ds2`/`moe_shared_ds2`/`moe_finish_ds2`).
  `generate.cpp` reads the geometry under `deepseek2.*` (`key_length_mla`, `q/kv_lora_rank`, `rope.dimension_count`,
  `expert_weights_norm/scale`, `leading_dense_block_count`) and dispatches to a dedicated `run_deepseek2` before
  any qwen4exp-only validation; the qwen4exp guard is relaxed only for deepseek2.
  The CPU expert pool gained a native single-token dispatch (`native_expert_pool_dispatch`) that uses the layer's
  own `n_embd`/`n_ff` instead of the Q2_0 artifact's 2560/640 (GLM is 2048/1536; the pool's hard-coded `H`/`FF`
  refused it).
  **Evidence (gfx1201), command:**
  `build_gfx1201\strata.exe --pack H:\OLLAMA-Models\strata-pack-glm --native H:\OLLAMA-Models\GGUF\GLM-4.7-Flash-APEX-I-Quality.gguf --tokens "..." --max-new 6 --max-context 1024`
  - `The capital of France is` (785,6722,315,9621,374) -> `12089 13 ...` = **" Paris."**
  - `The capital of Germany is` (785,6722,315,9851,374) -> `19808 13 ...` = **" Berlin."**
  - `2+2=` (17,10,17,28) -> `19 ...` = **"4"**
  - decode **12.9-13.0 tok/s**, prefill (token-at-a-time) **~12.1-12.4 tok/s**; expert arena 16.09 GiB.
- **M2 step 8 — PER-TOKEN PROFILE, and two launch-path fixes (2026-10-01).**  `STRATA_DS2_TIMING=1` in
  `ds2_token` reports, per token, the three walls that can be attacked independently: `sync` (the router
  handoff's `hipStreamSynchronize`), `pool` (the CPU expert pool), and `rest` (enqueue-only).  A `--no-pool`
  run measured the device event span **equal to the host enqueue time** (~26 ms of ~940 launches), i.e. the
  token-at-a-time path is **host-enqueue-bound, not GPU-bound**.
  - **Head-batched MLA MMVQ.**  The absorption (`q_abs[h] = wk_b[h] @ q_nope[h]`) and up-projection were two
    per-head loops of `n_head` (quantize + MMVQ) pairs — 80 launches a layer.  The concatenated per-head
    activations are now quantized once (`MlaBuffers::act_q8`), and `native_mmvq_heads` does the `n_head`
    independent products in ONE launch (a `blockIdx.z` head offset over the unchanged single-matrix kernels).
    Bitwise parity: `native_mmvq_heads_parity` → Q8_0/Q6_K, both kernel selections, 0 float bits differ.
    (A Q6_K per-head activation stride of `blocks_per_row` instead of `n_in/32` was caught only by the
    end-to-end token diverging; that is why the direct test exists.)
  - **Pool handoff on the compute queue.**  The three D2H staging memcpys and the H2D upload (four copy-engine
    ops a layer) are replaced by `doorbell_publish` and `copy_from_mapped` kernels on the compute queue.
  - **Measured (gfx1201, the command in the bullet above):** decode **12.46 → 19.05 tok/s**, prefill
    **11.90 → 17.35 tok/s**, output byte-identical (`12089 13 576 6722 315 9621`).  Breakdown per token:
    `sync` 28.3 → 9.5 ms, `rest` 23.0 → 13.0 ms, `pool` 30 ms (unchanged — the CPU pool is DRAM-bandwidth
    bound, so it is the floor: ~33 tok/s even with a free GPU).  Commits `cb87340`, `4269a8b`.
- **Still OPEN for M2:** a BATCHED prefill for deepseek2 (the prompt currently runs one token at a time through
  `ds2_token`; grouping the chunk's distinct experts once per chunk, rather than re-reading 4 experts per token,
  is the large prefill lever), the formal G-COH gate vs a CPU/llama.cpp reference (the greedy continuations
  above are coherent but were not yet scored L1 <= 1e-3 against a reference), a captured deepseek2 token graph
  (to remove the remaining 9.5 ms `sync` + 13 ms `rest`), and decode-60 / prefill-5000 tuning.



