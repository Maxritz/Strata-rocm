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
