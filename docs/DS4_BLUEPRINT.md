# DwarfStar 4 (`ds4`) — the MoE-streaming blueprint

Cloned at `C:\Users\rr\OneDrive\Desktop\AMDS\ds4` (`git clone https://github.com/antirez/ds4`). It is a
**model-specific** engine (not a general GGUF runner) for **DeepSeek V4 / V4.1 Flash, GLM 5.3/5.2, and
Qwen3.8 Flash Next** — and it already has a **ROCm/HIP port** and **SSD expert streaming**. Everything the
Strata DeepSeek/GLM work needs is here; this file is the map.

---

## 1. The ROCm kernel suite (`rocm/`, 24 files) — the part we would port

A complete HIP implementation of the whole tower, already split for our kind of tree:

| area | files |
|---|---|
| attention (MLA) | `ds4_rocm_attention.cuh`, `ds4_rocm_attention_launch.cuh` |
| sparse / DSA | `ds4_rocm_indexer.cuh`, `ds4_rocm_compressor.cuh` |
| MoE | `ds4_rocm_moe.cuh`, `ds4_rocm_moe_launch.cuh`, `ds4_rocm_router.cuh` |
| shared expert | `ds4_rocm_shared_expert.cuh` |
| hyper-connections | `ds4_rocm_hc.cuh`, `ds4_rocm_hc_output_launch.cuh` |
| GLM | `ds4_rocm_glm.cuh` |
| KV (FP8) | `ds4_rocm_fp8_kv.cuh`, `ds4_rocm_fp8_kv_launch.cuh` |
| matmul / q8 | `ds4_rocm_matmul.cuh`, `ds4_rocm_q8.cuh`, `ds4_rocm_hipblaslt.cuh` |
| norm / rope / embed / out | `ds4_rocm_norm_rope.cuh`, `ds4_rocm_embedding_launch.cuh`, `ds4_rocm_output.cuh` |
| glue | `ds4_rocm_common.cuh`, `ds4_rocm_runtime.cuh`, `ds4_rocm_misc_launch.cuh`, `ds4_rocm_current_api_compat.cuh`, `ds4_rocm_deepseek4_vision.cuh` |

So the DeepSeek port stops being "write MLA from scratch" (docs/DEEPSEEK.md) and becomes **port these
kernels the way `src/kernels/cuda` was ported for Strata** — exactly the workflow this repo already knows.

---

## 2. SSD streaming — the design to copy (docs/SSD_STREAMING.md)

- **Bounded expert cache + read the misses from the GGUF.** *"SSD streaming keeps a bounded cache of routed
  experts and reads missing experts from the GGUF."* That is our `RingExpertSource` — but here it is the
  finished, working version.
- **Two ways to size the budget:** a **byte budget** (`--ssd-streaming-cache-experts 32GB`) or a **slot
  count** (`... 4000`). *"A byte budget is a target, not a guaranteed allocation"* — it reserves
  routed-prefill headroom and fits the cache to the remaining model/graph/context budget.
- **Allocation policy:** by default it spends the budget on **selected experts across all layers**;
  `--ssd-streaming-full-layers N` reserves **full routed prefix layers** instead.
- **Non-routed weights must stay resident.** *"they are needed by every token, and paging them back in
  delays generation after a prompt. An oversized expert cache can displace those weights and slow
  decoding."* — the exact failure mode we must guard (VRAM cache vs dense weights).
- **Prefill vs decode:** *"Generation is usually more sensitive to cache misses than prefill."*
- **Wide-batch prefill with overlap:** *"Large SSD prefills process layers in wide batches. Metal overlaps
  computation with the next layer's reads; CUDA stages experts into its bounded device cache."* — our
  large-chunk + streamed-ring idea, with the overlap that our naive attempt got wrong.
- **Engram rows** (*DeepSeek's 189 GiB hash table*) are *"read directly from the file as needed, never
  loaded as a resident table"* — the same treatment our PLE n-gram table gets.

---

## 3. The precomputed expert hotlist (`ds4_streaming_hotlist*.inc`)

`ds4_streaming_hotlist.inc` (205 KB) + `_glm52` (100 KB) are **generated, static hotlists**:

```c
/* Generated from ds4 expert hotlist profiles; sorted by hits/weight. */
static const uint16_t ds4_default_streaming_hotlist_pro[][2] = { {44, 213}, {25, 315}, {56, 253}, ... };
```

Pairs sorted by `hits/weight`. That is the **admission order** for the bounded cache — computed offline
from profiles and compiled in, rather than derived at runtime from `--expert-profile` the way Strata does.
Adopting this shape (a generated, sorted hotlist) is the concrete upgrade for our `--expert-profile`
admission and the frontier the **RCO** paper (docs/EXPERT_RESIDENCY_FINDINGS.md §10d) would optimise.

---

## 4. Measured streaming (docs/SSD_STREAMING.md, M5 Max 128 GB)

| Model | Initial prefill | Continued prefill | Generation |
|---|---:|---:|---:|
| GLM 5.3 Flash Q4_K, 177.77 GiB | 121 t/s | 104 t/s | 11.9 / 14.9 t/s |
| DeepSeek Flash Vision Exp MXFP4, 145.26 GiB | 300 t/s | 263 t/s | 11.9 / 19.3 t/s |

Also: *"Full GLM 5.3 IQ2_XXS (196.58 GiB) … with an 8K context and a 61.35 GiB effective expert cache, a
16-token append fell from 30.8 to 2.9 seconds"* — cache reuse is the lever, and hard numbers exist to
compare against.

---

## 5. Model support (ds4) and how it maps to our gaps

| family | ds4 support | our status |
|---|---|---|
| DeepSeek V4 / V4.1 Flash | Metal, CUDA, **ROCm** (resident MXFP4 path), SSD streaming | `--model-info` recognises; no decoder |
| GLM 5.3 Flash / 5.2 (full) | Metal, CUDA, **ROCm incl. streaming** | `deepseek2` recognised; no decoder |
| Qwen3.8 Flash Next | Metal, single-GPU CUDA (+MTP, vision) | **runs** (`qwen4exp`) — ds4's `docs/QWEN38_FLASH_NEXT.md` and `ds4_qwen4_cuda.cuh` are a **cross-check for our encoder** |
| Qwen4 packs | `gguf-tools/qwen4_pack*.py`, `qwen4_exp_convert.py`, `qwen4_iq2.py`, `qwen4_native_ngrams.py` | our `iq_pack.py` — compare before re-inventing |

**DSpark** (our `deepseek4-dspark`): ds4's speculative drafter (`docs/SPECULATIVE_DECODING.md`), a
Markov/block drafter — not a model.

---

## 6. The realistic plan (what to do, in order)

1. **Port the ROCm kernel suite** (`rocm/*.cuh`) for the attention + MoE it covers, starting with the
   DeepSeek attention/indexer pair, verified against `tests/test-vectors/official.vec`. This is the
   multi-session core and it now has a complete reference *and* a parity gate.
2. **Adopt the streaming policy wholesale:** byte-budget cache, cross-layer expert selection, full-prefix
   layers, non-routed weights pinned in VRAM, wide-batch prefill with read/compute overlap. This is the
   direct fix for our nondeterministic `RingExpertSource` — ds4's version is the correctness reference.
3. **Move the hotlist to a generated artifact** (offline profile → sorted `uint16[layer,expert]` table),
   replacing runtime `--expert-profile` ordering; then let **RCO** optimise it under a byte budget.
4. **Reuse `gguf-tools/qwen4_pack*.py`** rather than extending `iq_pack.py` where it overlaps.
