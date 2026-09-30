# Strata MoE Contract — the unwavering goals

Status: binding. Every clause is testable; a clause is DONE only when its **Acceptance** command has been run on
gfx1201 and its output recorded here. No clause ships on an unmeasured claim. No numerics change ships without a
parity gate. This file supersedes optimism: if the acceptance has not been run, the clause is OPEN.

Date opened: 2026-10-01. Engine version at open: 0.1.15.

---

## 0. The invariant goal (does not change)

**Fastest possible coherent inference for ALL supported MoE models.**

Binding acceptance of the goal (all three at once — any one failing fails the goal):

- **[G-COH]** Coherence: **L1 ≤ 1e-3** vs the CPU reference (`expert_pool_dispatch_multi` / llama.cpp top-1
  agreement), on every supported model, every config. A faster token produced by a wrong kernel is a violation.
- **[G-PERF]** Performance on gfx1201: **decode ≥ 60 tok/s**, **prefill ≥ 5000 tok/s** on a ≥ 2048-token chunk.
- **[G-GPU]** Compute on the **GPU/tensor cores**. No expert matmul on the critical path runs on the CPU. The
  CPU may route, quantize activations, and orchestrate — it may not compute experts.

## 1. Non-negotiables (apply to every clause)

- **[N-PARITY]** Every new kernel is parity-gated vs the validated scalar reference (`dequantize_*`, bit-exact
  layout) BEFORE it is wired into a run path. New kernels default OFF until green.
- **[N-MEASURE]** No fabricated numbers. Every performance claim is a measured gfx1201 number with the command.
- **[N-NOSTUB]** No stubs, no placeholders, no "for now". If something cannot be done, it fails loudly.
- **[N-RAM]** Experts stream lazily and boundedly; the hot set is resident in VRAM; host RAM stays bounded.

---

## 2. DONE (measured, committed)

| ID | item | evidence | commit |
|---|---|---|---|
| D1 | FP4 GEMV decode (MXFP4+NVFP4), parity-green | **1.140 TOPS / 10.10x naive**, 606 GB/s (95% peak); parity worst 4.5e-5 | `4380b04` |
| D2 | MTP draft bin built + enabled | decode 17.8 → **37.0 tok/s** (`--spec 4`), drafts 15/18 accepted | `c9cf0cf` |
| D3 | prefill chunk "do not fit" (`set_geometry` before lend) | chunk 2048 now fits (was abort) | `30dd187` |
| D4 | `--spec 6` LDS wall (TILE 2560→1280) | window 8 runs | `30dd187` |
| D5 | Ring transient-source GPU hang | `gather_native` launch-failure gone; prefill 239→459.7 tok/s | `434bf90` |
| D6 | WMMA peak measured | **f16 95.3 / iu4 195.8 T-MAC/s** | `3697691` |

---

## 3. PENDING — KERNELS (the G-GPU / G-PERF path)

### K1 — int4 WMMA expert GEMM  *(highest priority; measured ceiling 196 TOPS)*
Requirement: MoE expert matmul (gate/up/down) on the int4 tensor core via
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` (or the `pk_int4` 16x16x32 form), fragments staged via
ck_tile's `amdgcn_mma` layout (`G:\ROCM10RT-gfx1201\include\ck_tile\core\arch\mma\wmma\`).
- **Acceptance:** (a) `wmma_iu4_parity` green vs a scalar int4 reference on a 16×16×16 tile and a tiled 128×128;
  (b) measured **≥ 40 T-MAC/s** on a realistic tiled GEMM at the expert shape; (c) wired into prefill's MMQ slot
  with G-COH re-run green.
- **K1a DONE (2026-10-01, `eeba8e7`):** 16×16×16 fragment layout (RDNA4 ISA §7.12.2) + signed-int4 MMA —
  parity PASS 0/256.  The fragment-layout risk is retired.
- **K1b DONE (2026-10-01, `bench/micro/wmma_iu4_gemm_lds.cu`):** tiled int4 WMMA GEMM, parity **PASS 0 wrong**.
  Key fix (found by disassembly, not guessed): store tiles **packed** and read each fragment as **one `ds_read_b32`**
  (A row-major packed; B **N-major packed — the K3 layout**).  Staging becomes a pure 128-bit copy; no
  unpack/repack.  15.8 → **52.8 T-MAC/s @2048³, 64.7 @4096³, 55.8 @2048×1280×2560 (expert gate/up, 2048-tok chunk)**
  → **acceptance (b) MET**.  4 accumulators/wave (16×64), BK=64 → 16 MMAs/barrier.
  - Measured: `wmma_iu4_gemm_lds.exe 2048 1280 2560` → 55.8; `… 4096 4096 4096` → 64.7.
  - Small-M still lower (512×2560×2560 = 34.2, 1024×640×2560 = 24.8) — launch/memory-bound; the grouped/persistent
    form (K1c) is the fix, not a dense-tile tweak.
- **K1c OPEN:** wire this into prefill's MMQ slot (acceptance (c), G-COH re-run).  Requires the grouped form:
  pack each expert's weight N-major (K3) and dispatch [concatenated tokens × N × K] per expert.
- **Owner-clause:** do not enable until (a) and (b) pass.

### K2 — MXFP4 / NVFP4 tensor-core path  *(TODO §6: the RDNA4 FP4 path and WMMA are the SAME project)*
Requirement: the FP4 decode (`fp4_gemv_q8`, D1) extended to the batched tensor-core form for
Qwen3.5-35B-A3B / qwen-agentworld-nvfp4 / gpt-oss-120b.
- **Acceptance:** parity vs `dequantize_mxfp4`/`dequantize_nvfp4`; measured TOPS at B=128; G-COH on a MXFP4 model.

### K3 — direct-to-LDS + wide (uint4) loads
Requirement: repack the MXFP4 17-byte block to a 16-byte-aligned layout so `raw_buffer_load_lds` and 128-bit
loads vectorize (the 17-byte stride currently defeats them).
- **Acceptance:** `fp4_gemv_q8` ≥ previous 1.140 TOPS AND the repack is a lossless relayout (byte-parity test).
- **DONE (2026-10-01):** `include/strata/kernels/fp4_repack.hpp` splits each tensor into a SCALE array and a
  **16-byte-aligned DATA array** (SoA) — MXFP4 1+16 B, NVFP4 4+32 B per block, copied verbatim.  `fp4_gemv_q8`
  is untouched (still 1.140 TOPS, contract method).  `bench/micro/fp4_repack_parity.cu` decodes the ggml layout
  and the repacked layout on the device and requires bit-identical floats:
  `fp4_repack_parity.exe 1048576` → **0 wrong values, data 16B-aligned yes, PASS**.
  Follow-on (K2/K4): make the batched kernel read the aligned DATA array with `uint4` loads.

### K4 — fuse gate/up/down into one expert kernel (Mega-MoE)
- **Acceptance:** parity vs the 3-kernel path; fewer launches; measured end-to-end gain.

---

## 4. PENDING — MODEL COVERAGE (the "ALL MoE models" clause)

Each is DONE only when it passes G-COH and reports decode/prefill on gfx1201.

| ID | model | what it needs | gate |
|---|---|---|---|
| M1 | **qwen35moe** (Qwen3.5-35B-A3B, ornith-35b, Tiel-Coder-35B, qwable) | GQA (16/2, 8:1) + interleaved partial-RoPE 64/256 (`[11,11,10,0]`); GDN reuses verbatim | blocked on K2 (MXFP4) |
| M2 | **deepseek2 / GLM-4.7-Flash** | MLA (`kv_lora_rank`), 64/4+1; **no new quant** — cheapest | independent |
| M3 | **k2-horizon**, **laguna**, **gemma4-26B-A4B** | plain GQA MoE / per-layer head lists | after M2 |
| M4 | **DeepSeek-V4-Flash** | MLA + Engram + sparse attention + sigmoid MoE | `DS4_BLUEPRINT.md`; gate on `tests/test-vectors/official.vec` |
| M5 | **nemotron_h_moe, qwen3moe, olmoe** | own decoders | recognised only |
| M6 | **ROCmFPX** (Ornith-1.0-9B) | AMD FP4/FP6/FP8 native on gfx1201 | = K2 |

- **M2 MILESTONE (2026-10-01, NOT the gate): deepseek2/GLM-4.7-Flash RUNS.**  The MLA decoder + sigmoid MoE are
  now wired into a session (`ds2_token`) and dispatched from `generate.cpp` (`run_deepseek2`), token-at-a-time
  (no batched prefill).  gfx1201, `strata.exe` on `strata-pack-glm` + the GLM GGUF: decode **12.9-13.0 tok/s**,
  prefill **~12.1-12.4 tok/s**; greedy continuations are coherent ("The capital of France is" -> " Paris.",
  "The capital of Germany is" -> " Berlin.", "2+2=" -> "4").  `qsa_parity`/`gdn_parity`/`gr_parity`/
  `native_expert_parity`/`mla_parity`/`ds2_moe_parity` all green after the change.  **The gate is still OPEN**:
  G-COH (L1 <= 1e-3 vs a CPU/llama.cpp reference) has not been measured, prefill is one-token-at-a-time, and
  decode/prefill are far below G-PERF.
  - **M2 perf step 8 (2026-10-01, `cb87340` + `4269a8b`): decode 12.46 -> 19.05, prefill 11.90 -> 17.35 tok/s,
    byte-identical output.**  `STRATA_DS2_TIMING=1` (`ds2_token`) + a `--no-pool` control showed the path is
    **host-enqueue-bound** (device event span == host enqueue), not GPU-bound.  Fixes: `native_mmvq_heads` —
    the MLA absorption/up-projection's `n_head` per-head MMVQs in one launch (bitwise; parity
    `native_mmvq_heads_parity`), and the CPU-pool handoff moved from four copy-engine memcpys a layer to
    compute-queue `doorbell_publish`/`copy_from_mapped` kernels.  Remaining per token: sync 9.5, rest 13,
    **pool 30 ms (the DRAM-bound floor)**, so a token graph alone tops out near ~30 tok/s; decode-60 needs the
    expert bytes out of the CPU path (K-series/E3).  Details in `docs/DEEPSEEK.md` §7 (M2 step 8).

---

## 5. PENDING — ENGINE (the G-PERF / N-RAM path)

| ID | item | acceptance | ref |
|---|---|---|---|
| E1 | **Transient release, event-gated** — reclaim the ~2× prefill the ring guard costs | prefill ≥ 900 tok/s on --expert-ram-gb with MMQ ON, G-COH green | `TODO §28`, findings §5 |
| E2 | **Hot-expert profile from real routing** — raise VRAM hit rate (CPU experts/layer down) | `--dump-routing` → `make_profile` → CPU experts/layer halves; decode up | `TODO §13-19` |
| E3 | **Decode 60+** — MTP policy + resident hot set + CPU-pool→GPU | measured decode ≥ 60 tok/s, G-COH green | `TODO §23` |
| E4 | **Adaptive chunk sizing** for short prompts | short-prompt prefill no longer PCIe-dominated | `TODO §1` |
| E5 | **Expert bundling** (gate/up/down contiguous; Apple/ZipMoE) | one sequential PCIe read per miss; measured prefill gain | `§12.2 #1` |
| E6 | **KV `q8_0`/`q4_0`** | free VRAM → larger resident hot set; measured decode gain, G-COH green | `§12.2 #2` |
| E7 | `--expert-cache-per-layer` `verify_slot` abort | per-layer residency works | `TODO §4` |

- **E7 root cause FIXED (2026-10-01):** `ExpertCache::open_sized` zeroed `layer_next_` right after `open` had set
  each layer's start to its own range, so every layer admitted into slot 0 (ranges overlapped) and the startup
  read-back compared two experts -> `verify_slot: slot 0 differs from the arena at byte 0`.  Fixed in
  `expert_cache.cpp` and unit-tested: `expert_cache_per_layer_test` -> **PASS** (layer 0 -> slots [0,4) -> <0,1>,
  layer 1 -> [4,8) -> <4,5>).  An end-to-end gfx1201 run is still to be recorded.

---

## 6. The one-line contract

> **No model below G-COH. No expert matmul on the CPU. Every kernel parity-gated and measured.
> Decode 60, prefill 5000, for every MoE we support — or it is OPEN.**

Anything not run and recorded here is not done.
