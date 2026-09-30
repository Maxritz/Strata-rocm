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
- **Owner-clause:** do not enable until (a) and (b) pass.

### K2 — MXFP4 / NVFP4 tensor-core path  *(TODO §6: the RDNA4 FP4 path and WMMA are the SAME project)*
Requirement: the FP4 decode (`fp4_gemv_q8`, D1) extended to the batched tensor-core form for
Qwen3.5-35B-A3B / qwen-agentworld-nvfp4 / gpt-oss-120b.
- **Acceptance:** parity vs `dequantize_mxfp4`/`dequantize_nvfp4`; measured TOPS at B=128; G-COH on a MXFP4 model.

### K3 — direct-to-LDS + wide (uint4) loads
Requirement: repack the MXFP4 17-byte block to a 16-byte-aligned layout so `raw_buffer_load_lds` and 128-bit
loads vectorize (the 17-byte stride currently defeats them).
- **Acceptance:** `fp4_gemv_q8` ≥ previous 1.140 TOPS AND the repack is a lossless relayout (byte-parity test).

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

---

## 6. The one-line contract

> **No model below G-COH. No expert matmul on the CPU. Every kernel parity-gated and measured.
> Decode 60, prefill 5000, for every MoE we support — or it is OPEN.**

Anything not run and recorded here is not done.
