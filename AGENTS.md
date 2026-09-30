# AGENTS.md — Strata standing task

## The standing task is `docs/CONTRACT.md`. Read it first, every session.

The goal is unwavering and it is the whole job:

> **Fastest possible COHERENT inference for ALL supported MoE models.**
> No model below G-COH (L1 ≤ 1e-3 vs CPU/llama.cpp). No expert matmul on the CPU critical path (G-GPU).
> Decode ≥ 60 tok/s, prefill ≥ 5000 tok/s on gfx1201 (G-PERF).

Work the contract's clauses **in order**, top to bottom. Each clause is DONE only when its **Acceptance command has
been run on gfx1201 and its output recorded in `docs/CONTRACT.md`.** Everything unrun is OPEN. Do not reorder to
what is easy; do not mark anything green that was not measured.

## Operating rules (non-negotiable, from the contract)

- **[N-PARITY]** Every new kernel is parity-gated vs the validated scalar reference (`dequantize_*`) BEFORE it is
  wired into a run path. New kernels default OFF until green.
- **[N-MEASURE]** No fabricated numbers. Every perf claim is a measured gfx1201 number with the command.
- **[N-NOSTUB]** No stubs, placeholders, TODOs, or "for now". If it cannot be done, it fails loudly.
- **[N-RAM]** Experts stream lazily and boundedly; the hot set stays resident in VRAM.

## Current position (update this line as clauses close)

**The authoritative task ledgers are `docs/TODO.md` (all outstanding items, with priorities and what "done"
means) and `docs/RESEARCH_NOTES.md` (a verified 45-paper corpus screened against this exact hardware).  Read
those two before picking work; `docs/PAPER_FINDINGS.md` maps the Strata paper's findings to the clauses.**

- DONE: D1–D7 (FP4 GEMV 1.140 TOPS · MTP 37 tok/s · prefill-fit · spec-6 LDS · ring hang · WMMA peak 196 ·
  K1a layout parity · **K1b int4 WMMA GEMM 55.8-64.7 T-MAC/s, parity PASS, acceptance (b) met**).  Plus: CPU
  probe SIGILL fixed (the P0 #0 crash) with the AVX2 oracle validated; upstream 8acd17c/53f9e7a/#257.
- **STRATEGIC FINDING:** prefill is **bandwidth-bound on host-arena expert streaming** (upstream #269), not
  compute, so a faster GEMM does not raise it by itself; for the RUNNING model the int4 MMA would repack 2-bit
  Q2_0 to 4-bit, DOUBLING the stream.  K1c therefore targets **already-4-bit** models.
- **OPEN NOW (per TODO "immediate next action" + P1 #3):** the next model is **`deepseek2`/GLM-4.7-Flash**
  (TODO calls it the best-value shelf item: MLA, 64 experts/4, **all quants already supported** -> pure decoder
  work) or **`qwen35moe`** (GDN+MoE reuse directly; new code is GQA + interleaved partial-RoPE; gated behind the
  MXFP4 primitive for its UD/NVFP4 files).  `olmoe` is the simplest but is TODO #11, not first.
- Then: K2 (MXFP4/NVFP4 tensor core, TODO #6 - gates the biggest models), K3 (16B repack), K4 (fusion);
  E1 transient release (code done, GPU gain unmeasured), E2 hot-expert profile, E3 decode 60 (TODO #23), E4-E7.
- **K5 (new, from the paper - docs/PAPER_FINDINGS.md):** a **T-MAC-style LUT** CPU expert kernel.  The paper's
  finding 7 says the i-quants are limited by CPU *arithmetic* (not RAM), and this host has **no AVX-512**, so
  the IQ2_S/Q2_0 CPU rows gate every model's decode here.  T-MAC (arXiv:2407.00088) reports ~4x over llama.cpp
  with bit-wise table lookups (no dequant, no multiplies).  This is the highest-value kernel for this machine.

## Model set this must serve

The engine today implements **one** architecture: `qwen4exp` (Flash-Next: 36 GDN + 12 QSA, hyper-connections
hc=4, 512-expert top-10 MoE).  `generate.cpp` reads geometry as `u("qwen4exp.block_count", …)`, so any other
`general.architecture` is a **new engine feature**, not just a repack.

| model | arch key | L | hidden | experts | top | heads | weights | status |
|---|---|---|---|---|---|---|---|---|
| Swift *(running)* | `qwen4exp` | 48 | 2560 | 512 | 10 | QSA | IQ2_S+Q2_0 | DONE |
| reap-288 | `qwen4exp` | 48 | 2560 | 288 | 10 | QSA | Q2_0 | pack built |
| ngram-embeddings | — | — | — | — | — | — | — | **PLE table** (`--ple-gguf`), not a model |
| Qwen3.5-35B-A3B (M1) | `qwen35moe` | 40 | 2048 | 256 | 8 | 16/2 GQA | Q4_K | TODO |
| qwen-agentworld (M1) | `qwen35moe` | 40 | 2048 | 256 | 8 | 16/2 | **NVFP4** | TODO (K2) |
| GLM-4.7-Flash (M2) | `deepseek2` | 47 | 2048 | 64 | 4 | 20/1 **MLA** | Q6_K/Q8_0 | TODO |
| gemma4-26B-A4B (M3) | `gemma4` | 30 | 2816 | 128 | 8 | 16/varies | IQ4_XS | TODO |
| k2-horizon (M3) | `k2-horizon` | 48 | 2560 | 100 | 8 | 32/8 | Q4_K | TODO |
| laguna (M3) | `laguna` | 40 | 2048 | 256 | 8 | 48-64/8 | IQ4_XS | TODO |
| olmoe (M5) | `olmoe` | 16 | 2048 | 64 | 8 | 16/16 MHA | Q4_K | TODO - simplest, start here |
| gpt-oss-120b | `gpt-oss` | 36 | 2880 | 128 | 4 | 64/8 | **MXFP4** | TODO (K2) |

MXFP4/NVFP4 (qwen-agentworld, gpt-oss-120b) ride the FP4 + WMMA path — on gfx1201 those are the same project.

## Upstream sync (Niko1221/Strata)

Kept in sync by hand; the engine here is a fork.  Applied: merged RDNA4 `3d99089`; prefill stream floor 1024
(`8acd17c`); volatile doorbell store (`53f9e7a`); AVX2 Q2_0 b256 unpack (#257).  Not applicable: #262's
packed-byte intrinsics (this fork has no `hip_compat` shim - kernels use `__builtin_amdgcn_*` directly).


## Build / test (this host)

- ROCm: `G:\ROCM10RT-gfx1201` (add `G:\ROCM10RT-gfx1201\bin` to PATH for the runtime DLLs).
- Engine: `cmake --build build_gfx1201 --target strata`. Kernels: `--target fp4_gemv_parity`.
- Microbenches: `hipcc -O3 --offload-arch=gfx1201 bench/micro/<x>.cu`.
- Reference model (present): `H:\OLLAMA-Models\strata-pack-swift` + `...\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-*.gguf`.
- MTP runtime (built): `C:\Users\rr\AppData\Local\Temp\opencode\mtp-rt`.
