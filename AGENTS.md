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

- DONE: D1–D7 (FP4 GEMV 1.140 TOPS · MTP 37 tok/s · prefill-fit · spec-6 LDS · ring hang · WMMA peak 196 ·
  K1a int4 MMA layout parity).
- **OPEN NOW: K1b** — int4 WMMA expert GEMM. Best measured 9.7–12.8 T-MAC/s (parity PASS); acceptance ≥ 40.
  Next change: 4 accumulators/wave (16×64) + larger BK → ≥8 MMAs/barrier, then double-buffered LDS.
- Then: K2 (MXFP4/NVFP4 tensor core), K3 (16B repack), K4 (fusion); M1 qwen35moe … M6 ROCmFPX; E1–E7.

## Model set this must serve

qwen4exp: **reap-288 (Q2_0), ngram-embeddings (Q4_0), Swift (IQ2_S+Q2_0)**. Then **deepseek2/GLM (MLA)**,
**qwen35moe** (GQA + interleaved partial-RoPE), **k2-horizon, laguna, gemma4-26B-A4B**, **DeepSeek-V4**,
**nemotron/olmoe/qwen3moe**, **ROCmFPX**. MXFP4/NVFP4 (Qwen3.5-35B-A3B, qwen-agentworld, gpt-oss-120b) ride the
FP4 + WMMA path — on gfx1201 those are the same project.

## Build / test (this host)

- ROCm: `G:\ROCM10RT-gfx1201` (add `G:\ROCM10RT-gfx1201\bin` to PATH for the runtime DLLs).
- Engine: `cmake --build build_gfx1201 --target strata`. Kernels: `--target fp4_gemv_parity`.
- Microbenches: `hipcc -O3 --offload-arch=gfx1201 bench/micro/<x>.cu`.
- Reference model (present): `H:\OLLAMA-Models\strata-pack-swift` + `...\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-*.gguf`.
- MTP runtime (built): `C:\Users\rr\AppData\Local\Temp\opencode\mtp-rt`.
