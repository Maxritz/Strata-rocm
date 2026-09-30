# Strata — master TODO (all outstanding items, as of 2026-09-30)

The single list. Priorities: **P0** blocks a shipping milestone, **P1** a family/feature, **P2** speed/scale,
**P3** backlog. Each item says what "done" means.

---

## P0 — correctness (blocks trust in any output)

1. **Verify-window nondeterminism (`--spec`).** The native path requires `--spec ≥ 2`, and that window
   diverges on ~1/6 runs (Swift: correct `248068 198 760…` vs degenerate). Ruled out: VRAM cache size, the
   CPU pool's sync, the prefill, draft variability. It is the window's own logits. *Done:* `--spec 2`
   reproduces byte-for-byte over N≥20 runs.
   - **Blocker on the fix:** `--dump-layers` (the bisection tool) can't run under `--spec` — the window
     hard-calls the **fused** HC read, while `--dump-layers` switches to the **unfused** residual layout
     (`layer_verify_compatible` refuses). Groundwork committed (`set_layer_dump` + the residency-table
     ordering). *Next:* let the window run unfused when dumping, or dump a fused-valid quantity; then bisect
     the first divergent layer.
2. **Whittle run to coherence** (qwen4exp 2048). Loads pack/geometry/embedding now; blocked at the
   **quantized QSA-indexer tensor** (`blk.3.indexer.k_proj/k_proj`), which `iq_pack.py` marks shape-only and
   `weights.cpp` refuses, while Swift's went to `dense.bin` (it was float in its GGUF). *Done:* the model
   answers coherently and matches llama.cpp top-1.

## P1 — models & families

3. **`qwen35moe`** (Qwen3.5-35B-A3B, ornith-35b, Tiel-Coder-35B, qwable, Unsloth-Ornith, Qwen3.8-Distill-35B
   — the biggest shelf, 17-25 GB). New decoder: reuse GDN + MoE (identical tensor names), write a plain
   **GQA + partial-RoPE** attention, drop hyper-connections. *Done:* load + coherent + parity.
4. **`deepseek4`** (DeepSeek-V4-Flash, MLA + sparse attention). Port ds4's `rocm/*.cuh`, gate on
   `tests/test-vectors/official.vec`. Largest job; `docs/DS4_BLUEPRINT.md`.
5. **`deepseek2` / GLM-4.7-Flash** (17 GB, fits). ds4 has a ROCm path.
6. **`laguna`, `muse-glimmer`, `k2-horizon`, `nemotron_h_moe`, `gpt-oss`, `qwen3moe`, `olmoe`** — each its
   own decoder. `--model-info` already recognises them.
7. **Pack tooling for new families** — `iq_pack.py` handles `qwen4exp`; each new architecture needs its
   tensors mapped.

## P1 — the profiling / "usage helper" feature (requested)

Run a model on a workload once, record the routing, and make the next load faster + help pick the model.
Serves the poor-hardware mission directly. **Mostly assembled already** (`--dump-routing`,
`tools/make_profile.py`, `--expert-profile`, `--expert-cache`; ds4 ships a compiled hotlist).

8. **`profile` workflow** — one command: run N prompts (a directory), `--dump-routing`, aggregate, write
   `<model>_<usage>_helper.bin`. Overnight/batch mode.
9. **Auto-load** the helper at start (no `--expert-profile` flag).
10. **`--usage <class>`** switch (`coding`, `docs`, `chat`, …) selecting `<model>_<class>_helper.bin`.
11. **Multi-file helpers** — keep them small and per-class; merge (weighted union, re-sorted, truncated to
    the cache) or, better, **select one per session** (merging dilutes).
12. **Concentration check** — `--dump-routing` reports whether the workload's hot set fits the cache
    *before* investing in a profile.
13. **Decision/bench step (optional)** — run candidates (models × quants) on the workload, report tok/s,
    cache-hit rate, RAM/VRAM, **and a quality gate** (compile/run code, validate schema, or top-1 vs
    llama.cpp). Without the quality gate, "pick the model" degenerates to "pick the smallest quant".
14. **RCO ordering (arXiv 2605.00649)** — rank experts by *loss impact per byte* under an exact budget
    instead of raw hit count. Pack-time optimiser.

## P1 — scale-down / hardware targets

15. **gfx1031 (RX 6700 XT) tuning pass** — the same binaries run (no WMMA; `sdot4` path), but kernel shapes
    and cache sizes must be measured on ~384 GB/s, not the 9070 XT's ~640.
16. **`--expert-ram-gb` default from available RAM** (48 GB box → ~32; leave the OS room).
17. **Canonical packs** for the models users want — they run the deterministic token path and sidestep the
    verify-window bug entirely.

## P2 — performance

18. **Decode 40+**: the PCIe floor for decode is ~24 ms/token (~42 tok/s) at 720 MB/token; needs *both*
    overlap (decode leaves ~half the link idle) and residency. `--kv-resident`, warm experts for the verify
    pass, `--spec` tuning, larger draft depth.
19. **Q2_0 vs IQ2_XS** — upstream measured 93.8 vs 70.3 decode; also a smaller footprint. Validate.
20. **Co-activation-ordered expert layout** (ZipMoE) — only after the window bug; only if the gather shows
    fragmentation.
21. **WMMA/MFMA injection** for RDNA3/4 (the dp4a path is the universal fallback; this is a speed-up only).
22. **Fuse gate/up/down** into one expert kernel.

## P2 — the bounded ring

23. The ring's ~27% decode gap versus the arena, and re-enabling the GPU alias/DMA path with a
    per-consumer release (docs §15.5 step 2). *Note:* the ring was **not** the nondeterminism source.

## P3 — backlog

24. `--expert-cache-per-layer` `verify_slot` abort ("slot 0 differs from the arena at byte 0").
25. Kernel-load-at-start (the `CUDA_MODULE_LOADING=EAGER` analog) to avoid mid-prompt OOM.
26. `iq_parity` fixtures (needs a generator linked against ggml's C quantizers).
27. DeepSeek `deepseek4-dspark` draft head (needs its target model).
28. Vision / `clip` / `mmproj` encoders.

---

## The immediate next action

**P0 #2 — the Whittle indexer blocker**, because it is a few lines and it unblocks a whole 35B-A3B model
*(done = coherent answer)*. Then **P0 #1 (the verify window)**, because every native model inherits it.
