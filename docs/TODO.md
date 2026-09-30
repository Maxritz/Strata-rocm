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
2. ~~**Whittle run to coherence**~~ — **PARKED 2026-09-30 by owner decision; the GGUF was deleted.** The safety
   guard in `generate.cpp` stays (it is what stopped the hang), and `--model-info` still recognises the
   family — recognition is separate from the run guard. Kept here so the next attempt starts from evidence
   instead of from scratch. **The hang's root cause is NOT established — do not re-derive the dead theories
   below.**
   *Disproved by a proof, not a guess:* there is **no prefill heap overflow**. `gdn_set_bytes` (prefill.cpp
   399-404), `qsa_set_bytes` (405-414) and `moe_set_bytes` (449-465) are each line-for-line mirrors of their
   carve counterparts (568-570 / 573-578 / 581-593), and `region = max(...)` (561-562), so every
   allocator's `used <= region` **by construction, for any model shape**. `Alloc b`'s sizes are all literals
   (so identical for Swift, which works), and `cap = qsa_selection_width(kTopkMaxCells=32768)` = 2048+4-1 =
   **2051 for both models**, so the QSA sizing cannot be model-dependent either. A 3-agent debate converged
   on "`qsa_set_bytes` omits `m.Qf`" — **false**: line 408 does count it (`T * 12288`).
   *Confirmed real, but SEMANTIC (wrong math), not memory-safety:* the QSA path hardcodes 24 heads where
   Whittle has 16 (`rms_rows(m.q, wqn, T * 24, ...)` prefill.cpp:1102, `rope(m.q, T, 24, 256, 6144, ...)`
   :1103), and `qsa_shapes` (layer.cpp:475) never propagates `indexer.top_k`. Table:
   | | Swift | Whittle |
   |---|---|---|
   | attention head_count | 24 | **16** |
   | key/value length | 256 | 256 |
   | indexer heads / key | 4 / 128 | 4 / 128 |
   | **indexer.top_k** | 2048 | **262144** (whole context) |
   *THE SECOND BLOCKER, FOUND LAST — the PLE GPU block is compile-time Swift, not runtime.* The **reader** is
   generic (`ngram.cpp` derives `head_dim` from the tensor shape, `heads_per_ngram`/`n_heads`/`table_rows`
   from metadata), but the **compute** is not: `src/kernels/cuda/ple.cu:235-236` pins
   `n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM` and then
   `static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed")`.
   So a 2048-wide model silently computes the PLE at 2560/10240. And the tables are **not the same shape** —
   read from the two GGUFs, not inferred:

   | | Swift (Qwen3.8-Flash-Next) | Whittle-35B |
   |---|---|---|
   | `embedding_length_per_layer_input` | 160 | **256** |
   | `ple.heads_per_ngram` | 8 | **4** |
   | `per_layer_token_embd.weight` | `[160, 320001536]` IQ4_NL | `[256, 39040000]` Q4_K |
   | `embedding_length` | 2560 | **2048** |

   Per the upstream card, Whittle's table is a **graft** — "keep 8 of Qwen's 16 heads and 25% of its rows,
   write those rows verbatim into dims [0, 640) of each 1024-wide order block" — so it is a genuinely
   different geometry, not a resized copy. Any Whittle attempt must **generalise `ple_block` to runtime
   geometry first** (delete the `static_assert`, thread `hc`/`hc_dim`/`head_dim`/`heads_per_ngram` through,
   and re-run `src/kernels/ple_parity.cpp` against both shapes), or the PLE output is garbage even with the
   hang fixed. Note the PLE reader also defaults `PleGeom` to Swift's values, so a missing metadata key
   silently means Swift — worth a hard error instead.
   *Also still open:* `ple.layers = [1]` means this garbage enters the residual stream at layer 1, upstream
   of the router — so it is a candidate for the corruption, though nothing yet proves it caused the hang
   (the hang is still unexplained). A **prefill watchdog** is the other prerequisite: the verify window has a
   20 s timeout, the prefill has none.

## P1 — models & families

3. **`deepseek2` / GLM-4.7-Flash — best value on the shelf, do this next.** Measured from the local GGUF:
   arch `deepseek2`, **47L / d 2048 / 20 heads / 1 kv, MLA (`kv_lora_rank` present), 64 experts / 4 used +
   1 shared**, 17.9 GB. **Quants are 100% supported** (IQ4_XS 54% / Q6_K 24% / Q5_K 20% / Q8_0) — *no new
   quant primitive*, so it is pure decoder work, and it unlocks the whole DeepSeek/GLM line. Port ds4's
   `rocm/*.cuh` MLA + shared-expert path.
4. **`k2-horizon` (K2-Horizon-MoVA-36B-A4B)** — 48L / d 2560 / 32 heads / 8 kv, **100 experts / 8 used +
   1 shared**, 20.8 GB, standard Q4_K family. Plain GQA MoE, no MLA, no exotic quant. Cheapest MoE after #3.
5. **`laguna` (Laguna-XS.2)** — 40L / d 2048, **256 experts / 8 used + 1 shared**, 16.8 GB, IQ4_XS/IQ4_XS.
   Quirk: `attention.head_count`/`head_count_kv` are **per-layer lists** (48/64 and 8), not scalars.
6. **MXFP4 primitive — gates the biggest models.** 86–98% of three collection files is a type we cannot
   dequant at all, and it is the single missing primitive for the whole modern-AMD-qu shelf:

   | file | arch | MXFP4/NVFP4 |
   |---|---|---|
   | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | `qwen35moe` | 86.5% MXFP4 |
   | `qwen-agentworld-35b-a3b-nvfp4.gguf` | `qwen35moe` | 98.5% **NVFP4** |
   | `gpt-oss-120b-Q8_0.gguf` | `gpt-oss` | 98.1% MXFP4 |

   MXFP4 measured **from the file itself**: **17 bytes per 32 elements = 4.25 bits/elem** (16 B packed FP4 +
   1 B E8M0 scale) — `gguf_reader.hpp` currently only has the *name* (`case 39`), and `moe_mmq.cu:145`
   handles Q2_K…Q6_K only. `dequant.hpp` is the natural home (one `inline dequantize_*(block, out)` each).
   Also add **NVFP4** (`case 40`) for the agentworld variant, and note **ROCmFPX** (AMD's own FP4/FP6/FP8,
   used by e.g. `Ornith-1.0-9B-ROCmFPX-STRIX_LEAN`) is a *separate* layout — but see P2 #21: on gfx1201 the
   RDNA4 FP4/FP6 paths are **native**, so ROCmFPX support and the WMMA item are the same project.

   NVFP4 now measured from `ggml-common.h` too (**not** 17 B/16 elem — that is the scale sub-block, and
   conflating the two is the easy mistake): **36 bytes per 64 elements = 4.5 bits/elem** = 4 × UE4M3 scale
   (one per 16-elem sub-block) + 32 B packed E2M1. Both layouts, the E2M1 codebook, and a 45-paper survey
   of what does and does not apply to gfx1031 are in **`RESEARCH_NOTES.md`** — read §3 before writing the
   dequant and §4.1 for why activations stay INT8. Two adopted specifics: **keep the FP4 packed and expand
   via an int8 LUT into `__dp4a`** (no offline FP4→int8 expansion), and **test the two scale decoders
   against `ggml-impl.h` verbatim** — both are non-obvious and both fail silently. `ggml_e8m0_to_fp32_half`
   is `2^(x-128)`, i.e. *half* of E8M0, because `kvalues = 2 * E2M1_float`; the "half" is the easy thing to
   omit and it is a uniform 2x error. `ggml_ue4m3_to_fp32` also halves, and maps **both `0x00` and `0x7F`
   to 0.0f** — `0x7F` is the standard UE4M3 NaN encoding and ggml deliberately zeroes it, so a NaN scale
   silently kills 16 weights instead of poisoning them. Note the sentinels are `0x7F` (UE4M3) and `0xFF`
   (E8M0 → 2^127); `0xFF` fed to a UE4M3 decoder is a *valid* 240.0, not an error.
7. **`qwen35moe`** (Qwen3.5-35B-A3B, ornith-35b, Tiel-Coder-35B, qwable, Unsloth-Ornith-1.5). Measured:
   40L / d 2048 / **16 heads / 2 kv** (8:1 GQA) / key=val=256, **256 experts / 8 + shared 512**, GDN on 3 of
   every 4 layers (`full_attention_interval = 4`), **partial RoPE 64 of 256** with
   `rope.dimension_sections = [11,11,10,0]` (sums 32 = 64/2, i.e. interleaved RoPE), no hyper-connections.
   GDN + MoE tensor names match the working path; the new code is the GQA + interleaved partial-RoPE
   attention. **Blocked behind #6 (MXFP4)**. Built-in **MTP** heads on the Tiel-Coder-35B variant would beat
   our external draft head once this lands.
8. **`gpt-oss` (120b)** — 36L / d 2880 / 64 heads / 8 kv, 128 experts / 4, 59 GB, 98% MXFP4. Its sliding
   window + attention sinks are the extra work beyond #6+#7.
9. **`gemma4` — mostly dense; only the 26B-A4B is MoE.** Verified against Google's Gemma 4 docs + tech
   report (arXiv 2607.02770), and against our local files. **E2B / E4B / 12B / 31B are dense**; the only MoE
   in the family is **26B-A4B** (25.2B total / 3.8B active, **30 layers, 128 experts / 8 active + 1 shared**,
   sliding window 1024, 256K ctx). The "E" in E2B/E4B is **effective** parameters — they use **PLE**
   (per-layer embeddings, as in Gemma 3n), so a 4.5B-effective model carries ~8B of weights. Our local
   `gemma4-12B-Q4_K_M.gguf` has **zero** expert tensors (layer 0 is plain dense: `attn_{q,k,v,output}` +
   `ffn_{gate,up,down}` + norms + `layer_output_scale`), which confirms the dense shape.
   *Real quirk:* `attention.head_count_kv` is a **per-layer list** — `8,8,8,8,8,1` repeating (48 entries for
   the 12B, `16…4` for the 31B). That is **KV sharing** ("E2B/E4B IT (dense, kv-shared layers)"), not a
   local/global window, so a loader that assumes a scalar `head_count_kv` will be wrong. Six `gemma4` files
   are on the shelf; dense ⇒ the cheapest new family after `qwen35`. Note `gemma-4-E4B-it` itself is **not**
   downloaded — only its MTP drafter is (`df\mtp-gemma-4-E4B-it-Q8_0.gguf`).
10. **`deepseek4` (DeepSeek-V4-Flash, MLA + sparse attention)** — **no local file**; ds4 reference only.
    Largest job; `docs/DS4_BLUEPRINT.md`; gate on `tests/test-vectors/official.vec`.
11. **`nemotron_h_moe`, `qwen3moe`, `olmoe`** — each its own decoder. `--model-info` already recognises them.
12. **Pack tooling for new families** — `iq_pack.py` handles `qwen4exp`; each new architecture needs its
    tensors mapped (GLM's MLA + shared expert, `k2-horizon`, `laguna`'s per-layer heads).

## P1 — the profiling / "usage helper" feature (requested)

Run a model on a workload once, record the routing, and make the next load faster + help pick the model.
Serves the poor-hardware mission directly. **Mostly assembled already** (`--dump-routing`,
`tools/make_profile.py`, `--expert-profile`, `--expert-cache`; ds4 ships a compiled hotlist).

13. **`profile` workflow** — one command: run N prompts (a directory), `--dump-routing`, aggregate, write
   `<model>_<usage>_helper.bin`. Overnight/batch mode.
14. **Auto-load** the helper at start (no `--expert-profile` flag).
15. **`--usage <class>`** switch (`coding`, `docs`, `chat`, …) selecting `<model>_<class>_helper.bin`.
16. **Multi-file helpers** — keep them small and per-class; merge (weighted union, re-sorted, truncated to
    the cache) or, better, **select one per session** (merging dilutes).
17. **Concentration check** — `--dump-routing` reports whether the workload's hot set fits the cache
    *before* investing in a profile.
18. **Decision/bench step (optional)** — run candidates (models × quants) on the workload, report tok/s,
    cache-hit rate, RAM/VRAM, **and a quality gate** (compile/run code, validate schema, or top-1 vs
    llama.cpp). Without the quality gate, "pick the model" degenerates to "pick the smallest quant".
19. **RCO ordering (arXiv 2605.00649)** — rank experts by *loss impact per byte* under an exact budget
    instead of raw hit count. Pack-time optimiser.

## P1 — scale-down / hardware targets

20. **gfx1031 (RX 6700 XT) tuning pass** — the same binaries run (no WMMA; `sdot4` path), but kernel shapes
    and cache sizes must be measured on ~384 GB/s, not the 9070 XT's ~640.
21. **`--expert-ram-gb` default from available RAM** (48 GB box → ~32; leave the OS room).
22. **Canonical packs** for the models users want — they run the deterministic token path and sidestep the
    verify-window bug entirely.

## P2 — performance

23. **Decode 40+**: the PCIe floor for decode is ~24 ms/token (~42 tok/s) at 720 MB/token; needs *both*
    overlap (decode leaves ~half the link idle) and residency. `--kv-resident`, warm experts for the verify
    pass, `--spec` tuning, larger draft depth.
24. **Q2_0 vs IQ2_XS** — upstream measured 93.8 vs 70.3 decode; also a smaller footprint. Validate.
25. **Co-activation-ordered expert layout** (ZipMoE) — only after the window bug; only if the gather shows
    fragmentation.
26. **WMMA/MFMA injection** for RDNA3/4 (the dp4a path is the universal fallback; this is a speed-up only).
27. **Fuse gate/up/down** into one expert kernel.

## P2 — the bounded ring

28. The ring's ~27% decode gap versus the arena, and re-enabling the GPU alias/DMA path with a
    per-consumer release (docs §15.5 step 2). *Note:* the ring was **not** the nondeterminism source.

## P3 — backlog

29. `--expert-cache-per-layer` `verify_slot` abort ("slot 0 differs from the arena at byte 0").
30. Kernel-load-at-start (the `CUDA_MODULE_LOADING=EAGER` analog) to avoid mid-prompt OOM.
31. `iq_parity` fixtures (needs a generator linked against ggml's C quantizers).
32. DeepSeek `deepseek4-dspark` draft head (needs its target model).
33. Vision / `clip` / `mmproj` encoders.

---

## The immediate next action

**P1 #3 — `qwen35moe`**, since Whittle is parked and this is the biggest shelf in the collection
(`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` 18.3 GB, `qwen-agentworld-35b-a3b-nvfp4.gguf` 18.4 GB,
`Qwable-27b_Q4_K_M.gguf` 15.4 GB, ornith-35b/Tiel-Coder-35B/Qwen3.8-Distill-35B). GDN + MoE reuse directly
(identical tensor names); the work is a plain **GQA + partial-RoPE** attention with hyper-connections dropped.

**P0 #1 (the verify window)** stays P0 — every native model inherits it, and canonical packs
(P1 #17) sidestep it entirely. When Whittle comes back: generalise `ple_block` to runtime geometry first
(P0 #2), and add a prefill watchdog before any GPU run.
