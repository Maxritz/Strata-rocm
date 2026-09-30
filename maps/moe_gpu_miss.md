# Map: GPU MoE miss-path lift — file:line anchors

Source of truth for the GOAL in `plans/moe-gpu-miss-pipeline.md`.  This is a READ-ONLY
map (Step 1 artifact).  All paths relative to the repo root (`C:\Users\rr\OneDrive\Desktop\Strata-rocm`).

Host: RX 9070 XT / gfx1201 (84 CUs, ~15.9 GB VRAM, REBAR ~4 GB/s host-blob direct read).
Model: Qwen3.8-Flash-MoE-288 — Q4_K gate/up (Fmt<12>), Q5_0 down (Fmt<6>), Q5_0 PLE,
n_embd=2048, n_ff=1536, k=6, 48 layers.  307/315 experts resident (~0.93 GB VRAM cache,
round-robin), 8-10 distinct PCIe-read experts/layer.  Current decode 7.6-11 tok/s,
prefill 14.3; CPU miss-path is ~57 ms/round (gate/up 31-32 ms + down 25-26 ms) with the
GPU idle.

## How the split currently works
The engine splits each layer's top-k expert compute between a GPU "HIT" path (VRAM-resident
experts) and a CPU "POOL" path (missed / PCIe experts).  `Plan v0.3 P6` in `generate.cpp`
classifies each routed expert into `kind ∈ {-1 CPU, 0 VRAM, 1 PCIe}`:
- `kd == 0` (VRAM resident): ptr into the VRAM cache (`cache_base + slot_off`) → GPU `moe_hit_grouped_s2`.
- `kd == 1` (PCIe, resident-but-uncached): ptr via `d.src->blob(d.layers, e)` (the PinnedArena host pointer,
  REBAR-mapped) — **currently still handed to the CPU pool** (`expert_pool_dispatch_multi`), which is the 57 ms drain.
- `kd == -1` (CPU only / refused): CPU pool.

`fill_slot` (expert_source.cpp:521) already copies a missed expert from `d.src->blob()` over REBAR into
the VRAM cache on the compute stream — so the REBAR-read plumbing EXISTS; the GOAL is to feed `kd==1`
experts straight to a GPU kernel (no staging copy, no CPU round-trip) and skip the CPU pool for them.

## CPU miss path (the thing to lift)
- `drive_pool_multi` — `src/program/generate.cpp:496`  (the doorbell callback for the verify-window / pooled miss path).
- → `expert_pool_dispatch_multi(ExpertDispatch& d, x_f, ids, n_tok, k, out)` —
  `src/core/expert_source.cpp:275`, declared `include/strata/core/expert_source.hpp:254`.
  Multi-token; the single-token pool is the sibling `run` path in expert_source.cpp (~240-270).
- Classification + VRAM/PCIe ptr collection: `src/core/expert_source.cpp:306-364`
  (`kind[]`, `distinct[]`, `dma_src[]`, `P.ptr[start]/P.dst/dst/t`).

## GPU hit path (the thing to extend to PCIe misses)
- `expert_hit_run(void* user, void* stream, HitPhase phase, ids, k)` — `src/core/expert_source.cpp:489`,
  declared `include/strata/core/expert_source.hpp:263`.  Phases: `Launch` (classify cache
  resident/admit/fill) then `Combine` (`add_inplace(d.parts_out, d.hit_out, …)` at :593).
- Cache residence test: `d.cache->slot_of(d.layers, e)` — `src/core/expert_source.cpp:510`.
- Admit + fill a slot from the resident-but-uncached expert: `d.cache->admit` +
  `d.cache->fill_slot(cand, b=d.src->blob(d.layers,e), cs, …)` — `src/core/expert_source.cpp:512-529`.
  NOTE `fill_slot` reads the PinnedArena host pointer directly (REBAR); this is the no-staging-copy
  precedent the PCIe-direct kernel should reuse.
- Hit compute dispatch (resident experts):
  - `moe_hit_grouped_s2` / `moe_hit_grouped_s2_cpu_order` — `src/core/expert_source.cpp:570-573`,
    implemented in `src/kernels/cuda/s2_expert_grouped.cu` (grouped GPU expert, "R4's grouped GPU expert").
  - `add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs)` — `src/core/expert_source.cpp:593`
    (parts = CPU pool output; `hit_out` = GPU hit output).
- Activation quantization for the hit path: `quantize_q8_0_scaled` — `src/core/expert_source.cpp:567`
  (`iq_kernels.cu`), parity-checked vs the CPU `act_quant_q8_1` — see the 80/80 note at :555-561;
  a parity mismatch here would silently corrupt results, so any new path must reuse `quantize_q8_0_scaled`.

## The Q4_K / Q5_0 decode primitives (what to reuse, NOT the FP4 tile)
- `vec_dot_q4_K_q8_1(vb, bq8_1, kbx, iqs)` — `src/kernels/cuda/iq_kernels.cu:82`.
- `vec_dot_q5_0_q8_1(vb, bq8_1, kbx, iqs)` — `src/kernels/cuda/iq_kernels.cu:359`.
- Q4_K codebook table `get_int_from_table_16` — `src/kernels/cuda/iq_kernels.cu:42` (`Fmt<12>`).
  **Lesson from the FP4 tile (2dbf569):** the non-linear, out-of-4-bit-range codebooks
  (`{0,1,2,3,4,6,8,12,…}`) must NOT be fed raw to a 4-bit dot instruction; load the codebook into
  SHARED memory and index it (the 2.12x divergent-constant penalty seen for IQ4NL applies here).
- Dispatch kernels in `iq_kernels.cu`:
  - `mmvq_kernel` — `src/kernels/cuda/iq_kernels.cu:425` (single expert matvec).
  - `native_gu_kernel` — `src/kernels/cuda/iq_kernels.cu:443` (gate/up).
  - `native_down_kernel` — `src/kernels/cuda/iq_kernels.cu:475` (down).
  - `swiglu_entries_kernel` — `src/kernels/cuda/iq_kernels.cu:466`.
  - `quantize_q8_1_kernel` — `src/kernels/cuda/iq_kernels.cu:497`.
- Q4_K/Q5_0 DEVICE dequant helpers `dq_*` — `src/kernels/cuda/iq_kernels.cu:523+` (e.g. `dq_iq2_xxs`
  pattern; the Q4_K/Q5_0 equivalents are adjacent).

## The expert cache + resident arena (residency / eviction)
- `class ExpertCache` — `include/strata/core/expert_cache.hpp:57` (`slot_of`, `admit`, `fill_slot`,
  `release_layer`, the round-robin / compulsory-miss / `adapt_swaps` eviction policy).
- `PinnedArena` (REBAR-mapped host arena) — `include/strata/core/pinned.hpp:25`, impl
  `src/core/pinned.cu:128` (`hipHostRegister` at pinned.cu:139/`hipHostRegisterPortable | Mapped`).
  Holds `experts.bin` once, at startup (generate.cpp:315 context).  This is the host-resident
  buffer the GPU reads missed-expert blobs from directly.
- Cache-backed arena allocation: `src/core/expert_source.cpp:709` and `:851` (`new PinnedArena(...)`
  for the resident slots / cache).
- Cache wiring / flags (OFF by default, the goal is turning it ON + PCIe-direct):
  - `int expert_cache` — `src/program/generate.cpp:186` (0 = off; -1 = auto-size; >0 = slots).
  - `expert_cache_per_layer` — :193 (the "ROUND 328" admission fix).
  - `--expert-cache auto` sizing + VRAM-reserve accounting: `:1690-1730`.
  - Hit/run wiring: `src/program/generate.cpp:1969`
    `(o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run`.

## KV staging (Step 5 target)
- KV is currently staged in pinned HOST RAM, not VRAM: see the "whole K/V in pinned RAM; the freed
  VRAM goes to expert slots" note at `src/program/generate.cpp:315`.  Moving KV to VRAM (no per-token
  host round-trip) is independent of expert compute.

## CPU quant type conventions to mirror (AGENTS.md: match iq_kernels.cu idioms, no dp4a)
- `Fmt<6>` = Q5_0 down; `Fmt<12>` = Q4_K gate/up; PLE = Q5_0.  (These Fmt indices are the q4/q5 type codes
  used in `ExpertLayout`; see the Q4_K 16-entry codebook above.)

## File index (the map, in one list)
CPU miss path:
- src/program/generate.cpp:496 `drive_pool_multi`
- src/program/generate.cpp:1969 hit/run wiring
- src/program/generate.cpp:306-364 P6 classification (`kind[]`, VRAM/PCIe ptr collection)
- src/program/generate.cpp:315 KV-in-pinned-RAM note
- src/core/expert_source.cpp:275 `expert_pool_dispatch_multi`
- src/core/expert_source.cpp:489 `expert_hit_run` (Launch→Combine)
- src/core/expert_source.cpp:510-537 cache classification + admit/fill
- src/core/expert_source.cpp:567 `quantize_q8_0_scaled` (parity-checked activation quant)
- src/core/expert_source.cpp:570-573 `moe_hit_grouped_s2` / `_cpu_order` dispatch
- src/core/expert_source.cpp:593 `add_inplace` (Combine)
- src/core/expert_source.cpp:709,851 cache arena allocation (`new PinnedArena(...)`)
- include/strata/core/expert_source.hpp:254 (dispatch_multi decl), :263 (hit_run decl)

GPU primitives:
- src/kernels/cuda/iq_kernels.cu:82  `vec_dot_q4_K_q8_1`
- src/kernels/cuda/iq_kernels.cu:359 `vec_dot_q5_0_q8_1`
- src/kernels/cuda/iq_kernels.cu:42  `get_int_from_table_16` (Q4_K codebook, shared-mem load)
- src/kernels/cuda/iq_kernels.cu:425 `mmvq_kernel`
- src/kernels/cuda/iq_kernels.cu:443 `native_gu_kernel`
- src/kernels/cuda/iq_kernels.cu:466 `swiglu_entries_kernel`
- src/kernels/cuda/iq_kernels.cu:475 `native_down_kernel`
- src/kernels/cuda/iq_kernels.cu:497 `quantize_q8_1_kernel`
- src/kernels/cuda/iq_kernels.cu:523+ `dq_*` device dequant helpers
- src/kernels/cuda/s2_expert_grouped.cu  grouped GPU expert (`moe_hit_grouped_s2`)
- src/core/pinned.cu:128-199 `PinnedArena` (`hipHostRegister` at :139)

Cache/arena types:
- include/strata/core/expert_cache.hpp:57 `class ExpertCache`
- include/strata/core/pinned.hpp:25 `struct PinnedArena`

## What the GOAL reduces to (the engineering delta, not a rewrite)
Extend `expert_hit_run` (:489) and `moe_hit_grouped_s2` (s2_expert_grouped.cu) to accept the `kd==1`
(PCIe-direct) experts in addition to `kd==0` (VRAM-resident): read the still-uncached expert blob from
`d.src->blob(d.layers, e)` (PinnedArena, REBAR) straight into the grouped kernel — no `hipMemcpy` staging,
no CPU `drive_pool_multi` call for those experts.  Gate each new kernel with a parity test against the
CPU `expert_pool_dispatch_multi` output (L1 <= 1e-3).  The admission/eviction policy in `ExpertCache`
and the `fill_slot` REBAR read in `expert_hit_run` are the precedents to follow.
