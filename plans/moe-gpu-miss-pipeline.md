# Blueprint: GPU-resident MoE (miss-path GPU lift + lazy REBAR load + KV-VRAM staging)

Objective: move the CPU-side missed-expert MoE GEMM onto the GPU and keep all
expert compute + KV resident, reading uncached expert blobs directly over REBAR
with no host staging; reach decode >= 50 tok/s with L1 parity <= 1e-3 vs the CPU
reference.  Grounded in `task_gpu_miss.txt`.

Scope: Qwen3.8-Flash-MoE-288 (Q4_K gate/up, Q5_0 down, Q5_0 PLE, n_embd=2048,
n_ff=1536, k=6, 48 layers).  Host: RX 9070 XT / gfx1201 (84 CUs, ~15.9 GB VRAM,
REBAR ~4 GB/s host-blob direct read).  Currently: cache-hit experts already on GPU
via `iq_kernels.cu` (`vec_dot_q4_K_q8_1`, `vec_dot_q5_0_q8_1`); the MISS path runs
on CPU in `drive_pool_multi` (`src/program/generate.cpp:459-497`) — 31-32 ms gate/up
+ 25-26 ms down = ~57 ms/round, GPU idle.  307/315 experts resident (~0.93 GB VRAM
cache, round-robin profile), 8-10 distinct PCIe-read experts/layer.

Tile 0 prerequisite (COMPLETE, committed `2dbf569`): the MXFP4/NVFP4 GEMV parity
tile (`fp4_gemv` baseline + `fp4_gemv_fast` vectorized decode, shared
`include/strata/kernels/fp4_decode.hpp`, bit-identical to `dequantize_mxfp4` /
`dequantize_nvfp4`).  **NOTE on relevance**: the MoE weights here are Q4_K/Q5_0, not
FP4, so tile 0 is a *foundational but non-blocking* dependency — the fused MoE GEMM
reuses the Q4_K/Q5_0 `vec_dot` idioms, not the FP4 kernel.  Tile 0 matters because
the "fast decode" lesson (vectorize the codebook lookup, never trust a raw 4-bit
dot for a nonlinear codebook) carries over to the Q4_K/Q5_0 decode.

Anti-patterns (from `task_gpu_miss.txt` + FP4 tile): no dp4a (scalar q4_K), match
`Fmt<6>/<8>/<12>`, `s2_expert_grouped` conventions; do NOT quantize activations to
int8 in the fused path (numerics change); do NOT stage expert blobs through a host
copy (defeats REBAR); load the codebook/shared decode into shared memory (not
fast/constant) to avoid the 2.12x divergent-read penalty seen for IQ4NL; do NOT move
top-k to GPU unless the gate output is bit-identical to the CPU top-k.

---

## Step 1 — Map the CPU↔GPU boundary and the resident-expert cache (PREREQUISITE)

**Goal:** produce a precise map (`maps/moe_gpu_miss.md`) of (a) which
`src/kernels/cuda/*.cu` files own the resident-expert hit path, (b) the CPU miss path
in `src/program/generate.cpp:459-497` (`drive_pool_multi`), (c) how the 0.93 GB VRAM
cache pins/evicts experts, (d) where the Q4_K/Q5_0 codebook lives, (e) how the gate's
top-k dispatches rows to experts, (f) where attention KV is staged.

**Context brief (cold-start):** the resident hit path is in `src/kernels/cuda/iq_kernels.cu`
(`vec_dot_q4_K_q8_1`, `vec_dot_q5_0_q8_1`) and `s2_expert_grouped.cu`.  The miss path is
`drive_pool_multi` at `src/program/generate.cpp:459-497`.  `PinnedArena` exposes the
host-resident buffers to the GPU via `hipHostRegister` (REBAR).  GFUF reader is
`include/strata/artifact/gguf_reader.hpp`.  Quant types: Q4_K = `Fmt<12>`, Q5_0 =
`Fmt<6>` (per `task_gpu_miss.txt`).

**Task list:**
- grep `src/kernels/cuda/*` for `vec_dot_q4_K_q8_1`, `vec_dot_q5_0_q8_1`,
  `s2_expert_grouped`, codebook arrays, and `__device__`/`hipHostRegister` usage.
- grep `src/program/generate.cpp` for `drive_pool_multi`, `topk`, `expert`, `kv`,
  `hipMemcpy`/`hipMalloc` around 459-497.
- read `include/strata/artifact/gguf_reader.hpp` for the Q4_K/Q5_0 dequant layout and
  the Fmt template parameters.
- write `maps/moe_gpu_miss.md` with: file:line for each component above, the
  CPU↔GPU crossing points (memcpys), the cache eviction policy, the gate→dispatch→scatter
  data flow.

**Verification commands:**
- `grep -rn "vec_dot_q4_K_q8_1\|vec_dot_q5_0_q8_1" src/kernels/cuda/iq_kernels.cu`
- `grep -rn "drive_pool_multi\|hipMemcpy" src/program/generate.cpp | sed -n '1,40p'`
- compile still green: `cmake --build build_gfx1201 --target strata_kernels 2>&1`

**Exit criteria:** `maps/moe_gpu_miss.md` exists and every cited file:line resolves to
the claimed component; one compile is green (no code changed in this step).

**Parallel:** none — Step 1 blocks Steps 2, 3, 4, 5.

---

## Step 2 — Fused GPU MoE GEMM for a single resident-but-uncached expert (parity gate)

**Goal:** one fused kernel that does gate/up (Q4_K) → down (Q5_0) for ONE expert read
directly over REBAR, numerically identical to the CPU `drive_pool_multi` for that expert.
This is the minimal numerical gate before scaling dispatch.

**Context brief:** reuse the Q4_K/Q5_0 `vec_dot_*` idioms from Step 1, NOT the FP4
`fp4_gemv_fast` (weights are Q4_K/Q5_0).  Kernel signature: `moe_fused_one_expert(x,
gate_q4k, up_q4k, down_q50, y, n_in, n_ff, block_bytes_gate, block_bytes_up, block_bytes_down)`.
Read `gate_q4k`/`up_q4k`/`down_q50` straight from the `PinnedArena` host pointer (REBAR,
no `hipMemcpy`).  Single thread-block (n_ff-wide) to start; k=1.

**Task list:**
- write `src/kernels/cuda/moe_fused_one_expert.cu` + header entry
  `include/strata/kernels/moe_fused.hpp`
- parity test `src/kernels/moe_fused_one_expert_parity.cpp`: hand-compute the same
  single expert on CPU via the validated Q4_K/Q5_0 decoder, compare in fp32, tol 1e-3
  L1 (per `task_gpu_miss.txt`).
- register in CMakeLists (`/build_gfx1201`), run on gfx1201.

**Verification commands:**
- `cmake --build build_gfx1201 --target moe_fused_one_expert_parity`
- `build_gfx1201\moe_fused_one_expert_parity.exe`
- `L1 <= 1e-3` printed by the test

**Exit criteria:** parity test green, `moe_fused_one_expert` committed.  No CPU fallback
introduced.

**Parallel:** Step 4 (cache/REBAR load) can start in parallel once Step 1 is done.

---

## Step 3 — Dispatch fusion (top-k gate → expert → weight-load → GEMM → scatter)

**Goal:** collapse the CPU top-k → expert selection → weight fetch → GEMM → output
scatter into one GPU kernel that hits cached experts in VRAM and streams uncached ones
over REBAR.  k=6 per row.

**Context brief:** depends on Step 2 (the fused GEMM) and Step 1 (cache + top-k).  The
CPU top-k gate must move only if its output (top-6 indices per row) is bit-identical
to the existing CPU top-k — assert that in the test before replacing it.

**Task list:**
- write `src/kernels/cuda/moe_dispatch.cu`: per-row top-k (Q4_K gate), dispatch to the
  resident/uncached expert buffers, accumulate into per-row output scatter.
- parity test `src/kernels/moe_dispatch_parity.cpp`: compare the full MOE block output
  (gate+topk+grouped-GEMM+scatter) to the CPU `drive_pool_multi` reference, L1 <= 1e-3.
- register + run on gfx1201.

**Verification commands:**
- `cmake --build build_gfx1201 --target moe_dispatch_parity`
- `build_gfx1201\moe_dispatch_parity.exe`

**Exit criteria:** dispatch parity green at k=6; `moe_dispatch` committed.

**Parallel:** Step 5 (KV-VRAM) is independent of Step 3 once Step 1 is done.

---

## Step 4 — Hot-expert VRAM cache + lazy REBAR load / eviction

**Goal:** make the uncached-expert read in Step 3 actually lazy and cache-aware: experts
not in the 0.93 GB VRAM cache are loaded on first hit over REBAR (no staging copy),
evicted round-robin (the existing profile), so the PCIe-read bandwidth observed in
`task_gpu_miss.txt` is the floor, not a per-token copy.

**Context brief:** the cache is already 307/315 resident round-robin; this step
integrates the lazy on-miss load into the dispatch kernel from Step 3, using
`PinnedArena` host pointers read directly by the GPU.

**Task list:**
- extend `moe_dispatch.cu` with the miss-path: on a cache miss, read the expert blob
  from the `PinnedArena` host pointer over REBAR inside the kernel (no `hipMemcpy`).
- test `src/kernels/moe_cache_miss_parity.cpp`: force a cold cache, run one dispatch,
  assert the uncached expert's contribution matches the CPU reference (L1 <= 1e-3) and
  that throughput is REBAR-bound not host-copy-bound.
- register + run on gfx1201; measure PCIe-read bytes vs cache-hit throughput.

**Verification commands:**
- `cmake --build build_gfx1201 --target moe_cache_miss_parity`
- `build_gfx1201\moe_cache_miss_parity.exe`

**Exit criteria:** cold-miss parity green; measured PCIe-read throughput reported (not
fabricated); `moe_dispatch` updated and re-tested against Steps 2/3 parity.

**Parallel:** Steps 2 and 4 overlapped once Step 1 done; Step 4 needs Step 3's dispatch
shape, so it's serial-after-3 for the integrated kernel (can prototype the cache
independently in parallel).

---

## Step 5 — KV-VRAM staging (no host round-trip)

**Goal:** keep the attention KV cache resident in VRAM for the whole decode stream and
feed it to the MoE block's residual path without a host↔device copy per token.

**Context brief:** independent of Steps 3/4 (KV is per-layer attention state, not expert
weights).  Locate the KV allocation in `generate.cpp`; move it to a VRAM buffer that the
GPU kernels write/read in place.

**Task list:**
- read Step 1 map for KV staging points; allocate KV in VRAM (`hipMalloc`), remove the
  per-token host writeback in the decode loop.
- test `src/kernels/moe_kv_vram_parity.cpp`: run a 20-token decode, assert KV stays on
  device (count host<->device KV bytes == 0) and output matches CPU reference L1 <= 1e-3.
- register + run on gfx1201.

**Verification commands:**
- `cmake --build build_gfx1201 --target moe_kv_vram_parity`
- `build_gfx1201\moe_kv_vram_parity.exe`

**Exit criteria:** zero host↔device KV bytes per token; parity green; committed.

---

## Step 6 — End-to-end coherence parity gate

**Goal:** one gate that runs a real decode pass (multi-layer MoE block with Steps 2-5
wired in) and asserts L1 <= 1e-3 vs the CPU reference across the 48-layer
Qwen3.8-Flash-MoE-288, with no NaN/regressions.  Added to CTest as
`moe_e2e_coherence_parity`, fails the build if violated.

**Context brief:** the gate from `task_gpu_miss.txt` (<= 1e-3 L1 diff).  Runs on gfx1201
(hardware only).

**Task list:**
- `src/kernels/moe_e2e_coherence_parity.cpp`: instantiate the fused MoE pipeline over the
  real model weights directory, run N tokens, diff each layer's output vs the CPU
  `drive_pool_multi` path.
- register + run on gfx1201; add a CPU-only build guard so the test compiles (but is a
  no-op / skipped) on the gfx1031 tier and on Windows-without-GPU.

**Verification commands:**
- `ctest --test-dir build_gfx1201 -R moe_e2e_coherence_parity --output-on-failure`

**Exit criteria:** gate green; `PASS` printed; registered for CI (see Step 8).

---

## Step 7 — Benchmark to goal (measured, not fabricated)

**Goal:** report measured decode tok/s on gfx1201 vs current 7.6-11 tok/s, confirm
>= 50 tok/s, and report where it is if short.  Honest measurement only.

**Context brief:** gate against `task_gpu_miss.txt`'s "decode 7.6-11 tok/s, prefill 14.3"
baseline.  Measure with the same harness (single-stream, n_embd=2048, k=6).

**Task list:**
- `bench/micro/moe_decode_throughput.cpp`: time 50 decode tokens end-to-end with the
  fused GPU path, print tok/s, PCIe-read bytes, cache-hit vs miss split, and the L1
  vs CPU (parity re-asserted in the same run so the number is honest).
- register as a `bench`-only target (not a test); document the run command.
- run on gfx1201, record the real number in `maps/moe_gpu_miss.md`.

**Verification commands:**
- `cmake --build build_gfx1201 --target moe_decode_throughput`
- `build_gfx1201\bench_micro_moe_decode_throughput.exe`

**Exit criteria:** a real measured tok/s printed; `maps/moe_gpu_miss.md` updated with
the number and the remaining gap (if any); no fabricated speedup.

---

## Step 8 — CI / portability guard (Windows CPU + gfx1031 exclusion)

**Goal:** keep the GPU MoE path Windows/MSVC/D3D12-first (per `AGENTS.md`) and keep the
portable parity tests from trying to run a GPU-only pipeline on the gfx1031 tier or on
CPU.  The `moe_e2e_coherence_parity` gate from Step 6 is GPU-only on gfx1201.

**Task list:**
- guard the new `.cu` sources and the parity/exe targets behind `if (ROCM_ENABLED)` and
  the gfx1201 arch in `CMakeLists.txt`, mirroring the existing `strata_kernels` pattern.
- add a `ctest` label `gpu-only` / `gpu-moe` so the portable `ctest` run never invokes a
  GPU binary; document the run command for this tier (`ctest --test-dir build_gfx1201 -L gpu-moe`).
- verify: `cmake --build build_gfx1201 --target dxgt_tests` clean (portable tier unaffected);
  `ctest --test-dir build_gfx1201 -L gpu-moe` runs the GPU gates.

**Verification commands:**
- `cmake --build build_gfx1201 --target dxgt_tests 2>&1` (still builds)
- `ctest --test-dir build_gfx1201 -L gpu-moe --output-on-failure`

**Exit criteria:** build green on gfx1201; GPU tests labeled and gated; portable tests
unaffected.

---

## Dependency graph

```
Step 1 (map boundary)       [PREREQUISITE, no code]
   |  \  \
   |   \  \-- Step 4 (cache/REBAR load)   [parallel w/ Step 2]
   |    \
   |     Step 2 (fused GEMM, one expert)  [depends on 1]
   |      |
   |      +-- Step 3 (dispatch fusion)   [depends on 2]
   |
   +-- Step 5 (KV-VRAM staging)           [depends on 1 only; independent of 3/4]

Step 6 (e2e coherence)                   [depends on 2,3,4,5]
Step 7 (benchmark)                       [depends on 6]
Step 8 (CI/portability guard)            [depends on all targets landing; can fold into each PR]
```

Critical path: 1 → 2 → 3 → {4 integrated}, 5; then 6 → 7, with 8 folded into each step's branch.

## Parallelism / ordering summary
- Serial prerequisite: Step 1.
- Parallel after Step 1: Step 2 (fused GEMM parity) and Step 5 (KV-VRAM) have DISJOINT
  files (`moe_fused_*.cu` vs KV staging) and no shared output → run in two worktrees /
  two tasks.
- Step 4 (cache miss) can prototype its REBAR-read parity test independently of Step 3's
  dispatch shape, then merge.
- Step 3 must follow Step 2 (uses the fused GEMM).
- Steps 6-8 are gated on everything.

## Rollback
- If Step 2 parity fails: the fused GEMM is a standalone kernel + test; unshipped, CPU
  miss path (`drive_pool_multi`) still runs — no regression to the live path.
- If Step 6 coherence fails at any sub-step: the new kernel is not wired into
  `generate.cpp`'s dispatch table (the wiring commit lands LAST in Step 3, behind the
  parity gates) — so a parity failure leaves performance unchanged, not correctness broken.

## Adversarial review (self, against the checklist)
Critical findings addressed:
1. **Numerics trap (HIGH):** the GPU fused path must NOT quantize fp16 activations to
   int8 (the FP4 tile caught the raw-4-bit-dot trap; the same hazard exists for Q4_K:
   `{0,1,2,3,4,6,8,12,...,-12}` is nonlinear and out of 4-bit range).  Steps 2-4
   decode Q4_K/Q5_0 to fp32 scalar-then-dot, parity-tested vs the CPU reference.  FIX:
   no int-quantization of activations; codebooks via the validated Q4_K/Q5_0 decoder.
2. **Coherence vs CPU (HIGH):** `task_gpu_miss.txt` requires <= 1e-3 L1.  Every GPU step
   has an explicit parity test vs the CPU `drive_pool_multi` output.  FIX: Steps 2,3,4,6
   each carry the 1e-3 L1 gate.
3. **REBAR staging (MEDIUM):** reading uncached experts over REBAR must not introduce a
   host staging copy (defeats the 4 GB/s budget).  FIX: kernel reads `PinnedArena` host
   pointer directly; Step 4 test counts PCIe-read bytes to prove no staging copy.
4. **GPU idle window (MEDIUM):** the CPU miss path takes 57 ms while GPU is idle — the
   fused kernel must not reintroduce a CPU round-trip per expert.  FIX: Step 3 dispatch
   is one kernel launch per MoE layer; host-side only sequences launches.
5. **Portability (LOW):** `AGENTS.md` says GPU code is `#ifdef _WIN32`/`D3D12`-first and
   portable tests excluded on non-Windows.  The gfx1201 ROCM build here is a research
   tier; the live product builds MSVC/D3D12.  FIX: Step 8 guards the `.cu` under
   `ROCM_ENABLED` and labels CTest `gpu-moe`; portable tests stay green.

No strongest-model sub-agent was wired in this harness (the blueprint skill delegates
adversarial review to one); this review is self-done against the checklist above.  Re-run
with an external reviewer before finalizing Step 2 if available.

## Step count & parallelism
- 8 steps. Two naturally parallel pairs: (2, 5) after Step 1, (3 vs 4-prototype).
- Estimated effort: ~2.5-4 agent-weeks (Step 1 mapping ~0.5w; Steps 2,4,5 ~0.5w each;
  Step 3 ~1w; Steps 6,7,8 ~0.25w each).
