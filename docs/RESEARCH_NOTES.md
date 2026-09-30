# Research Notes: FP4, MoE Streaming, and KV/State Quantization

Status: synthesis of a 45-paper corpus. Every numeric claim below carries a
verification state. Nothing here is a placeholder.

## 0. Which hardware this analysis is written against

**Read this first. The corpus was filtered against gfx1031, and the host we
actually run on is gfx1201.** The verdicts are only as good as that filter, so
the premise is stated up front rather than buried in section 1.

`CMakeLists.txt` builds **both** `gfx1031` and `gfx1201` by default, so the
project is dual-target. This document was written from the gfx1031 lens. Any
verdict whose stated reason contains "gfx1031", "no WMMA", "no tensor core",
"12 GB", or a bandwidth figure is **premise-dependent** and must be re-read for
gfx1201 before the work is scheduled. Section 2 separates the two tiers.

### Measured on this host (2026-09-30, `strata-device --selftest`)

| Property | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT |
| Arch | gfx1201 (RDNA4), compute capability 12.0 / sm_120 |
| SMs | 32 |
| VRAM | 15.922 GiB total (17,095,983,104 B) |
| Driver / runtime | 71260602 / 71260602 |
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads (Zen 3 — **no AVX-512**) |
| System RAM | 95.9 GB |
| ROCm (this tier) | `G:\ROCM10RT-gfx1201` |

Not measured on this host, and deliberately not guessed: **VRAM bandwidth**.
The `~350 GB/s` figure quoted throughout this document belongs to the gfx1031
tier, not to this GPU. Measure it with the project's own bench harness before
any bandwidth-derived decision.

### Two known-bad premises this corrects

- **"RX 6700 XT, 12 GB" is not the machine under test.** That is the gfx1031
  tier, not this host. Any capacity, residency, or bandwidth arithmetic that
  assumes 12 GB is wrong here by 33% of VRAM.
- **The gfx1031 binaries cannot be executed on this host.** They are built for a
  different GPU, and a separate defect (see `docs/TODO.md`) makes their CPU-side
  kernels raise `STATUS_ILLEGAL_INSTRUCTION` (0xC000001D) here regardless. So
  "gfx1031 parity passes" is **not** re-verifiable from this machine; it can only
  be checked on gfx1031 hardware.

## 1. Scope and how to read this document

**Primary analysis tier.** AMD Radeon RX 6700 XT, gfx1031 (RDNA2), 12 GB VRAM,
48 GB system RAM, ~350 GB/s VRAM bandwidth. This was the binding constraint
assumed while reading the corpus. See section 0 — the host we run on is
gfx1201/16 GB, so treat gfx1031-specific verdicts as provisional.

**Target software.** `strata` (this repository), an OpenAI-compatible
CPU+ROCm hybrid inference engine that disk-streams MoE expert weights, keeps a
hot-expert LRU resident in VRAM, and quantizes the non-expert path to reduce
VRAM traffic.

**Verdict legend**

| Verdict | Meaning |
|---|---|
| `ADOPT` | Implement. Directly actionable with the code we already have. |
| `ADOPT-HYBRID` | Take the idea, not the artifact. The paper's kernel/hardware is wrong for us; the algorithm transfers. |
| `DEFER` | Sound idea, wrong time. Park it with a named trigger for revisiting. |
| `REJECT` | Does not apply to our scope. **If the stated reason is a gfx1031 property, it is only rejected for that tier** — re-check before treating it as settled on gfx1201. |

**Verification legend**

| State | Meaning |
|---|---|
| `VERIFIED` | Digits appear verbatim in the arXiv abstract or the fetched body text. |
| `MATH-MANGLED` | The claim exists but the number sat inside a LaTeX span that our HTML-to-text conversion dropped. Value is plausible but **unconfirmed**. |
| `TABLE-SOURCED` | Number lives in a figure/table that does not survive text extraction. Unconfirmed. |
| `MISATTRIBUTED` | The number does not appear in this paper at all. It came from a different paper. |

Verification method is in section 8. Per-paper verification counts are in
section 6.

## 2. Hardware reality check (the filter applied to every paper)

These are the facts assumed while reading the corpus. They disqualify a large
fraction of the literature on contact. **Read the tier column before acting on
any of them** — three of the five facts are true of gfx1031 and *unverified*
for the gfx1201 host we actually run on.

| Fact | gfx1031 (analysis tier) | gfx1201 (this host) |
|---|---|---|
| FP4 / INT4 matmul hardware | none; no FP4 tensor core, no WMMA-INT | **UNVERIFIED** — see below |
| WMMA | absent | present in hardware; **the project uses no WMMA at all** |
| 2:4 structured sparsity | no benefit | **UNVERIFIED** |
| Native MXFP4/NVFP4 unpack | none, software only | **UNVERIFIED** |
| VRAM | 12 GB | 15.922 GiB measured |
| Bandwidth | ~350 GB/s assumed | **not measured** |

- **No FP4 or INT4 matrix-multiply hardware (gfx1031).** `__dp4a` (INT8
  dot-4-accumulate) and `sdot4` are the integer fast paths, and both tiers have
  them. Any paper whose contribution is "run FP4 on the FP4 tensor core" is
  `REJECT` *as a kernel* on gfx1031, regardless of how good the algorithm is.
- **The project emits no WMMA.** Grepping the tree finds no
  `__builtin_amdgcn_wmma*` use, so on gfx1201 the WMMA path is available but
  unclaimed. Whether gfx1201 has a *usable* FP4 path is **UNVERIFIED** and is
  worth one probe kernel before any FP4 work is scheduled. Verify with
  `rocminfo | grep -i -E 'fp4|int4|wmma'` on gfx1201 hardware, or by compiling a
  single `_builtin_amdgcn_...` probe and reading the ISA from the disassembly —
  do not infer it from RDNA generation alone.
- **No native MXFP4/NVFP4 unpack path (gfx1031).** Every FP4 weight must be
  converted in software. This makes *software-only* FP4 error-reduction
  techniques unusually valuable, because we are already paying the conversion
  cost. Unverified for gfx1201.
- **No sparse tensor cores (gfx1031).** 2:4 structured sparsity gets no hardware
  benefit. Unverified for gfx1201.
- **Bandwidth-bound, not compute-bound (gfx1031).** At 350 GB/s VRAM and
  ~1.5-2 TB/s host-to-device, almost every win in this corpus is a *traffic* win,
  not a FLOP win. Papers reporting "throughput" on B200/B300/H100 are reporting
  a different bottleneck entirely. **This reasoning is unchanged on gfx1201 in
  direction but not in magnitude** — the absolute figure is unmeasured.
- **VRAM is 12 GB (gfx1031); 15.922 GiB here.** This is why the engine streams
  experts from host memory and disk on the small tier. The 16 GB host has more
  room, which shifts the expert-cache/ring tuning thresholds rather than
  removing the need to stream.

## 3. Authoritative format facts (verified from source, not from papers)

These come from the local ggml checkout, not from any paper. Several agent
reports contradicted them, so they are recorded here as the tiebreaker.

`ggml/src/ggml-common.h`:

```c
#define QK_MXFP4 32
typedef struct {
    uint8_t e;              // E8M0 scale
    uint8_t qs[QK_MXFP4/2]; // 16 bytes, two E2M1 values per byte
} block_mxfp4;              // sizeof == 17

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // 4 UE4M3 scales, one per 16-element sub-block
    uint8_t qs[QK_NVFP4/2];           // 32 bytes, packed E2M1
} block_nvfp4;              // sizeof == 36
```

E2M1 codebook (`ggml-common.h`, `int8_t kvalues_fp4[16]`):
`{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}`.

**The nibble order is a split half, not an interleave** — the single easiest
thing to get wrong here, because "low nibble first" reads as if byte *j* holds
elements `2j`/`2j+1`. It does not. From `dequantize_row_mxfp4` /
`dequantize_row_nvfp4` in `ggml-quants.c`:

```c
// MXFP4: y[i*32 + j] = low nibble, y[i*32 + j + 16] = high nibble
const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];
y[i*qk + j + 0   ] = x0*d;
y[i*qk + j + qk/2] = x1*d;          // qk/2 == 16

// NVFP4: per sub-block s, byte index s*8 + j
//   yb[j] = low nibble, yb[j+8] = high nibble, where yb = y + i*64 + s*16
const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];
yb[j + 0       ] = v0*d;
yb[j + qk_sub/2] = v1*d;            // qk_sub/2 == 8
```

So MXFP4 is exactly the Q4_0/IQ4_NL split-half convention this codebase already
implements, and NVFP4 is the same idea 16 elements at a time. An interleave
reading is bit-plausible and numerically wrong, and it survives any spot check
that only inspects element 0.

**Both scale decoders halve their input**, because `kvalues_fp4 = 2 *
E2M1_float`. From `ggml-impl.h`:

```c
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {   // == 2^(x-128)
    uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float result; memcpy(&result, &bits, sizeof(float)); return result;
}

static inline float ggml_ue4m3_to_fp32(uint8_t x) {       // == 0.5 * ue4m3(x)
    if (x == 0 || x == 0x7F) return 0.0f;                 // 0x7F is the NaN encoding, zeroed on purpose
    int exp = (x >> 3) & 0xF, man = x & 0x7;
    float raw = exp == 0 ? ldexpf((float) man, -9)
                         : ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}
```


Overhead:

| Format | Bytes | Bytes/element | Relative to 0.5 |
|---|---|---|---|
| MXFP4 | 17 / 32 | 0.53125 | 1.0625x |
| NVFP4 | 36 / 64 | 0.56250 | 1.1250x |

**Corroboration.** Paper 2609.24519 independently states that FP4 values scaled
by 2 form the set `S = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`, which is exactly the
ggml E2M1 codebook with signs, and that `S` contains every residue modulo 13.
The codebook in the local checkout is correct.

**Correction to agent output.** One agent report described NVFP4 as
"17 bytes per 16 elements, 1.0625 bytes/element". That is wrong by 47%. The
real figure is 36 bytes per 64 elements. Any capacity or bandwidth estimate
built on the agent's number is void. This is why the layout above is sourced
from code rather than from a paper or a model.

**No FP4 `vec_dot` exists upstream.** ggml provides reference quant/dequant for
both formats but has no FP4 vector-dot kernel, so there is no upstream
implementation to inherit or validate against. Ours would be the first in this
codebase, which raises the bar for tests accordingly.

## 4. Cross-paper synthesis

### 4.1 The single most important finding: INT8 activations, not FP4 activations

Multiple papers converge, from different directions, on the same conclusion
that happens to be exactly right for gfx1031:

- **2606.12280** (Ideogram 4.0 on **RTX 3090**, a GPU with *no FP8 tensor
  cores*) reports an INT8 W8A8 recipe that is statistically indistinguishable
  from FP8 on CLIP and PickScore (paired bootstrap CIs include zero), within
  ~0.004 HPSv2, and is the most faithful reproduction of the FP8 reference
  (LPIPS 0.243 vs 0.277/0.306 for half-size 4-bit baselines).
  `VERIFIED`. The hardware situation is deliberately the same as ours.
- **2510.25602** finds that at 8 bits with block size 32, **MXINT8 beats its FP
  counterpart in both algorithmic accuracy and hardware efficiency**, while at
  4 bits FP usually holds the accuracy advantage, and that NVINT4 can surpass
  NVFP4 once Hadamard rotation is applied. `VERIFIED`.
- **2606.23406** finds that after a Randomized Hadamard Transform, **int8 beats
  fp8 on the lattice output** at the 8-bit MMA path, and separately that
  int8 beats fp8 at 4-bit MMA precision. `VERIFIED`.
- **2603.02731** (Hopper, training) makes the complementary point: keep core MoE
  math in FP8/INT8 and use FP4 only to *compress* activations and
  expert-parallel traffic. `VERIFIED`.
- **2609.33135** (MpFA, Blackwell) shows the converse failure mode: making
  attention *fully* FP4 fails to convert tensor-core throughput into end-to-end
  speed, because post-softmax online quantization and the PV conversion dominate.
  Their fix is to keep QK in FP4 but PV in FP8. `VERIFIED`.

**Conclusion.** Weights are 4-bit; activations are INT8; math is `__dp4a`.
This is not a compromise forced by missing hardware — it is the configuration
the literature independently converges on, and the one whose only missing piece
for us is a software unpack kernel. Our existing plan is correct and now has
multi-paper support.

### 4.2 MXFP4 needs help; NVFP4 does not get one for free

- **2509.23202** (MR-GPTQ) is the sharpest warning: state-of-the-art PTQ
  "struggles with FP4" for two structural reasons — **NVFP4's small group size
  provably neutralizes traditional outlier-mitigation techniques**, and
  **MXFP4's power-of-two scale quantization severely degrades accuracy due to
  high induced error**. `VERIFIED`. Their fix is block-wise Hadamard rotation
  plus format-specific GPTQ, fused into the weights offline.
- **2603.08713** supplies the software-only complement: Overflow-Aware Scaling
  (OAS) and Macro Block Scaling (MBS) reduce the end-to-end accuracy gap
  between MXFP4 and NVFP4 **from about 10% to below 1% on average, for 6.2%
  average GEMM overhead**, with no hardware change. They also cite 12% relative
  tensor-core area savings for MX. `VERIFIED`.
- **2603.13180** (MXNorm) shows there is free accuracy available from the scale
  field we already store: estimating RMS from the MXFP block scales gives a
  **32x decrease in the reduction needed for normalization**, 2.4x kernel
  speedup over RMSNorm under `torch.compile`, worth 1.3% of Llama-3-8B layer
  time at MXFP8 and 2.6% at NVFP4. `VERIFIED`.

**Conclusion.** OAS/MBS belong in our MXFP4 dequant path. MXNorm's trick is free
for us because we hold the block scales in registers anyway. Hadamard rotation
is worth doing offline (fused into weights, zero runtime cost) but must be
evaluated against the rotation's interaction with the fixed E2M1 grid.

### 4.3 Layer sensitivity is measurable and not what people assume

- **2603.08747** (NVFP4 and MXFP4, Qwen2.5 0.5B/7B/14B) finds MLP up- and
  down-projections dominate sensitivity, gate projections are moderately
  sensitive, **attention projections are the least sensitive**, and
  **sensitivity does not localize to final blocks — early blocks can be highly
  sensitive, especially under MXFP4**. `VERIFIED`.
- **2504.14152** (FGMP) gives the selection criterion: weight the per-value
  perturbation by **Fisher information** to decide which blocks keep higher
  precision, plus a sensitivity-weighted clipping step. Their hardware is an
  ASIC augmentation, so the hardware half is `REJECT`, but the selection policy
  is pure software. Their prototype result — <1% perplexity degradation versus
  an all-FP8 baseline at 14% less energy and 30% less weight memory — is
  `VERIFIED` as a number but not portable to us as a hardware claim.
- **2605.12327** ("Grid Games") studies choosing a different scale *grid* per
  group, extending microscaling. Its abstract is **truncated in the arXiv
  metadata we fetched**, so we have the premise and not the result. Flagged
  unresolved rather than summarized.
- **2608.03867** (AdaMX) makes the heterogeneity point: MX formats fix either the
  element format or the precision-recovery scheme across all blocks, capturing
  only limited heterogeneity; selecting per block and per operand at equal
  equivalent bit-width removes 83% of MXFP4 accuracy loss on commonsense tasks
  and 82% on MMLU (43%/27% for NVFP4). Validated on a 22nm ASIC prototype, so
  hardware is `REJECT`, principle is `ADOPT-HYBRID`.

**Conclusion.** We should not blanket-quantize. A Fisher-weighted
sensitivity pass feeding a mixed-precision policy (higher precision on MLP
down-projections and early blocks, INT8-friendly path elsewhere) is the
supported design, and it is software-only.

### 4.4 Prefill and decode want different quantizations — and this is our architecture

Two independent papers land on the same phase split, and our engine is already
built around exactly that split (disk-streamed prefill, hot-expert decode):

- **2605.20315** (Mix-Quant): quantizing the *entire* inference process causes
  significant degradation, but **the prefilling stage has substantial
  quantization redundancy and can be quantized with minimal accuracy loss
  despite being the dominant compute cost**. Applying NVFP4 to prefill while
  keeping decode in BF16 gives **up to a 3x prefill speedup** and largely
  preserves agentic task performance. `VERIFIED`. They evaluate on Qwen3.5-9B
  and **Gemma-4-26B-A4B-it**, one of our target architectures.
- **2609.26333** (Disaggregated Quantization) makes the mechanism explicit:
  "prefill and decode reward different approaches to quantization —
  **low-precision arithmetic accelerates prompt processing, while compact
  weights reduce memory traffic during generation**." Training separate
  compute-native prefill weights against released GGUF decoders improves
  1-bit accuracy by **32.5 points on MMLU-Pro and 35.3 on MMMU-Pro**. Their
  offloaded disaggregated prefill (ODP) **streams its weights from SSD,
  amortizing loading over prompt length**, and delivers a **1.78x
  time-to-first-token speedup** over the weight-only baseline at 8K prompt
  length in llama.cpp. `VERIFIED`.
- **2609.34117** (SlimWise) is the decode-side mirror: prefill with the full
  expert pool, decode with a **pruned** pool, and **directly reuse the
  prefill-generated KV cache without conversion**. Up to **1.81x decode
  throughput at 50% expert pruning** with minimal accuracy loss, training-free
  handoff; they add an optional low-cost distillation stage for the residual
  gap, and warn that benchmark accuracy conceals pruning-induced changes in
  generation length. `VERIFIED`.

**Conclusion.** The strongest actionable synthesis in the corpus: keep prefill
on the compute-friendly low-precision arithmetic path (our disk-streamed
expert fetch is already prefiller-shaped), keep decode on the weight-compaction
path (our pinned hot-expert LRU is already the decode-side resource), and
never convert KV between phases. A pruned decode expert pool is a natural
extension of the LRU we already maintain.

### 4.5 Expert streaming: the closest match to our actual bottleneck

- **2603.19289** (Speculating Experts) is the only paper in the corpus that
  attacks exactly our problem. In memory-constrained MoE inference where expert
  weights are offloaded to CPU, they use **currently computed internal model
  representations to speculate future experts**, letting transfers overlap
  computation. Future experts are reliably predictable; executing speculated
  experts preserves downstream accuracy, removing the need to re-fetch
  true router-selected experts. **Up to 14% reduction in TPOT** over on-demand
  loading from CPU memory. `VERIFIED`. They also ship lightweight estimators
  when speculation alone degrades accuracy.
- **2603.10031** (AMD MI325X, the closest hardware paper we have) shows the
  AMD shape of the problem: all four models hit a **common throughput
  saturation point consistent with a memory-bandwidth bottleneck** — ~500
  concurrent for short sequences, ~100-200 for longer. `VERIFIED`.
- **2609.33889** gives the arithmetic for deciding where to spend sparsity
  effort: derive the **byte crossover** at which projection-traffic savings and
  KV-traffic savings are equal, from model dimensions and keep ratios alone.
  Their byte account predicted measured crossings for three keep-ratio pairs, a
  second model, and a second GPU **to within 4.1K tokens**. They also show
  activation sparsity composed with attention-scored KV selection decodes
  **14-26% faster than the best single branch** at matched perplexity, and warn
  that timing a masked dense baseline instead of split-K inflates a KV policy's
  apparent speedup **about fivefold**. `VERIFIED`.

**Conclusion.** Adopt expert prefetch speculation (14% TPOT, `VERIFIED`) and
adopt the byte-crossover formula as the *decision procedure* for whether
activation sparsity or KV sparsity is worth it on a given model at a given
context length. That formula is model-and-context parameterized, so it is
directly reusable for our 12 GB budget.

### 4.6 MLA is a different animal from GQA — and we have an MLA target

- **2603.10031** on MI325X, stated plainly: **MLA models require block size 1
  and cannot use KV cache offloading, while GQA models benefit from both.** The
  AMD AITER runtime is required for competitive MLA throughput and **must be
  selectively disabled for architectures with incompatible attention head
  configurations**. A controlled ablation (n=5/condition) shows AITER gives a
  modest 3-5% throughput benefit at high concurrency but **2-16x higher
  measurement variability**, confirming its large speedups target MoE/MLA
  kernels specifically. `VERIFIED`.
- **2609.36760** (QuantMLA) models MLA's **dual-path** quantization error —
  content cache and decoupled RoPE key cache behave differently, with
  **pronounced amplification of RoPE-path error**. Path-specific transformations
  are derived to be fusible into model parameters offline, so there is no
  online transformation overhead. Joint INT4 content+RoPE caching with minimal
  degradation, content down to INT2 while RoPE stays INT4, and a native kernel
  that **integrates unpacking and dequantization directly into attention
  computation**. **3.59x cache compression at 128K context** and **5.168x higher
  whole-job output throughput than BF16** in a cache-pressure workload.
  `VERIFIED` (abstract, both figures).
- **2609.15030** is a GLM-5.3-Flash engineering report on exactly the model class
  we target, and its headline is a correctness bug: a **complete-hit recovery
  mismatch restored state for the full prompt while the scheduler credited one
  fewer token**. They fixed it with **strict-prefix lookup**, matched checkpoint
  scheduling, and fixed per-rank kernel configurations, taking agreement with a
  recomputation control from **34/36 to 36/36** generations of 64 token IDs
  each. CPU reload cut TTFT by **46-64%** and total request time by **1.9-7.0%**
  versus modified cold recomputation. `VERIFIED`.

**Conclusion.** Three concrete obligations for our GLM/deepseek2 path:
(1) treat MLA KV state as non-offloadable and design around a resident latent
cache, exactly as 2603.10031 reports; (2) if we ever quantize the MLA cache,
quantize the two paths separately and fuse unpack into attention rather than
pre-converting, following QuantMLA; (3) implement strict-prefix cache alignment
and per-rank kernel pinning as a correctness gate before trusting any
resume or hybrid-state result, following 2609.15030.

### 4.7 Recurrent state must stay high precision — and now we have two papers saying so

Our existing invariant is that GDN/SSM recurrent state stays F32 even when GDN
*weights* tolerate W4A4. Two papers in the corpus independently justify it,
and they are easy to confuse with a different claim, so stating the
distinction precisely:

- **2608.27513** (DAMP): uniform quantization of GDN/KDA recurrent state gives
  a poor accuracy/storage trade-off — **INT8 and FP8 already degrade accuracy
  on complex reasoning tasks, while INT4 and NVFP4 reduce it to near zero**.
  Needs **9.9 bits per state value** to stay close to FP32. Yields 69.1% state
  storage reduction, 2.01x faster state-update kernel, 10.9% lower full-model
  TPOT. `VERIFIED` (69.1%, 9.9, 10.9 in abstract; 2.01 `MATH-MANGLED`, in body
  as "[math]").
- **2609.38169** (STEPQuant): error impact depends on time (long-lived memory
  retains error across many decode steps) and space (key rows differ in impact;
  magnitudes vary along rows and columns). Matches FP32-state accuracy only
  under a nominal **6-bit** budget, and beats uniform INT8 at 4 bits. Over 5x
  state compression and 68.7% serving-memory reduction at 6 bits. `VERIFIED`
  (abstract).
- **2609.04098** (Minima) is the *complementary* claim and is about a different
  object: **NVFP4 W4A4 on all 496 linear layers of Qwen3.8-27B, GDN block
  included**, matches BF16 within seed noise (5-task average -0.52) at 17.5 GiB
  with 14-19% faster prefill. The fragile part people protect — the decay and
  write-strength **gates** — turns out to be *least* sensitive: softplus/exponential
  and sigmoid parameterizations compress **~11% GEMM error to ~2% output
  error**. `VERIFIED`.

**Conclusion, stated precisely: quantize the recurrent half's WEIGHTS to 4-bit;
do not quantize its STATE.** DAMP and STEPQuant are about state and both say
low precision fails. Minima is about weights and says 4-bit is fine. These are
consistent, not contradictory, and conflating them would be the most damaging
possible reading of this corpus. We keep F32 state.

Minima contributes one more directly applicable warning: they **repair a
global-scale mismatch that arises when per-module-calibrated NVFP4 checkpoints
are served by kernels that fuse those modules into one GEMM**. We fuse. That
bug class is ours by construction, and it must have a test.

### 4.8 What the corpus says about not being clever with attention

- **2609.37261** proves, under standard complexity assumptions, that **no truly
  subquadratic algorithm can approximate attention with any nontrivial additive
  or relative guarantee uniformly over all inputs** — and this survives
  polynomial preprocessing of the KV cache. `VERIFIED`. Practical reading: do
  not chase a uniform-guarantee fast attention; chase data-dependent,
  input-specific savings like the ones in 4.5 and 4.6, and validate them
  against perplexity and task quality directly.
- **2609.33410** (FoldAttention) shows the reference for softmax need not be
  discovered: fix a finite reference `Z_i` in advance, since softmax is
  invariant to a common shift, and each weight `2^(s_ij - Z_i)` is final when
  computed, so contributions add across disjoint key ranges without rescaling.
  On Hopper this gives 1.36-2.30x over the fastest BF16 baseline, and it also
  makes the backward deterministic because CTAs round bounded partial gradients
  onto an integer grid declared before the reduction. `VERIFIED`. The Hopper
  kernel is `REJECT`; the math is ISA-independent and is attractive for our
  long-context decode, which has no fast rescale path.
- **2609.33334** (DGE) makes a point worth internalizing before we ever build KV
  eviction: evicting at end of prefill fails in two distinct ways —
  **compensation** (restoring evicted attention mass recovers the
  attention-level target without recovering task quality) and **selection**
  (covering more true decode-query mass can *hurt* when recovered mass is
  fragmented rather than concentrated). Deferring eviction by drafting the
  first k=2 answer tokens fixes both, reaching 44.2 on LongBench against
  FullKV's 44.3, and their timing-only control scores the same, so the gain is
  from *when* eviction happens, not *what* is selected. `VERIFIED`.
- **2609.34049** and **2609.36722** are both training-side or mechanism studies
  (sliding-window information relay; query-side low-rank adapters trained by
  distillation, up to 3.73x speedup, training fewer than 0.05% of parameters).
  `VERIFIED` as claims, `DEFER` for us.
- **2609.33477** (SuffixReplay) is the interesting hybrid-LLM case: linear
  attention states cannot be rolled back to arbitrary prefix boundaries, so
  instead of checkpointing, **replay only a recent suffix of input hidden
  states retained as anchors** and let the decay forget the distant past.
  91.4-100% of full-prefill quality, 0.36-0.51x the amortized per-token storage of
  SGLang's 8192-token checkpoint cache, 15-70% median TTFT reduction,
  2.3-4.3x throughput when the working set exceeds HBM. `VERIFIED`.

## 5. Papers that do not apply to gfx1031

These are not dismissed; they are excluded on a specific, stated ground, so
they can be revisited if the hardware or scope changes.

| ID | Paper | Why excluded |
|---|---|---|
| 2503.18773 | BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache | Contribution is using Tensor Cores with NVFP4/MXFP4 and Hopper warpgroup instructions. No FP4 TC on gfx1031. Layout and warp-level dequant parallelism are `ADOPT-HYBRID`. |
| 2504.14152 | FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference | Hardware augmentation (datapath + mixed-precision activation quantizer). ASIC design. Selection policy is `ADOPT-HYBRID`. |
| 2509.23202 | Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization | Performance claims on B200/RTX5090. Algorithmic warning about MXFP4 power-of-two scales is `ADOPT`. |
| 2601.09527 | Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs | RTX 5060 Ti/5070 Ti/5090, NVFP4 tensor cores. Deployment methodology is reusable. |
| 2603.02731 | Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs | Training recipe, and FP4 there is for compression of activations/EP traffic with FP8 core math. |
| 2603.05232 | SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity | 2:4 sparse Tensor Cores. No sparse TC on gfx1031. |
| 2605.06067 | Normalized Architectures are Natively 4-Bit | nGPT hypersphere training-time robustness. Not applicable to inference of pretrained models. |
| 2605.31035 | MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations | Repurposes the sign bit of the FP8 E4M3 block scale to select a micro-format; needs the NVFP4 MMA path. |
| 2608.06812 | DGEMM with Ozaki Scheme I/II on FP4 Tensor Cores: A Base-13 E2M1 Limb Representation | Base-13 E2M1 limb representation executed *on FP4 tensor cores*. No such hardware. |
| 2608.17071 | KernelArc: A Multi-Agent Framework for GPU Kernel Optimization | Multi-agent optimization framework evaluated on H100/B200 SOL-ExecBench. Process inspiration only. |
| 2609.24519 | AWE: Adaptive Weight Encoding for Exact Integer Matrix Products with Fewer GEMMs on FP4 Tensor Cores | Fewer-GEMM encodings for exact integer products on FP4 tensor cores. Source of the E2M1 residue-mod-13 corroboration in section 3. |
| 2609.33135 | MpFA: Hardware-Efficient Train-Free QK4V8 FlashAttention Kernels on Blackwell GPUs | QK4V8 FlashAttention on Blackwell NVFP4/FP8 tensor cores. Mixed-precision QK/PV conclusion is `ADOPT-HYBRID`. |
| 2609.33496 | Chameleon: Dynamic Format Adapter for Efficient Diffusion | Diffusion models (SDXL, PixArt-alpha). Wrong domain. |
| 2609.36722 | ATTUNER: Recomputation-Free KV Cache Reuse via Query-Side Adaptation | Requires training query-side adapters. |
| 2609.37261 | Efficiently Approximating Attention Is Hard | Complexity-theoretic negative result. No implementation content. |
| 2609.37693 | FP64 Is All You Want, INT8 Is All You Need, FP4/6/8 Is All You Have | Ozaki-scheme GEMM-count optimization on B200/B300/RTX PRO 6000. |

## 6. Master index: all 45 papers

`Verified` = fraction of the numbers this study cited that were located
verbatim in the arXiv abstract or fetched body.

| ID | Title | Theme | Verdict | Key claim | Verified |
|---|---|---|---|---|---|
| 2503.18773 | BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache | KV | ADOPT-HYBRID | 7.5x avg decode speedup vs FP16 FlashDecoding-v2; up to 8.6x on Blackwell NVFP4; 4.3x over SOTA; 3x lower single-batch latency at 128K on LLaMA-3.1-8B | 8/8 |
| 2504.14152 | FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference | Mixed precision | ADOPT-HYBRID | Fisher-weighted block selection; <1% Wikitext-103 PPL degradation vs all-FP8, 14% less energy, 30% less weight memory | 4/4 |
| 2509.23202 | Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization | FP4 | ADOPT | NVFP4's small group size provably neutralizes outlier mitigation; MXFP4 power-of-two scale quantization severely degrades accuracy; MR-GPTQ 3.6x layer / 2.2x e2e on B200, 6x/4x on RTX5090 | 6/6 |
| 2510.25602 | INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats | Format | ADOPT | At 8 bits with block size 32, MXINT8 beats FP on accuracy and hardware efficiency; FP often wins at 4 bits; NVINT4 can surpass NVFP4 with Hadamard rotation | n/a (no numbers cited) |
| 2601.09527 | Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs | Deployment | DEFER | RTX 5090 gives 3.5-4.6x throughput over 5060 Ti with 21x lower RAG latency; NVFP4 1.6x over BF16 at 41% less energy, 2-4% quality loss; $0.001-0.04/Mtok electricity-only | 8/8 |
| 2603.02731 | Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs | Training | REJECT | Direct FP8-to-FP4 conversion; core MoE math in FP8; at 671B, 14.8% (11.8 GB) lower peak activation memory, 1157 to 1302 tok/s/GPU (+12.5%) | 4/4 |
| 2603.05232 | SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity | Sparsity | REJECT | 2:4 collapses Qwen3 reasoning 54% to 15%; 6:8 preserves accuracy; measured 1.33x approaches theoretical 4/3 | 3/3 |
| 2603.08713 | Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction | FP4 | ADOPT | OAS + MBS cut the MXFP4-vs-NVFP4 end-to-end accuracy gap from ~10% to below 1% average, for 6.2% average GEMM overhead; 12% relative TC area saving for MX | 5/5 |
| 2603.08747 | Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4 | Sensitivity | ADOPT | MLP up/down-projections dominate sensitivity; attention least sensitive; sensitivity does not localize to final blocks, early blocks can be highly sensitive especially under MXFP4 | n/a (no numbers cited) |
| 2603.10031 | Architecture-Aware LLM Inference Optimization on AMD Instinct GPUs: A Comprehensive Benchmark and Deployment Study | AMD | ADOPT | MLA requires block size 1 and cannot use KV offloading while GQA benefits from both; AITER needed for MLA, must be disabled for incompatible head configs; AITER gives 3-5% at high concurrency with 2-16x higher variability; common bandwidth bottleneck at ~500 (short) / ~100-200 (long) concurrency; 15,944 and 15,343 tok/s; Qwen3-VL-235B 47,873 vs Kimi-K2.5 7,327 tok/s; 18.9M tokens over 17,406 requests, 100% success | 8/8 |
| 2603.13180 | MXNorm: Reusing MXFP block scales for efficient tensor normalisation | Norm | ADOPT | RMS from MXFP block scales; 32x decrease in reduction size; 2.4x kernel speedup; 1.3% of Llama-3-8B layer time MXFP8, 2.6% NVFP4 | 6/6 |
| 2603.19289 | Speculating Experts Accelerates Inference for Mixture-of-Experts | MoE streaming | ADOPT | Predict future experts from current internal representations to overlap transfers; up to 14% TPOT reduction vs on-demand CPU loading; A6000, Qwen3-30B-A3B | 4/4 |
| 2605.06067 | Normalized Architectures are Natively 4-Bit | Training | REJECT | nGPT hypersphere constraint enables end-to-end NVFP4 training; 1.2B dense and 3B/30B hybrid MoE | 2/2 |
| 2605.12327 | Grid Games: The Power of Multiple Grids for Quantizing Large Language Models | FP4 | DEFER | Per-group selection of the scale grid is a natural extension of microscaling. **Abstract truncated in fetched metadata; result unknown** | n/a (no numbers cited) |
| 2605.20315 | Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs | Phase | ADOPT | Prefill has substantial quantization redundancy; NVFP4 prefill + BF16 decode gives up to 3x prefill speedup; evaluated on Qwen3.5-9B and Gemma-4-26B-A4B-it | 4/4 |
| 2605.31035 | MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations | FP4 | REJECT | Select E2M1 or E1M2 per block by repurposing the E4M3 scale sign bit; 3.1% area, 1.5% power overhead | 2/2 |
| 2606.12280 | Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs | Quant config | ADOPT | INT8 W8A8 on RTX 3090 (**no FP8 TC**) statistically indistinguishable from FP8 on CLIP and PickScore, within ~0.004 HPSv2; LPIPS 0.243 vs 0.277/0.306 for 4-bit; 55% OCR exact-match under JSON captions | 5/5 |
| 2606.23406 | HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline for Large Language and Diffusion Models | Quant | ADOPT | Beats HIGGS at every 3-5 bps operating point and TurboQuant/OCTOPUS down to 1.7 bps; on H100 at 4 bps compresses linear weights ~3.9x and KV ~3.79x near-losslessly; **int8 beats fp8 on post-RHT lattice output**; quantizes 19B LTX-2 DiT without artifacts | 5/5 |
| 2608.03867 | Heterogeneity-Aware Microscaling for Efficient Low-Bit LLM Inference | FP4 | ADOPT-HYBRID | Per-block precision-recovery and per-operand representation at equal equivalent bit-width; removes 83% MXFP4 loss (commonsense) / 82% MMLU, 43%/27% NVFP4 loss; on Gemma-4 12B leads MXFP4 on all four VL benchmarks, keeps up to 96% of FP16; 22nm ASIC, ~1% system energy | 6/6 |
| 2608.06812 | DGEMM with Ozaki Scheme I/II on FP4 Tensor Cores: A Base-13 E2M1 Limb Representation | Numeric | REJECT | Base-13 E2M1 limbs keep intermediate sums error-free in FP32; INT8 GEMM emulated bit-exactly on FP4 TC; competitive with FP8 Ozaki at 16384^3 on RTX PRO 6000 | 5/5 |
| 2608.11693 | Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8 Availability on NVIDIA Blackwell Ultra | Methodology | ADOPT (process) | B300 spec'd ~30:1 FP8:INT8; PTX never exposes tcgen05.mma .kind::i8 on sm_103a; CUTLASS skips INT8 UMMA for 103a; vLLM ships no INT8 GEMM for Blackwell and hard-fails at first forward; SGLang AOT INT8 stops at Sm90. **Direct justification for treating ROCmFPX as uncharacterized** | 4/4 |
| 2608.17071 | KernelArc: A Multi-Agent Framework for GPU Kernel Optimization | Methodology | DEFER | Ranked first on every representative L1/L2/Quantization/FlashInfer SOL-ExecBench task evaluated (Aug 20 2026 snapshot); BF16 GEMM, cuBLASLt Expert-API tables, fused MoE backward, shape-gated decoder fusion, native NVFP4 GQA, paged prefill | 4/4 |
| 2608.27513 | DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization | Recurrent state | ADOPT (evidence) | Uniform INT8/FP8 already degrade complex reasoning; INT4/NVFP4 near-zero; **9.9 bits** needed to stay near FP32; 69.1% state storage reduction, 2.01x state-update kernel, 10.9% TPOT | 4/4 |
| 2609.04098 | Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM | Hybrid GDN | ADOPT | **NVFP4 W4A4 on all 496 linear layers of Qwen3.8-27B (48 GDN, 16 attention), GDN included**; matches BF16 within seed noise (5-task avg -0.52) at 17.5 GiB, +14-19% prefill; gates least sensitive, ~11% GEMM error compresses to ~2% output error; repairs a global-scale mismatch when per-module-calibrated modules are fused into one GEMM; FP8 KV scales performance-free | 11/11 |
| 2609.04526 | Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs | Tooling | ADOPT (policy) | Naive merge deletes adaptation by **up to 39 pp** because against an on-grid base the optimum is the base; **never write back through a quantizer**; scale-only merge is bit-exact; drops STE cost 3.9x/step; ~125x faster scale-only task swap | 4/4 |
| 2609.14060 | AGENTQ: Quantization-Conditioned Backdoor Attacks on LLM Agents | Security | ADOPT (policy) | First QCA study against agents; up to **100% post-quantization ASR** across three trigger-action pairs and three codebooks (NF4, FP4, INT8) with minimal benign-utility loss | 3/5 (2 `MISATTRIBUTED`, see 7.1) |
| 2609.15030 | Validating Hybrid-State Cache Recovery for GLM-5.3-Flash with vLLM and LMCache | MLA/hybrid | ADOPT | Complete-hit recovery mismatch restored full-prompt state while the scheduler credited one fewer token; strict-prefix lookup plus matched checkpoint scheduling and fixed per-rank kernels took agreement 34/36 to 36/36 generations of 64 token IDs; CPU reload cut TTFT 46-64%, total request time 1.9-7.0% | 8/8 |
| 2609.15627 | DeepSeek-V4-Flash on AMD gfx90a: Correctness Recovery and Inference Performance Engineering | AMD MoE | ADOPT | Routed-expert W2 layout mismatch made a fast path numerically incorrect; repaired the output permutation at load time and added fixed-token and hash-based checks; on four MI250 GCDs TP4/EP1 decode ~74.5 tok/s, 4,604-token prompt 2.061 s TTFT (~2,234 input tok/s); limited by FP4 execution-format mismatch, low MFMA utilization, per-layer sync | 7/7 |
| 2609.24519 | AWE: Adaptive Weight Encoding for Exact Integer Matrix Products with Fewer GEMMs on FP4 Tensor Cores | Numeric | REJECT | FP4 x2 gives S = {0,±1,±2,±3,±4,±6,±8,±12} containing every residue mod 13; INT8xINT8 in 6 FP4 products, INT4xINT8 in 4; FP64 significand 75 products reduced to 59; boundary near input width 15 | 8/8 |
| 2609.26333 | Disaggregated Quantization: Specializing LLM Prefill and Decode | Phase | ADOPT | Low-precision arithmetic accelerates prompt processing, compact weights reduce generation traffic; NVFP4 prefiller improves 1-bit accuracy by **32.5 MMLU-Pro / 35.3 MMMU-Pro** over released GGUF decoders; ODP streams prefiller weights from SSD and gives **1.78x TTFT** at 8K prompt in llama.cpp | 3/3 |
| 2609.33135 | MpFA: Hardware-Efficient Train-Free QK4V8 FlashAttention Kernels on Blackwell GPUs | Attention | ADOPT-HYBRID | Fully FP4 attention fails to convert TC throughput to end-to-end speed (post-softmax quantization, tensor/shared movement, softmax contention); NVFP4 QK + FP8 PV; rank-one smoothing compensation; 2.81x output throughput over BF16 FA4 at 16K-128K; recovers 62.5% of accuracy loss at ~2.0% kernel overhead | 8/9 (1 terminology) |
| 2609.33334 | When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression | KV policy | ADOPT-HYBRID | Compensation: restoring evicted attention mass recovers the attention target but not task quality; Selection: fragmented recovered mass hurts; defer eviction to after drafting k=2 tokens; 44.2 on LongBench vs FullKV 44.3; timing-only control ties, so gain is from *when* not *what* | 6/6 |
| 2609.33410 | FoldAttention: Declared-Reference Softmax for Fast Decode and Deterministic Backward | Attention | ADOPT-HYBRID | Fix reference Z_i in advance so each weight is final when computed and contributions add without rescaling; final weights gate key/value reads; depth T cuts keys below 2^-T; on H100 at T=16, 1.36-2.30x over fastest BF16, up to 3.09x across MHA/GQA, within 1.5% error on six of seven; 1.14-1.30x reading every key; 1.46x whole decode step on Qwen3-8B; deterministic backward 1.84x over det. FA3/4 | 10/10 |
| 2609.33477 | Just Let Linear States Forget the Distant Past: Prefix Caching via Suffix Replay for Hybrid LLMs | Hybrid prefix | DEFER | Replay only a recent suffix of input hidden states retained as anchors instead of materializing recurrent-state checkpoints; 91.4-100% of full-prefill quality; 0.36-0.51x amortized per-token storage of SGLang's 8192-token checkpoint cache; 15-70% median TTFT reduction; 2.3-4.3x throughput when working set exceeds HBM | 8/8 |
| 2609.33496 | Chameleon: Dynamic Format Adapter for Efficient Diffusion | Diffusion | REJECT | Holds bit-width fixed and chooses the format per weight channel and per (layer, timestep bucket) from kurtosis and diffusion SNR; best FID in all six backbone x bit-width settings; CLIP within 0.24 of FP16; best at W4A8 | 3/3 |
| 2609.33889 | Where Activation Sparsity and KV-Cache Sparsity Cross in LLM Decoding | Sparsity | ADOPT | Byte crossover and ideal speedup bounds derived from model dimensions and keep ratios alone; predicts measured crossings for three keep-ratio pairs, a second model, and a second GPU within 4.1K tokens; masked-vs-split-K dense baseline inflates a KV policy's speedup ~5x; composition 14-26% faster than best single branch at matched perplexity | 4/4 |
| 2609.34049 | Thinking Outside the Box: Retention and Transmission of Information in Sliding-Window KV Inference | Mechanism | DEFER | Retaining previously computed states beats recomputing the final window; Muse Glimmer and Mistral 7B show strongest latent information relay, recovering information after source tokens leave the cache | 4/4 |
| 2609.34117 | SlimWise: Decoupling Expert Pruning Across Prefill and Decode for Efficient MoE Serving | MoE | ADOPT | Prefill full pool, decode pruned pool, directly reuse prefill KV **without conversion**; training-free handoff; **1.81x decode throughput at 50% expert pruning** on Qwen3.6-35B-A3B; optional low-cost distillation for residual loss; benchmark accuracy conceals generation-length distortion | 3/3 |
| 2609.36654 | Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference | FP4 tooling | DEFER | Every 16 E2M1 weights share an E4M3 block scale; GPTQ's forward update means independent block scoring misestimates final error, so reproduce the updates each scale causes; 99.35% / 100.84% question-weighted recovery from BF16 across seven benchmarks; 15.17x / 23.14x per-layer time over ModelOpt / LLM Compressor at 35.0 GB peak per GPU | 9/9 |
| 2609.36722 | ATTUNER: Recomputation-Free KV Cache Reuse via Query-Side Adaptation | KV reuse | DEFER | Positional mismatch has minor effect; the failure localizes to attention, not KV; low-rank query adapters trained by distillation, under 0.05% of parameters, up to 3.73x speedup matching full-prefill quality | 2/2 |
| 2609.36760 | QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching | MLA | ADOPT-HYBRID | RoPE-path error amplifies; path-specific transforms fusible into weights offline so no online overhead; joint INT4 content+RoPE; content to INT2 with RoPE INT4; kernel fuses unpack/dequant into attention; **3.59x cache compression at 128K**, **5.168x whole-job output throughput vs BF16** | 5/5 |
| 2609.37261 | Efficiently Approximating Attention Is Hard | Theory | REJECT | No truly subquadratic algorithm approximates attention with any nontrivial additive or relative guarantee uniformly over all inputs, even after polynomial KV preprocessing | n/a (no numbers cited) |
| 2609.37693 | FP64 Is All You Want, INT8 Is All You Need, FP4/6/8 Is All You Have | Numeric | REJECT | Ozaki schemes posed as a combinatorial program minimizing low-precision GEMM count; first FP6 schemes; FP4 scheme with the fewest GEMMs in range; up to 83x over native FP64 on B300 | 1/5 (4 `TABLE-SOURCED`) |
| 2609.38121 | WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms | KV | ADOPT-HYBRID | Separate key and value transforms from second-order statistics of both factors; value transform folded into weights, key transform applied after RoPE; near-optimal with QuEST INT under mild assumptions; lowest end-to-end perplexity among tested; at 2-bit comparable to or better than OSCAR | 4/4 |
| 2609.38169 | STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization | Recurrent state | ADOPT (evidence) | Error depends on memory lifetime and on key-row impact; matches FP32-state accuracy only at a nominal 6-bit budget and beats uniform INT8 at 4 bits; over 5x state compression, 68.7% serving-memory reduction | 5/5 |

## 7. Errors found and corrected during this synthesis

### 7.1 NVFP4 byte accounting (substantive)

An agent report stated NVFP4 as 17 bytes per 16 elements, 1.0625
bytes/element. Correct value is 36 bytes per 64 elements, 0.5625
bytes/element. Overstatement of 47%. All capacity and bandwidth reasoning in
that report is void. Source of truth is section 3, taken from
`ggml-common.h`. `QK_NVFP4_SUB` is 16 but that is the *sub-block* granularity for
scales, not the storage block size; conflating the two produced the error.

### 7.2 Cross-paper number misattribution (substantive)

Two numbers attributed to **2609.14060 (AGENTQ)** — `0.05` and `12,000` — do
not appear anywhere in that paper. `0.05` is the *fraction of parameters
trained* in **2609.36722 (ATTUNER)** ("fewer than 0.05% of the model
parameters"), and `12,000` is a benchmark size in **2609.36722**'s evaluation
family. The agent conflated two papers. AGENTQ's own verifiable quantitative
claim is the "up to 100% post-quantization attack success rate" across three
trigger-action pairs and three codebooks. Security conclusions are unaffected,
but the provenance error is recorded because it shows the failure mode to guard
against: **a plausible number is not evidence of a citation.**

### 7.3 Table-sourced numbers (not fatal)

**2609.37693**'s per-GPU throughput figures (2000/1000/1792/1490) live in a
hardware-specification table and do not survive HTML-to-text extraction. The
abstract's headline claim, "up to 83x on B300," is `VERIFIED`. Recorded as
`TABLE-SOURCED` rather than quoted as fact.

### 7.4 Terminology (minor)

**2609.33135**: an agent cited "WGMMA." The abstract says "Tensor Core MMA" and
`WGMMA` appears nowhere. Immaterial to the conclusion, recorded for accuracy.

### 7.5 Incomplete abstract (blocking for one paper)

**2605.12327 (Grid Games)**: the fetched arXiv metadata truncates the abstract
mid-sentence ("we are free to select the"). Premise captured, result not.
Flagged rather than guessed.

## 8. Verification method

Two independent passes, because the first pass was not good enough on its own.

1. **Body-text pass.** For each paper, the extraction pipeline pulled the
   section-selected body text (up to 9,000 characters, excluding the arXiv nav
   boilerplate and the `arXiv:`-anchored References block) and searched for each
   number cited in the agent reports. Result: 30/45 papers fully matched.
   Failures were investigated rather than assumed benign.
2. **Abstract pass.** The title-metadata capture also retained arXiv
   `citation_title` and `citation_abstract` for every paper. The abstracts are
   author-written and un-mangled, making them the authoritative statement of each
   claim. Re-running the check against abstracts resolved 13 of the 15
   remaining papers, yielding **43/45 fully verified** and the three exceptions
   recorded in section 7.

The lesson worth keeping: body-text extraction mangles exactly the numbers that
matter, because arXiv wraps them in LaTeX spans (`[math]`) that carry the
digits. Any future numeric audit of scraped papers should check abstracts first.

Artifacts (scratch, outside the repository):

- `papers/raw/*.txt` — 45 extracted papers
- `papers/titles.json` — raw arXiv metadata including abstracts
- `abstracts_clean.txt`, `clean_titles.md` — normalized
- `verify_out.txt`, `numbers_out.txt`, `abstracts_out.txt`, `probe_out.txt` —
  verification logs
- `deploy/lit{1,2,3}/agent_*.md` — seven Poolside extraction reports

## 9. Open questions

1. **ROCmFPX type IDs.** Still uncharacterized. We have the ggml layout from
   source but not the mapping from GGUF metadata type IDs to MXFP4 and NVFP4 on
   our ROCm build. This is the single blocking unknown for the whole FP4 path
   and must be resolved by probing the GGUF reader, not by inference.
2. **The two FP4 scale decoders.** Both are non-obvious and both fail silently,
   so they are transcribed from `ggml-impl.h` and pinned by a differential test
   against ggml's own `dequantize_row_mxfp4`/`dequantize_row_nvfp4` rather than
   trusted. `ggml_e8m0_to_fp32_half(x)` is `2^(x-128)` — *half* of E8M0, because
   `kvalues = 2 * E2M1_float`; dropping the "half" is a uniform 2x error that
   still produces plausible-looking output. `ggml_ue4m3_to_fp32(x)` also halves,
   and maps **both `0x00` and `0x7F` to `0.0f`**: `0x7F` is the standard UE4M3
   NaN encoding and ggml deliberately zeroes it, so a NaN scale silently
   discards 16 weights instead of poisoning them. The sentinels are `0x7F`
   (UE4M3) and `0xFF` (E8M0 → `2^127`); note `0xFF` through a UE4M3 decoder is a
   perfectly valid `240.0`, not an error, so a "scale 255" check written against
   the wrong format would pass while testing nothing.
3. **Fused-GEMM global-scale mismatch.** Minima found this as a real bug in
   per-module-calibrated NVFP4 served through fused GEMMs. We fuse. A test must
   assert that scales from different modules are never combined in one
   accumulator.
4. **OAS/MBS cost on our path.** 6.2% average GEMM overhead is a
   hardware-relative number. Our unpack is a software LUT, so the real cost is
   arithmetic we add to a bandwidth-bound kernel and may be near zero. Needs
   measurement, not extrapolation.
5. **Hadamard rotation vs the fixed E2M1 grid.** Rotation can be fused into
   weights offline at zero runtime cost, but it changes the distribution the
   fixed grid must cover, and MXINT8-outperforms-FP findings were measured on
   rotated data. Interaction with our unpack path is unmeasured.
6. **Expert prefetch predictor.** 2603.19289 reports 14% TPOT on CPU-to-GPU
   offload. Our path is disk-to-VRAM with a hot LRU already in front of it, so
   the marginal gain is smaller and unmeasured. The concept transfers; the
   number does not.
7. **Pruned decode expert pool.** SlimWise's 1.81x is on a full-pool HBM-resident
   deployment. With our LRU, pruning is partly redundant with what we already
   do. Interaction unknown.
8. **E2M1 LUT placement.** Constant/global memory versus shared memory versus
   immediate values is an unresolved micro-decision with real bandwidth impact
   on a bandwidth-bound kernel.

## 10. Implementation order implied by this corpus

1. Resolve the ROCmFPX type IDs before writing any kernel, and pin the two FP4
   scale decoders with a differential test against ggml's own
   `dequantize_row_mxfp4`/`dequantize_row_nvfp4` (§3) — including the split-half
   nibble order, the halved scales, and `0x7F -> 0`.
2. Write a reference MXFP4 and NVFP4 dequantizer matching section 3 exactly,
   with random **non-constant** scale tests and nibble-permutation tests.
3. Build the `__dp4a` path: keep FP4 packed, expand via exact int8 LUT,
   quantize activations to int8, accumulate in int32, apply block scales after
   accumulation. Never expand FP4 to int8 offline.
4. Land 2609.04098's fused-GEMM global-scale test before enabling any fused
   FP4 path.
5. Add expert prefetch speculation (2603.19289) on top of the existing hot-expert
   LRU, with the predictor's hit rate as a reported metric rather than an
   assumption.
6. Adopt 2603.08713 OAS/MBS in the MXFP4 scale path and measure actual cost.
7. For GLM/MLA: block size 1, no KV offload, dual-path cache handling if
   quantized, strict-prefix cache alignment and per-rank kernel pinning as a
   correctness gate.
8. Run the Fisher-weighted sensitivity pass (2603.08747, 2504.14152) to drive a
   mixed-precision policy, defaulting to higher precision on MLP
   down-projections and early blocks.
9. Only then consider KV/recurrent-state work, with the state-format decision
   kept at F32 unless a paper of DAMP/STEPQuant's quality shows otherwise for
   our 12 GB budget.
