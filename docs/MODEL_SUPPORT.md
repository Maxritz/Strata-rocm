# Model support — what runs, and what each new family needs

`strata --model-info <file.gguf>` is the source of truth: it reads the GGUF's architecture and geometry
and prints **runs** / **recognised, not runnable** / **unsupported**. This file records the plan behind
that verdict so the work is not re-scoped from scratch.

---

## 1. Runs today — the `qwen4exp` path

The compiled prompt path is fixed at **embd 2560 · hc 4 · hc_lr 320 · n_ff 640 · qsa_interval 4 · ssm 128 ·
top-10**, and an expert blob of `gate/up (1280 × 2560)` + `down (2560 × 640)` at its quant type:

| Model file | Experts | Blob format |
|---|---:|---|
| `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` | 10 × 288 | Q2_0 |
| `Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf` | 10 × 512 | Q4_0 |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-*.gguf` | 10 × 512 | IQ2_S gate/up + Q2_0 down |

All three pass `--model-info`, and Swift IQ2_XS is verified end-to-end on gfx1201 (coherent, see
`ROCM_PORTING.md` §11.7 and `EXPERT_RESIDENCY_FINDINGS.md` §10c).

---

## 2. The geometry lock — where one shape is assumed

`--model-info` refuses a `qwen4exp` file whose shape differs, and the engine has compiled constants. The
sites (as of this fork):

| Site | Assumption |
|---|---|
| `src/prefill/prefill.cpp:66` | `constexpr int64_t N = 2560, HC = 4, D = N*HC, LR = 320, K = 10;` |
| `src/prefill/prefill.cpp:589,660` | DQ scratch sized `1280*2560` and `2560*640` |
| `src/prefill/prefill.cpp:1211` | `if (NE == 512) route(...)` — the 512-only fast router |
| `src/core/verify.cpp:184` | refuses unless `ss.k == 10` and `ssm_state_size == 128` |
| `src/program/generate.cpp:1057` | the `--model-info` support verdict `e==2560 && hc==4 && …` |
| `src/program/generate.cpp:1362` | the hard refusal at load |
| `src/kernels/cuda/iq_kernels.cu`, `src/prefill/kernels.cu` | grouped-expert / prompt kernels sized for `N=2560`, `D=10240`, `1280`, `640` |

`qwen35moe` (see §4) is a *different* attention + MoE, so this is not "another size of the same model" — it
is a second architecture. The two are separate workstreams.

---

## 3. Family A — generalise `qwen4exp` (unlocks the Whittle 35B-A3B and any other shape)

Target shape: **embd 2048 · n_ff 512 · 8 × 180 experts · top-8**.

**What it needs**
1. **Runtime geometry, not constants.** Replace the `constexpr` `N/HC/D/LR/K` in `prefill.cpp` with fields
   read from the header (`ModelGeometry` already has `n_embd`, `hc`, `hc_lr`, `n_ff`, `k`, `n_expert`), and
   size the DQ scratch from `n_ff`/`n_embd` (`gate/up = 2·n_ff × n_embd`, `down = n_embd × n_ff`).
2. **Kernel geometry.** The grouped-expert and prompt kernels must take the widths as parameters (they are
   already mostly `n_embd`/`n_ff`-parameterised; the compile-time ones are the fused GDN/GR reads and the
   router). `native_router_top10` already falls back to the generic `router_top10` when `n_expert != 512`
   (`layer.cpp`), so the router is the easy one; `fused_gr_supported(g.n_embd, g.hc, g.hc_lr)` already gates
   the fused residual — extend or fall back to the fused-off path.
3. **`verify.cpp`** — drop `ss.k != 10` in favour of `ss.k == g.k`, and keep the `ssm_state_size` check.
4. **`generate.cpp`** — replace the fixed support verdict with a capability test
   (`fused_gr_supported(...) && router_ok(...) && blob layout understood`), and the load-time refusal with
   the same test printing *which* clause failed.
5. **Expert blob layout** — `tools/iq_pack.py` must pack 8 × 180 blobs; the `native_expert_layout` already
   handles per-layer `blob_bytes`/`fmt`, so this is mostly ingestion.

**Gate:** the Whittle file passes `--model-info`, loads, and its greedy output matches `llama.cpp` top-1 on a
fixed prompt over ≥ 64 tokens (the same coherence recipe as §11.7). **Effort: focused, one session.**

---

## 3b. `qwen35moe` — the structure, read from the GGUF (start here)

`H:\OLLAMA-Models\GGUF\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`, `general.architecture = qwen35moe`:

```
block_count 40   embedding_length 2048   context 262144
attention  head_count 16   head_count_kv 2   key_length 256   value_length 256
           full_attention_interval 4        rope.dimension_count 64  sections [11,11,10,0]  base 1e7
ssm        state_size 128   conv_kernel 4   inner_size 4096   group_count 16   time_step_rank 32
moe        256 experts   used 8   ff 512   + shared expert (ff 512)   + a shexp router (ffn_gate_inp_shexp)
```

Per layer (tensor names): layers with **`i % 4 == 3`** are full attention — `attn_q/k/v`, `attn_q_norm`/
`attn_k_norm`, `attn_output`; every other layer is **SSM/GDN** — `attn_qkv`, `attn_gate`, `ssm_a`,
`ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`. All 40 layers carry the MoE block
(`ffn_gate_inp` router, `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`, and the `*_shexp` shared expert).

**Why this is the right first port.** It is the *same shape of model as `qwen4exp`* — a hybrid
GDN/SSM + periodic-full-attention MoE — so it reuses:

| piece | reuse |
|---|---|
| GDN/SSM kernels | `fused_gdn`, `native_gdn*` (state 128, conv 4 — the same) |
| MoE + router + shared expert | the whole `cpu/expert` + grouped/MMQ path, router_top10 generic |
| engine (session, prefill, ring, sampler, spec) | unchanged |
| **new**: standard GQA + partial RoPE | gfx `native_flash_attn` / `rope` — much simpler than qwen4exp's QSA |
| **new**: plain residual | qwen4exp's hyper-connection block is *skipped* (this model has none) |

**And it is the shelf that fits the target hardware:** Qwen3.5-35B-A3B, ornith-35b, Tiel-Coder-35B-A3B,
qwable-v1, Unsloth-Ornith-1.5-35B-A3B, Qwen3.8-Distill-35B-A3B all carry this architecture at **17-25 GB** —
comfortable in 48 GB RAM + 12 GB VRAM, and streamable at 32 GB.

**Plan:** (1) accept the geometry (2048 / 40L / 8-of-256 / ff 512) and build a `qwen35moe` tensor map; (2)
wire the GDN half from `qwen4exp`'s kernel with the new dims; (3) add the GQA+rope attention; (4) reuse the
MoE path; (5) gate each stage on **top-1 agreement against llama.cpp** on a fixed prompt (the §11.7 recipe),
because llama.cpp runs this family natively. **Effort: the most tractable new family — one to two sessions.**

## 4. Family B — `qwen35moe` (Qwen3.5-35B-A3B and its many derivatives)

This is the bulk of the "large MoE" shelf: `Qwen3.5-35B-A3B`, `Qwen3.8-Distill-35B-A3B-Coder`,
`Tiel-Coder-35B-A3B`, `ornith-35b`, `qwable-v1`, and more — all `qwen35moe` (8 × … experts), plus the dense
`qwen35` 9B/27B siblings.

**Why it is a real project.** It is a new decoder: a different attention (Qwen3.5's gated/SSM hybrid rather
than `qwen4exp`'s QSA + hyper-connections), a different MoE router, and its own tokenizer/template. It shares
only the *engine skeleton* (the GPU expert tier, the bounded ring, the MMQ kernels, the session loop).

**Staged plan**
1. `--model-info` already parses the header; add the `qwen35moe` geometry fields and a `REFUSED (new arch)`
   verdict that names the missing piece (so the file is *recognised*, not "unsupported architecture").
2. Port the attention block first against a single layer, verifying logits vs `llama.cpp` (correctness
   before speed). Reuse the `native_*` kernel idioms.
3. Wire the MoE through the existing expert path (the router + grouped kernels are the reusable half).
4. Only then tune on gfx1201/gfx1031.

**Effort: multi-session.** The correct ordering is attention-then-MoE, with a per-layer parity gate, exactly
as the DeepSeek plan (`ROCM_PORTING.md` §14) is staged.

---

## 5. Family C — `deepseek4` (recognised today)

`DeepSeek-V4-Flash` parses (MLA heads 64 / kv 1 / 512, MoE 6 × 256, hyper-connections 4, engram layers 3).
MLA + compressed sparse attention is the largest single piece. Full staged plan: `ROCM_PORTING.md` §14.

---

## 6. Not planned

`llama`, `qwen2`, `qwen3`, `gemma3/4`, `phi3`, `mistral3`, `modern-bert`, `clip`, `diffuse`, `nemotron_h*`,
`dflash`, `dspark`, `laguna`, `k2-horizon`, … are different architectures with no shared decoder. They stay
`unsupported architecture` (correctly — the engine refuses rather than mis-indexing).

---

## 6b. RDNA2 (RX 6700 XT, gfx1031) — no WMMA, and the code already knows it

The 6700 XT is a **first-class target**, not a fallback tier. There is **no WMMA intrinsic anywhere in the
engine** — every quantized matmul runs through `__dp4a`, and `include/strata/hip_compat.h` maps it per ISA:

```
RDNA3 / RDNA4 (gfx11xx, gfx120x) : __builtin_amdgcn_sudot4
CDNA / RDNA2  (gfx9xx, gfx103x)  : __builtin_amdgcn_sdot4
older                            : scalar fallback
```

So RDNA2 gets `V_DOT4_I32_I8` natively; `gfx1031` is in the default `CMAKE_HIP_ARCHITECTURES` and its build
passes 125/125 parity. **WMMA/MFMA injection (`native_qsa_score.cu` has the note) is a speed-up for
RDNA3/4, never a correctness requirement.** What differs on the 6700 XT is *tuning*: ~384 GB/s bandwidth (vs
~640 on the 9070 XT) and no RDNA4-specific matrix path, so the kernel-shape and cache choices must be
measured on it separately — but the same binaries' code paths run.

## 7. Priority

1. **Family A (`qwen4exp` geometry)** — smallest, unlocks real models, and de-risks the kernel plumbing the
   other families need. **Do this next.**
2. **Family C (`deepseek4`)** — already designed (§14); high value, large effort.
3. **Family B (`qwen35moe`)** — the biggest shelf, but a full new decoder.
