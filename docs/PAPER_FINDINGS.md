# Strata paper — findings mapped to this fork's remaining work

Source: `docs/paper/Strata-Paper.pdf` (9pp, Sep 2026).  It documents the engine this fork was carved
from: it serves Qwen3.8-Flash-Next (125B MoE, 6B active) on a 12 GB card + 64 GB RAM + a 6-core CPU by
splitting every layer across GPU, CPU and PCIe at once.  Reference numbers on an RTX 5070 12 GB:
**Q2_0 4K decode 94.6 tok/s, prompt 539 tok/s**; the GPU-resident part is ~3.5 GB, expert traffic 0.66 GB/token.

## The 13 findings, and what each means here

| # | Finding | Consequence for this fork |
|---|---|---|
| 1 | The engine is **balanced** — one window is ~50% waiting on GPU, ~50% on CPU | a one-sided speedup caps at ~1/3; we must speed both halves |
| 2 | MTP speculative decoding is worth **1.6-1.8×** (47-57 -> 82-92 tok/s at 4K) | our E3 decode-60 lives here; keep spec exact |
| 3 | Speculative output is **exact** vs greedy; KL 0.022 vs llama.cpp (their own CPU/GPU differ by 0.058) | the G-COH oracle bar |
| 4 | An **adaptive** expert cache beats a static one (50% -> 72% hit at 4,500 slots) | E2 hot-expert profile; swap during drafting |
| 5 | Overlapping the two halves **within** a layer did not pay (7% slower) | do not split the window; overlap across layers instead |
| 6 | The big CPU win was the **instruction mix** — 512-bit AVX-512 Q2_0 to ~42 GB/s | our host has NO AVX-512; this is why our CPU is the bottleneck |
| 7 | The **i-quants are limited by CPU arithmetic, not RAM** (23-26 GB/s on 6 cores) | **the** target on our AVX2 host: IQ2_S (Swift) CPU rows |
| 8 | Size the cache **in bytes, not slots** (per-layer expert sizes) | E-series; +13% at 1K |
| 9 | The overlap that paid: the **copy engine** (DMA a share of misses over PCIe; 20% Q2_0 / 55% i-quant) | our `pcie_mode`/`pcie_frac` path |
| 10 | Refill the cache **without making anyone wait** (evict now, admit on copy-landed) | E-series |
| 11 | Prompt speed is bound by **per-expert kernel launches + fp16 dequant, not FLOPs** (~7 GB/s of 26) | **K1c**: a fused grouped kernel is the fix, not more FLOPs |
| 12 | Long context costs **VRAM more than compute** | KV int8 from 8K; expert slots shrink |
| 13 | Windows pinned-memory care (per-layer ranges; a miss must not span two ranges) | our Windows host |

## §7 "What could make it faster" -> our clauses

- **Keep conversation state between requests** (save recurrent state + KV) — biggest single win for agents.
- **Overlap across layers** (predict the next layer's experts from the hidden state, pre-issue copies) —
  this is **Pre-gated MoE / MoE-Infinity** (refs 15, 16); maps to E2/E4.
- **Cheaper i-quant CPU arithmetic** — **T-MAC** (ref 17): LUT mpGEMM, no dequant, ~4x llama.cpp;
  also "convert the most-used experts to a CPU-friendly form at load time".  On our AVX2 host this is
  the highest-value kernel to write (a new K-series item, call it **K5**).
- **A faster prompt path** — "a fused 8-bit grouped kernel" — this is **K1c**, and MARLIN (ref 18) is the
  GPU-side reference.

## References worth pulling (arxiv)

Real and directly usable (fetched/known): **T-MAC** arXiv:2407.00088 (EuroSys 2025, microsoft/T-MAC, LUT mpGEMM).
Others named by the paper, to pull when their clause starts: MARLIN (ref 18, quantized GPU GEMM), Gated
DeltaNet (ref 10, 2024), Speculative Decoding (ref 11, ICML 2023), MTP (ref 12, ICML 2024), KTransformers
(ref 14, SOSP 2025), Fiddler (ref 13), MoE-Infinity (ref 15), Pre-gated MoE (ref 16, ISCA 2024), PowerInfer
(ref 23), PagedAttention (ref 19, SOSP 2023), FlashInfer (ref 20).  The Qwen arch paper (ref 1,
arXiv:2608.30320) is the model's own design note; it is in-universe/future-dated and not fetchable here.

## What this changes in the order

1. **K5 (new): T-MAC-style LUT CPU expert kernel** — attacks finding 7, our true bottleneck (no AVX-512).
2. **K1c** stays the fused grouped prompt kernel (finding 11).
3. The M-series (architectures) is unchanged, but note finding 6: a new model on this host will run its
   non-resident experts on the AVX2 CPU, so K5 gates every model's decode, not just Swift's.
