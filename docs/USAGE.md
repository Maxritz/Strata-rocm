# Strata (ROCm / AMD) — usage and examples

This is the **ROCm port** of Strata for AMD GPUs (RDNA2/RDNA4). It runs the same
**Qwen3.8-Flash-Next** / **Swift 1.5** models on an AMD card with the HIP runtime.

> **Which GPU?** Verified on **gfx1201 (RX 9070 XT, RDNA4, 16 GB)** and building for **gfx1031
> (RX 6700 XT, RDNA2, 12 GB)**. ReBAR should be on (it is, on X570 boards). The bounded-RAM mode
> (`--expert-ram-gb`) is what makes the 12 GB / 48 GB-RAM tier run the model at all — see
> [Bounded RAM](#bounded-ram-20-24-gb-of-host-ram) and `ROCM_PORTING.md` §15.

The engine is a single executable, `strata`, and it takes **token IDs** (it has no tokenizer of
its own). Two helpers do the text ↔ IDs conversion: `tools/strata_tokenizer.py` (reads the pack's
tokenizer) and `tools/gguf_reader.py`.

---

## 1. Build

```sh
cmake -DSTRATA_ENABLE_HIP=ON -DROCM_INSTALL_DIR=<rocm> -DCMAKE_HIP_ARCHITECTURES=gfx1201 -B build_gfx1201
cmake --build build_gfx1201 --target strata
```

Run from the build directory (or put `<rocm>/bin` on `PATH`). On Windows the same commands work from
a Developer PowerShell; replace `gfx1201` with `gfx1030` (or your card's `rocminfo` ISA).

---

## 2. Which models run

`strata --model-info <file.gguf>` reads any GGUF's architecture and geometry and says whether this
build runs it. **Fully supported today** (the compiled `qwen4exp` prompt path — 2560 embd / 4
hyper-connections / n_ff 640 / qsa_interval 4 / ssm 128 / top-10):

| Model file | Experts | Notes |
|---|---|---|
| `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` | 10 × 288 | the `reap-288` pack (Q2_0 default) |
| `Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf` | 10 × 512 | full 512-expert original |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-*.gguf` | 10 × 512 | the Swift 1.5 fine-tune (IQ2_XS) |

**Recognised but not yet runnable:** `DeepSeek-V4-Flash-…` (`deepseek4` — MLA + compressed sparse
attention; plan in `ROCM_PORTING.md` §14). **Recognised, geometry not compiled:** the Whittle 35B-A3B
(`qwen4exp` but 2048 embd / 8 × 180). Everything else reports `unsupported architecture`.

```sh
strata --model-info H:\OLLAMA-Models\GGUF\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
```

---

## 3. First run — the smallest useful command

A model needs two things: a **pack directory** (the dense weights + `experts.bin` + tokenizer) and
the **PLE n-gram shard** (the model's second GGUF shard). Build the pack once with
`tools/iq_pack.py`; a prepared Swift pack is used throughout the examples below.

```powershell
# 1) turn a prompt into token IDs
python tools\strata_tokenizer.py --gguf <shard1.gguf> --text "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
#    -> prints the comma-separated IDs

# 2) run the engine on those IDs
build_gfx1201\strata.exe `
  --pack H:\OLLAMA-Models\strata-pack-swift `
  --native  H:\OLLAMA-Models\GGUF\Swift-...-00001-of-00002.gguf `
  --ple-gguf H:\OLLAMA-Models\GGUF\Swift-...-00001-of-00002.gguf `
  --tokens "248045,846,198,7734,...,198" --max-new 64 --max-context 4096
```

The engine prints `output : <ids>` and `T <id>` lines; decode them back to text with the tokenizer's
`decode`. (`--tokens-file PATH` takes the IDs from a file instead — use it for long prompts.)

---

## 4. Common flags

| Flag | What it does |
|---|---|
| `--tokens "1,2,3"` / `--tokens-file PATH` | the prompt (required) |
| `--max-new N` | tokens to generate (default 16) |
| `--max-context N` | the KV context window |
| `--prefill N` \| `--prefill auto` | prompt chunk size; `auto` picks the largest that fits |
| `--expert-cache N\|auto` | expert blobs resident in VRAM (the measured sweet spot is **2600–6000**; 9000 thrashes 16 GB) |
| `--expert-profile PATH` | a routing-frequency profile; ranks which experts to keep resident |
| `--expert-ram-gb N` | **bounded host RAM** (20 / 22 / 24): the LRU expert ring instead of pinning the whole set |
| `--spec T` | speculative decoding depth (T ≥ 2). `--mtp PATH` adds the draft head |
| `--kv-resident N` | KV streaming: keep the cache in RAM, only N positions in VRAM |
| `--model-info PATH` | report a GGUF's architecture and whether this build runs it |
| `--no-ple` | ablation: run without the PLE layer |

---

## 5. Bounded RAM (20–24 GB of host RAM)

By default the engine pins **all 33 GiB of experts** in host RAM (≈34 GB working set). On a 48 GB
machine — or to leave room for the OS — ask for a bounded pool instead:

```powershell
... --expert-ram-gb 22 ...
```

This uses `RingExpertSource`: a fixed ring of pinned slots, LRU-evicted, one read per miss, with a
per-layer event that guarantees a slot is never reused while a copy from it is still in flight.
Measured on Swift IQ2_XS / RX 9070 XT: **23.2 GB** peak (vs 34.2 GB) with byte-identical output.
**Sweep 20 / 22 / 24**: pick the smallest that still gives the speed you want — a bigger pool means
fewer re-reads.

---

## 6. Long context

```powershell
... --tokens-file long.ids --max-context 40960 --prefill 2048 --kv-resident 32768
```

`--kv-resident` keeps the far part of the KV cache in RAM and only what the attention is reading in
VRAM, which frees VRAM for experts (upstream measured +23% at very long context). For a 32K-token
prompt, expect roughly **1.5–3 minutes** to first token on the RX 9070 XT (see the README table).

---

## 7. Faster answers — speculative decoding

```powershell
... --spec 4 --mtp <draft-head.gguf> --max-new 256
```

Speculation drafts several tokens and verifies them in one pass; the answer is exactly the same,
only faster when drafts are accepted. The MTP head is the model's own small draft layer. `--spec 4`
accepts more than `--spec 2` when the text is predictable (code), less when it is not.

---

## 8. Server / OpenAI-compatible API

```powershell
build_gfx1201\strata.exe --serve --port 8080 --pack <pack> --native <shard1> --ple-gguf <shard1> ...
```

Point any "OpenAI-compatible" client at `http://127.0.0.1:8080/v1` (any key, any model name).
See `README.md` and `docs/DETAILS.md` for the web app and `chat.py`.

---

## 9. Checking an answer is coherent

The engine only emits IDs, so the decisive test is: decode the IDs, and compare the **greedy next
token per position** against a reference (e.g. `llama-cli --temp 0` on the same GGUF). All-pass =
coherent. On this port the Swift IQ2_XS model answers the "add two numbers" prompt with a correct
`<think>…</think>` block and `def add(a, b): return a + b` (see `ROCM_PORTING.md` §11.7 and
`EXPERT_RESIDENCY_FINDINGS.md` §10c). `tools/` also has `dump/compare` helpers.

---

## 10. When something is wrong

- **`hipErrorNoBinaryForGpu` / `no kernel image`** — the build's `CMAKE_HIP_ARCHITECTURES` does not
  match your card; rebuild with the ISA `rocminfo` reports.
- **The engine says the pack geometry differs** — it is refusing a mismatched pack rather than
  mis-indexing it; rebuild the pack with that model's tooling.
- **Very slow, disk light blinking** — the ring is thrashing (pool smaller than the working set).
  Raise `--expert-ram-gb`, or read the `expert ring … thrash` line in the summary.
- **Out of VRAM** — lower `--expert-cache`, add `--kv-resident`, or lower `--prefill`.

Fuller notes: `docs/ROCM_PORTING.md` (the port log) and `docs/EXPERT_RESIDENCY_FINDINGS.md` (the
expert-residency design).
