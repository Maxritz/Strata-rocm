# Expert residency on gfx1201 — findings and TODO

Status as of commit `c55f8bf` (branch `main`), session on the `Strata-rocm` tree.
Companion to `docs/ROCM_PORTING.md` §15 (the design in the main doc; this file is the full handoff).

---

## 0. Context

- **Engine:** Strata, a sparse-MoE LLM engine, ported to ROCm/HIP for RDNA4.
- **Hardware:** RX 9070 XT (gfx1201, RDNA4), **16 GB VRAM**, **16 GB ReBAR** window, PCIe 4.0 x16
  (measured **~30 GB/s** aggregate — saturated), 64 GB DDR5 host RAM, Ryzen 9 5900XT (Zen3, **no AVX-512**,
  AVX-2 only), X570. Windows + ROCm 10.1 (`G:\ROCM10RT-gfx1201`).
- **Model used for all numbers here:** Swift (Qwen3.8-Flash-Next-GS) IQ2_XS, canonical pack
  `H:\OLLAMA-Models\strata-pack-swift` (`experts.bin` = 35,454,976,000 B = ~33 GiB), shard 1
  `H:\OLLAMA-Models\GGUF\Swift-…-IQ2_XS-00001-of-00002.gguf`.
- **Geometry:** 48 layers, 512 experts/layer, top-10 routing; **max blob 1,510,400 B** (~1.5 MB);
  24,576 bflobs total. The experts **do not fit** in 16 GB VRAM.
- **Targets:** prefill **800 tok/s** (MET — 943-1192 tok/s), decode **40+ tok/s** (open — ~22 tok/s).

---

## 1. The problem

`ArenaExpertSource` pins the **entire** expert set at startup:
`hipHostRegister(PORTABLE)` of `lay.total` = **35,456,548,864 B** → measured peak WorkingSet **34.2 GiB**.
The alternative `--mmap-experts` maps the whole file → **30.1 GiB** (file-backed pages count in the process
WorkingSet) and is ~1.6x slower. Neither can be bounded to 20-24 GB: both are *unbounded residency with no
admission control*.

---

## 2. The deploy-3 debate (three Poolside agents, run via `/deploy-3`)

Task given: diagnose the current handling and propose the best bounded-RAM architecture, with code.
All three returned **OK** and **converged on the same root cause and the same fix**.

### 2.1 Consensus diagnosis
1. **Unbounded residency, no admission control** — cold experts are resident exactly like hot ones.
2. **Transient-pointer contract violation** — every call site treats `blob(l,e)` / `device_alias(l,e)` as
   **persistent**. Evidence: the whole-model `seq` vector in `prefill.cpp` (`~897`), the per-layer alias
   (`~1295`, `~1400`). A bounded reader makes a returned pointer valid only until its slot is reused.
3. **No hot/cold split** — nothing uses the routing profile to decide what stays resident.

### 2.2 Consensus fix
- **Bounded pinned slot ring** — `slots = floor(ram / slot_bytes)`, one blob per slot, unbuffered/`pread` read
  on a miss, **LRU eviction**.
- **Hot-set admission** seeded from `--expert-profile` (routing frequency).
- **Transient contract** — `acquire(l,e) -> {slot, ptr, version}` and `release(...)`; the caller copies the
  blob into its VRAM/group slot **before** release.
- **Optional** transfer ring for overlap — but our measurement (below) says naive overlap **regressed**
  (524 vs 485 ms), so it stays opt-in, not the fix.
- **VRAM cache** at the measured knee (`--expert-cache 2600..6000`; **9000 over-subscribes and thrashes** →
  230 tok/s).

### 2.3 Physical limits (all three agents agree — design against these)
- 33 GiB over a 30 GB/s PCIe-4.0 link is **≥ 1.1 s** cold; a full miss sweep cannot be free.
- All experts resident in 16 GB VRAM is impossible (would need ~768 GB total if all 512/layer).
- VRAM cache beyond ~6000 slots **thrashes**.
- Overlap must be stream-ordered; a second stream that is not event-gated regresses.

---

## 3. Measured evidence (this session)

**Command shape** (run from the repo root; exe `build_gfx1201\strata.exe`):

```
build_gfx1201\strata.exe --pack H:\OLLAMA-Models\strata-pack-swift \
  --native H:\OLLAMA-Models\GGUF\Swift-…-00001-of-00002.gguf \
  --ple-gguf  H:\OLLAMA-Models\GGUF\Swift-…-00001-of-00002.gguf \
  --expert-cache 2600 --expert-profile data\expert-profile.bin \
  [--expert-ram-gb 20|22|34] --spec 2 --prefill 128 --max-new 4 --max-context 1024 \
  --tokens "248045,846,198,7734,264,2716,12654,709,421,11039,1330,4947,13,248046,198,248045,74455,198"
```

### 3.1 RAM vs output (the decisive table)

| arm | peak WorkingSet | `output` |
|---|---|---|
| arena (default) | **34.18 GB** | `248068 198 760 1156` |
| `--expert-ram-gb 22` | **23.16 GB** | `248068 271 248069 271` |
| `--expert-ram-gb 20` | **21.16 GB** | `248068 271 248069 271` |
| `--expert-ram-gb 34` (no eviction) | 35.16 GB | `248068 198 760 1156` |

Readings:
- **The RAM target is met**: 22 → 23.16 GB, 20 → 21.16 GB (ring + ~1.2 GB weights/VRAM).
- **First token matches in every arm.**
- **The no-eviction ring reproduces the arena byte-for-byte.** Therefore the ring's read and indexing are
  **correct**, and the evicting arms' divergence is **purely slot reuse while a consumer still references the
  slot** — the transient contract, not a data/layout bug.
- Both 20 and 22 diverge to the *same* token sequence → a deterministic race, matching an eviction-driven
  overwrite of an in-flight copy.

### 3.2 Startup line (ring selected)

```
strata generate: expert ring: 15639 slots x 1510400 B = 22.00 GiB (--expert-ram-gb 22)
strata generate: NOTE the ring's transient-pointer release is not yet wired (docs §15.5 step 2);
                 correctness with prefill streaming is not guaranteed
```

### 3.3 Prior, still-valid numbers
- Prefill: 39-tok chunk 1153→485 ms (2.4x); large chunk 2047 tok **943 tok/s**; 4096-tok resident path
  **1191.8 tok/s** (vs 579 streaming). Prefill target **MET**.
- Decode: **~22 tok/s** (`--spec 2`). Open.
- `rocBLAS` one-time BF16/FP16 init ~630 ms — now paid at load via `Gemm::warmup` (commit `a6949a0`).

---

## 4. What is implemented (commit `c55f8bf`)

| file | change |
|---|---|
| `include/strata/core/expert_source.hpp` | **`RingExpertSource`** class: slot ring, LRU, `acquire`/`begin_layer`, `--expert-ram-gb` sizing, `slot_host`/`slot_device`, counters (`slots/hits/misses/evictions/thrash`). |
| `src/core/expert_source.cpp` | `RingExpertSource::{open,close,read_blob,acquire,blob,pinned,device_alias,begin_layer}`. `pread` miss via `fread`+64-bit seek; refuses native packs without `experts.bin` (would silent-corrupt). |
| `src/program/generate.cpp` | `--expert-ram-gb N` option + parse + help; ring selected when set; **mutually exclusive with `--mmap-experts`**; loud startup NOTE. |
| `docs/ROCM_PORTING.md` | §15: diagnosis, contract, eviction truth table, refactor plan, physical limits, measured table. |

**Protection policy today:** `begin_layer`/`acquire` un-hold the previous layer's slots when the layer
advances. This protects a layer's working set but **not** the async copy / alias beyond the layer boundary —
which is exactly why 20/22 diverge. The slot's `used` event exists in `prefill.cpp` (`m.used[sl]`,
`hipStreamWaitEvent(m.copy, m.used[sl])` at `~931`/`~1303`) and the copy records `m.copied[sl]` at `~940`; the
release must be tied to those.

---

## 5. TO DO

### P0 — correctness: wire the transient release (docs §15.5 step 2)
**Goal / gate:** `--expert-ram-gb 20` and `22` must reproduce the arena output **`248068 198 760 1156`**
(then the full coherence recipe, `ROCM_PORTING.md` §11.7), and RAM stays ≤ N GB + ~1 GB.

1. **Give the source a release handle.** `RingExpertSource::acquire` returns a slot id (add a public
   `release(int64_t slot)`; keep the returned pointer valid until then). Version-stamp so a stale handle is
   detectable.
2. **`prefill.cpp` per-layer path (`~1240-1410`).** Replace `m.src->blob(l,e)` / `device_alias(l,e)` with
   `acquire`; **release the slot only when its `m.copied[sl]` event completes** (the copy is async). For the
   **direct path** (`pf_direct()` + `device_alias`, `~1295`/`~1400`) the GPU reads the alias during the layer's
   compute — release after `post[l]` is enqueued on `m.cs` (record an event at that point and release on it).
3. **The whole-model stream (`stream_all`, `~895-946`).** It builds a `seq` of **all 24,576** blobs and issues
   copies later — a persistent-pointer assumption that a ring cannot satisfy. Either (a) acquire at issue time
   from `(l,e)` and release on `m.copied[sl]`, or (b) **force `stream_all=false` when the source is a ring**
   (add a virtual `bool transient() const` to `ExpertSource`, ring → true; `prefill.cpp ~895`).
4. **Decode / session path.** Audit `generate.cpp` call sites: `~2621`, `~3047`, `~3454`, `~3798` (host
   fallback compute), `~1740`/`~1753` (profile staging), `~2405` (pinned accounting — read-only, safe). Every
   consumer must release on its completion event.
5. **Re-run the gate** (§3.1) and the coherence recipe; then tune **20 / 22 / 24** for the knee.

### P1 — bounded RAM, tuned
6. Sweep `--expert-ram-gb {20,22,24}` for the smallest pool that reproduces the arena output at the target
   prefill/decode (the pool must hold at least one layer's working set, ≤ 512 × 1.5 MB ≈ 0.77 GB, plus the
   VRAM-resident set and the in-flight copies).
7. Consider **unbuffered reads** (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) for the miss path so the OS standby
   list does not grow; today `fread` is buffered (standby is reclaimable, so WorkingSet is fine, but it can
   pressure the page cache).
8. Report ring stats (`hits/misses/evictions/thrash`) in the end-of-run summary next to `expert blobs read`
   (`generate.cpp ~4103`).

### P2 — throughput
9. **Decode 40+**: MTP head (`--mtp`) + a `--spec` sweep; the P2.3 conclusion (docs §12.2) is that decode is
   spec-bound, not PCIe-bound.
10. **Prefill on the ring**: re-check the large-chunk path with the ring (step 3a/3b) — the 943-1192 tok/s was
    measured on the arena; the ring must match.
11. **Overlap**: only if event-gated. The earlier two-stream attempt regressed (524 vs 485 ms); the ring makes
    overlap *possible* (release on copy-done) but do not enable it without a measured win.

### P3 — the rest of the backlog (see `ROCM_PORTING.md` §12)
12. **DeepSeek-V4 support** (ARCH S2-S6; docs §14) — pack + `official.vec` parity gate; the MLA tower is the
    bulk. Reference `antirez/ds4` + `Anemll/ds4-ssd`.
13. **`qwen35moe` / Q2_0 packs** — needs `antirez/ds4` `qwen4_exp_convert.py` / `qwen4_iq2.py` run externally.
14. **PLE ≤ 20 GB** — blocked: no BF16 n-gram source locally (re-quantizing the IQ4_NL shard would
    double-quantize). Engine already keeps PLE ~90 MB resident + the 26.42 GiB on-disk shard.
15. LIT items: expert bundling, KV `q8_0`, autosestarch harness, Q2_0 pack (docs §12).

---

## 6. Reference map

- **The ring:** `include/strata/core/expert_source.hpp` (`RingExpertSource`), `src/core/expert_source.cpp`
  (`open`/`acquire`/`read_blob`/`begin_layer`/`evictable_slot`).
- **The call sites to wire:** `src/prefill/prefill.cpp` `~895` (stream_all), `~907`, `~1240-1410`
  (per-layer), `~1295`/`~1400` (direct alias); `src/program/generate.cpp` `~1740`, `~1753`, `~2405`, `~2621`,
  `~3047`, `~3454`, `~3798`.
- **The arena being replaced:** `ArenaExpertSource` (`expert_source.cpp ~652-767`) — pins `lay.total`.
- **The PLE `direct` read pattern** the ring copies: `src/kernels/ngram.cpp ~214` (`--ple-io direct`).
- **The design + this session's numbers:** `docs/ROCM_PORTING.md` §15 (§15.4 contract, §15.5 plan, §15.6
  limits, §15.7 measured).
- **Flag:** `--expert-ram-gb {20|22|24}` (`generate.cpp` help + parse + selection), mutually exclusive with
  `--mmap-experts`.

---

## 7. Edge0 (`C:\Users\rr\OneDrive\Desktop\Edge0`) — what to borrow (deploy-3, second run)

Edge0 is a Python/MLX engine (Apple Silicon), but its **SSD-streaming** design (`src/edge0/streaming/`,
docs `streaming.md` / `prerouter.md`) is the closest working reference for our problem. Three agents evaluated
it **independently and converged**.

### 7.1 Edge0's components (as read from source)
- `SafetensorsMmap` (`streaming/mmap.py`) — byte-range mmap, zero-copy `raw(name)`; `warm_pages` faults pages
  without retaining arrays; `seq_read` forces pages resident.
- `SharedExpertCache` (`streaming/cache.py`) — **one global LRU across all layers** (a flat budget; a thrashing
  layer evicts others), plus `PrefetchBuffer` — a separate cap-bounded buffer for prefetched-but-unused
  bundles, with `prefetch_wasted` counted and promotion into the LRU on use.
- `_get_bundles` (`streaming/layer.py:388`) — the tiered resolution order: **pinned → shared LRU → prefetch
  buffer → in-flight futures (wait) → thread-pool build**, each tier counted.
- `prefetch`/`_on_prefetch_done` (`layer.py:458`) — an async build pool + `_inflight` futures; the done-callback
  fills the `PrefetchBuffer`.
- `_hot_counts[e] = count*decay + 1` (`decay 0.75`), `hot_per_layer` / `hot_window` / `pin_bonus` — hot-expert
  pinning from a decayed frequency.
- `load_full_layer` (`layer.py:883`) — whole-layer prefill: the checkpoint stores each layer as ONE stacked
  tensor, so a whole-layer load is **9 direct reads**, not `E x 9` per-expert builds; overlapped with the
  previous layer's GPU work; `prefill_full_layers` limits it to the leading N layers.
- `prerouter` — a small per-layer MLP predicting the **next** layer's routing one token ahead (double shift:
  prev-layer + prev-token); `pin_bonus` keeps predicted experts resident.

### 7.2 Consensus ranking (3/3 agents)

**Decode (22 → 40+ tok/s):**
1. **Transient slot release tied to the copy event** — correctness, and the enabler for every async tier below.
2. **Shared cross-layer LRU + prefetch buffer + async prefetch** — the biggest throughput tier (agents scored
   it ~+8 tok/s / ~2x combined).
3. **Hot-expert pinning with a decayed counter** — ~+5 tok/s / ~1.3x.
4. **PrefetchBuffer + tiered resolution order** — structure that makes 2/3 safe and measurable.
5. **prerouter** — marginal (~5-10% / +3 tok/s / 1.2x); **build only the frequency-based version, and skip it
   if measured accuracy < 70%.**

**Prefill (target already met):** whole-layer load (fewer, bigger reads) > async prefetch > shared LRU.

### 7.3 Non-transferable
`SafetensorsMmap`'s JSON-header parsing and macOS `madvise`/`seq_read` are Apple/format specific. We have
`experts.bin` = one contiguous byte range (`lay.blob_offset(l,e)`), so byte-range reads are already the trivial
case; `seq_read` (force everything resident) is the *opposite* of the bounded goal.

### 7.4 The concrete mechanism to adopt (from agent 1, verbatim shape)
Per-slot completion event; a slot is reusable only after its copy event has fired:
```cpp
struct RingSlot { void* host_ptr; hipEvent_t copy_done_event; std::atomic<bool> in_use{false}; uint64_t last_used; };
// acquire: claim an !in_use slot; if it was ever copied from, hipEventSynchronize(copy_done_event) first.
// copy:    hipMemcpyAsync(...); hipEventRecord(slot.copy_done_event, copy_stream);
// release: in_use = false after the consumer's event (never before).
```
This is exactly the P0 fix, and it is a property of the **slot**, not of the caller — so it is safe even when
the host runs ahead of the GPU.

### 7.5 Why the earlier two-stream overlap REGRESSED (524 vs 485 ms) — agent 3's reading
The reverted attempt copied on a second stream but **the compute did not wait on the right event**, so it either
serialised (no overlap) or raced (re-copy). Correct overlap is: copy on `m.copy` → `hipEventRecord(copied[sl])`
→ compute does `hipStreamWaitEvent(m.cs, copied[sl])` — which `prefill.cpp` already does for the stage ring
(`~931`/`~1303`). The source ring must adopt the **same** event discipline, not a second ad-hoc stream.

## 8. Hardware tiers — the design must scale DOWN (new requirement)

Two target machines, both with ReBAR (so `device_alias` works on both):

| tier | GPU | VRAM | host RAM | CPU / chipset | ISA |
|---|---|---|---|---|---|
| **A (dev)** | RX 9070 XT | 16 GB | 64 GB | Ryzen 9 5900XT / X570 | gfx1201 (RDNA4) |
| **B (min)** | RX 6700 XT | **12 GB** | **48 GB** | Ryzen 5 5600X / X570 | **gfx1031 (RDNA2)** |

Implications:
- **Tier B cannot hold the expert set at all** (33 GiB vs 12 GB VRAM / 48 GB RAM), so the bounded ring is not
  an optimisation there — it is the *only* way the model runs. Decode on tier B will be PCIe/disk-bound, so
  **prefetch + hot pinning (7.2 items 2-3) matter more, not less** (higher miss rate).
- **Defaults must be derived, not hard-coded.** `--expert-ram-gb` should default from *available* RAM
  (tier B: ≤ ~24 GB so the OS + weights + KV fit; tier A: ≤ ~40 GB), and the VRAM expert cache must stay
  `auto` (tier B: a much smaller cache — the measured 9000-slot thrash is ~9 GB and would over-subscribe 12 GB).
- **gfx1031 is a separate kernel path** (RDNA2 has no gfx12 WMMA). The gfx1031 build already exists
  (`build_gfx1031`, 125/125 parity) — the ring, being host-side, is ISA-independent and needs no new kernel.
- **Disk speed is now on the token path.** On tier B the miss stream is large; the P1 unbuffered-read item
  and the P2 prefetch items become correctness-of-experience issues, not micro-optimisations.

## 9. Updated priority order

- **P0 (correctness):** event-tied slot release (§4 step 2, §7.4). Gate = ring 20/22 reproduces `248068 198 760 1156`.
- **P1 (scale-down):** derived `--expert-ram-gb` default from available RAM; `auto` VRAM cache; verify the
  gfx1031 build runs the ring. Gate = tier-B-like footprint (12 GB VRAM / 48 GB RAM) starts and decodes.
- **P2 (throughput, borrowed from Edge0):** shared-LRU hit accounting → `PrefetchBuffer` → async build pool →
  hot-expert pinning with `count*0.75+1` → (optional, only if accuracy ≥70%) frequency-based prerouter.
- **P3:** the existing backlog — DeepSeek S2, qwen35moe/Q2_0, PLE, LIT items (docs §12, §14).

## 10. One-line summary

The 34 GB arena is unbounded residency; the fix — a bounded pinned LRU ring (`RingExpertSource`,
`--expert-ram-gb`) — **meets the RAM target (22→23.16 GB, 20→21.16 GB) and reads correctly** (a no-eviction
ring reproduces the arena exactly). It needs **one correctness step** (event-tied release) plus **three
borrowables from Edge0** (shared-LRU/prefetch-buffer tiering, an async build pool, and decayed hot-expert
pinning) — and the same ring must be the path that lets the **12 GB VRAM / 48 GB RAM gfx1031** tier run the
model at all.
