"""tools/iq_pack.py - plan v0.3 P6: a native pack for any of the model files (Q2_0, IQ2_XS, IQ3_XXS).

    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The i-quant experts cannot be re-expressed in the Q2_0 pack form, so this pack keeps every quantized tensor in
its GGUF form:

  experts.bin          per layer, 512 blobs of [gate rows | up rows | down rows], the raw GGUF slices.  Blob
                       size is per layer (the files mix IQ1_M ... IQ3_S gate/up and Q2_0 / IQ4_NL down).
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: the BF16/F16/F32 tensors exactly as the GGUF stores them (index kinds 4/5/2).
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.

Split files: every shard of the model is read (<name>-0000N-of-0000M.gguf beside --gguf), so the layers may be
split anyhow (Swift 1.5's GGUFs put layers 13-47 in shard 2 and the PLE table in shard 1).  A layer whose experts
are not in shard 1 names its shard in native_experts.txt (v3).  Router tensors stored as F32 whose values are
exactly BF16 (Swift 1.5) are written as BF16, the form the engine's router takes; anything else is refused.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
NOT_IN_PACK = {"per_layer_token_embd.weight"}      # the 28.8 GB PLE table: read from its GGUF by the engine
# Tensors the engine reads as resident float even in a quantized (Q4_K_M) model: dequantize to BF16 in dense.bin.
FORCE_BF16 = {"blk.1.ple_value.weight"} | {
    "blk.%d.%s.weight" % (l, s) for l in range(48)
    for s in ("hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject",
              "ssm_alpha", "ssm_beta")
} | {"output_hc_down.weight", "output_hc_up.weight"}


def _bf16(f32: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7fff) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return u.astype(np.uint16)


def _f16pair(b: np.ndarray) -> np.ndarray:
    return b.copy().view(np.float16).astype(np.float32)


def dequant_q4_k(raw: bytes) -> np.ndarray:
    """ggml Q4_K -> f32 (matches the GPU dq_q4_k)."""
    b = np.frombuffer(raw, dtype=np.uint8)
    nb = len(b) // 144
    blk = b.reshape(nb, 144)
    d = _f16pair(blk[:, 0:2]).reshape(nb)
    dmin = _f16pair(blk[:, 2:4]).reshape(nb)
    sc12 = blk[:, 4:16]
    qs = blk[:, 16:144]
    out = np.empty((nb, 256), dtype=np.float32)
    for gi in range(8):
        if gi < 4:
            sc = (sc12[:, gi] & 63).astype(np.int32)
            m = (sc12[:, gi + 4] & 63).astype(np.int32)
        else:
            sc = ((sc12[:, gi + 4] & 0xF) | ((sc12[:, gi - 4] >> 6) << 4)).astype(np.int32)
            m = ((sc12[:, gi + 4] >> 4) | ((sc12[:, gi] >> 6) << 4)).astype(np.int32)
        q = qs[:, 32 * (gi // 2):32 * (gi // 2) + 32]
        codes = (q >> 4 if (gi & 1) else q & 0xF).astype(np.float32)
        out[:, gi * 32:gi * 32 + 32] = d[:, None] * sc[:, None] * codes - dmin[:, None] * m[:, None]
    return out.reshape(-1)


def dequant_q5_0(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    nb = len(b) // 22
    blk = b.reshape(nb, 22)
    d = _f16pair(blk[:, 0:2]).reshape(nb)
    qh = (blk[:, 2].astype(np.uint32) | (blk[:, 3].astype(np.uint32) << 8) |
          (blk[:, 4].astype(np.uint32) << 16) | (blk[:, 5].astype(np.uint32) << 24))
    qs = blk[:, 6:22]
    out = np.empty((nb, 32), dtype=np.float32)
    for j in range(16):
        xh0 = ((qh >> np.uint32(j)) << np.uint32(4)) & np.uint32(0x10)
        xh1 = (qh >> np.uint32(j + 12)) & np.uint32(0x10)
        lo = (((qs[:, j] & 0x0F).astype(np.uint32) | xh0).astype(np.int32) - 16)
        hi = (((qs[:, j] >> 4).astype(np.uint32) | xh1).astype(np.int32) - 16)
        out[:, j] = lo * d
        out[:, j + 16] = hi * d
    return out.reshape(-1)


def dequant_q6_k(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    nb = len(b) // 210
    blk = b.reshape(nb, 210)
    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208].copy().view(np.int8)
    d = _f16pair(blk[:, 208:210]).reshape(nb)
    out = np.empty((nb, 256), dtype=np.float32)
    for g in range(8):
        n, qu = g // 4, g % 4
        qlb = ql[:, 64 * n:64 * n + 64]
        qhb = qh[:, 32 * n:32 * n + 32]
        scb = sc[:, 8 * n:8 * n + 8]
        for l in range(32):
            if qu == 0:      q = (qlb[:, l] & 0x0F) | (((qhb[:, l] >> 0) & 3) << 4)
            elif qu == 1:    q = (qlb[:, l + 32] & 0x0F) | (((qhb[:, l] >> 2) & 3) << 4)
            elif qu == 2:    q = (qlb[:, l] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)
            else:            q = (qlb[:, l + 32] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)
            out[:, g * 32 + l] = d * scb[:, (l >> 4) + 2 * qu].astype(np.float32) * (q.astype(np.int32) - 32)
    return out.reshape(-1)


DEQUANT = {"Q4_K": dequant_q4_k, "Q5_0": dequant_q5_0, "Q6_K": dequant_q6_k}


class Model:
    """All shards of one model: name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path):
        import re
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        paths = [first]
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
            paths = [p for p in paths if p.exists()]
        self.paths = paths
        self.where = {}
        for p in paths:
            g = G.GGUFFile(p)
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)
ROLES = ("gate", "up", "down")
N_EXPERT = 512   # overridden from the model's `qwen4exp.expert_count` in main()
ALIGN = 64


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def tensor_bytes(mm, g, t) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def index_standalone(src, out, model: Model) -> int:
    """Every non-expert tensor of the model: floats into dense.bin as stored (exact-BF16 F32 routers as BF16),
    quantized ones native-only."""
    rows, at = [], 0
    served = 0
    with open(out / "dense.bin", "wb") as fo:
        for name, (g, t, mm, _) in model.where.items():
            if is_expert(t.name) or t.name in NOT_IN_PACK:
                continue
            if len(t.shape) > 2:
                print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
                return 1
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            if t.name in FORCE_BF16 and t.type_name not in FLOAT:
                fn = DEQUANT.get(t.type_name)
                if fn is None:
                    print("force-bf16 tensor %s is %s; no dequant" % (t.name, t.type_name))
                    return 1
                raw = _bf16(fn(tensor_bytes(mm, g, t).tobytes())).tobytes()
                rows.append([t.name, "0", "4", str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(raw)
                pad = (-len(raw)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(raw) + pad
                continue
            if t.type_name in FLOAT:
                raw = tensor_bytes(mm, g, t).tobytes()
                kind = {"BF16": "4", "F16": "5", "F32": "2"}[t.type_name]
                if t.type_name == "F32" and t.name.endswith(ROUTERS):
                    u = np.frombuffer(raw, dtype=np.uint32)
                    if np.count_nonzero(u & 0xFFFF):
                        print("router %s is F32 with values that are not BF16; the engine's router is BF16" % t.name)
                        return 1
                    raw = (u >> 16).astype(np.uint16).tobytes()     # the exact BF16 values
                    kind = "4"
                rows.append([t.name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(raw)
                pad = (-len(raw)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(raw) + pad
            else:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
    write_index(out, rows, src, served, 0)
    return 0


def write_index(out, rows, src, served, n_extra):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(a, src, base, out, g, T, mm) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bg = G.GGUFFile(base_src)
    BT = {t.name: t for t in bg.tensors}
    bmm = np.memmap(base_src, dtype=np.uint8, mode="r")
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = T.get(name), BT.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(tensor_bytes(mm, g, t), tensor_bytes(bmm, bg, bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(tensor_bytes(mm, g, t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    src = pathlib.Path(a.gguf).resolve()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    global N_EXPERT
    N_EXPERT = int(g.metadata.get("qwen4exp.expert_count", N_EXPERT))
    print("experts: %d (qwen4exp.expert_count)" % N_EXPERT)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    model = Model(src)
    T = {n: w[1] for n, w in model.where.items()}
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    if a.base:
        if any(w[3] != src for w in model.where.values() if not w[1].name in NOT_IN_PACK):
            print("--base needs a model whose tensors are all in shard 1")
            return 1
        rc = index_from_base(a, src, base, out, g, {t.name: t for t in g.tensors}, mm)
    else:
        rc = index_standalone(src, out, model)
    if rc:
        return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts
    n_layers = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk.") and n.endswith("_exps.weight"))
    layout, offset = [], 0
    for l in range(n_layers):
        ts = [T["blk.%d.ffn_%s_exps.weight" % (l, r)] for r in ROLES]
        per = [t.expected_bytes() // N_EXPERT for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            print("layer %d: gate and up differ in type" % l)
            return 1
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * N_EXPERT
    # v4: record n_embd / n_ff in the header, so the engine sizes the expert blobs from the model and not from its
    # compiled 2560/640 defaults (the Whittle 35B-A3B is 2048/512).  The gate_exps shape is [n_embd, n_ff, n_expert].
    n_embd0 = int(layout[0][5][0].shape[0]) if layout else 0
    n_ff0 = int(layout[0][5][0].shape[1]) if layout else 0
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                 "(n_expert %d, n_embd %d, n_ff %d, total %d; absolute offsets in %s, or in the named shard beside it)\n"
                 % (N_EXPERT, n_embd0, n_ff0, offset, src.name))
        for l, gt, dt, off, blob, ts in layout:
            ws = [model.where[t.name] for t in ts]
            if len({w[3] for w in ws}) != 1:
                print("layer %d: its gate/up/down tensors are in different shards" % l)
                return 1
            gg, shard = ws[0][0], ws[0][3]
            line = "%d %d %d %d %d %d %d %d" % (l, gt, dt, off, blob, *[gg.data_start + t.offset for t in ts])
            fo.write(line + ("" if shard == src else " " + shard.name) + "\n")
    if a.skip_experts or not a.experts_bin:
        if (out / "experts.bin").exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    path = out / "experts.bin"
    if path.exists() and path.stat().st_size == offset:
        print("experts.bin exists with the right size; not rewritten")
        return 0
    with open(path, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t.name).reshape(N_EXPERT, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)          # (512, blob): gate | up | down per expert
            assert chunk.shape == (N_EXPERT, blob)
            fo.write(chunk.tobytes())
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    print("experts.bin: %d layers, %.2f GiB" % (n_layers, offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
