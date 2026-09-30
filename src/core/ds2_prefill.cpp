// src/core/ds2_prefill.cpp - the batched deepseek2 prefill.  See include/strata/core/ds2_prefill.hpp.
//
// This is a SEPARATE runner from `ds2_token`: same weights, same MLA math, but the chunk's T rows pass through
// each layer at once.  `ds2_token` remains the decode path and is byte-for-byte unchanged.  Every kernel that
// needed a chunk-wide generalisation lives in `src/kernels/cuda/ds2_prefill.cu`; everything else is the
// token-path's own kernel called with T rows.
#include "strata/core/ds2_prefill.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ds2_moe.hpp"
#include "strata/kernels/ds2_prefill.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/mla.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {

uint64_t align256(uint64_t n) { return (n + 255) & ~(uint64_t) 255; }

struct Cursor {
    uint8_t* p;
    uint64_t used = 0;
    void* take(uint64_t bytes) {
        void* r = p + used;
        used = align256(used + bytes);
        return r;
    }
};

bool ck(hipError_t e, const char* what, std::string& err) {
    if (e != hipSuccess) {
        err = std::string("ds2_prefill: ") + what + ": " + hipGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace

uint64_t ds2_prefill_scratch_bytes(const ModelGeometry& g, int64_t max_T, int64_t k, int64_t n_ff_dense) {
    if (max_T <= 0 || k <= 0 || n_ff_dense <= 0) return 0;
    const int64_t H = g.n_embd, LQ = g.n_lora_q, KV = g.n_lora_kv, ROT = g.n_rot, NH = g.n_head, HD = g.head_dim;
    const int64_t NOPE = HD - ROT;
    const int64_t widest = std::max(std::max(H, LQ), std::max(NH * HD, n_ff_dense));
    uint64_t n = 0;
    auto add = [&](uint64_t bytes) { n = align256(n + bytes); };
    add((uint64_t) max_T * H * 4 * 2);              // X, XN
    add((uint64_t) max_T * LQ * 4 * 2);             // QA, QAN
    add((uint64_t) max_T * NH * HD * 4);            // Q
    add((uint64_t) NH * max_T * NOPE * 4);          // QNOPE
    add((uint64_t) max_T * NH * ROT * 4);           // QPE
    add((uint64_t) max_T * (KV + ROT) * 4);         // KVA
    add((uint64_t) max_T * KV * 4 * 2);             // KLAT, KPE (KPE uses ROT <= KV)
    add((uint64_t) max_T * NH * KV * 4 * 2);        // QABS, ALAT
    add((uint64_t) max_T * NH * HD * 4);            // ATTN
    add((uint64_t) max_T * H * 4 * 2);              // OUT, SH
    add((uint64_t) max_T * g.n_expert * 4);         // LOGITS
    add((uint64_t) max_T * k * 4 * 2);              // WEIGHTS, IDS
    add((uint64_t) max_T * k * H * 4);              // PARTS
    add((uint64_t) max_T * n_ff_dense * 4 * 2);     // DGATE, DUP
    add((uint64_t) max_T * H * 2);                  // XBF16
    add((uint64_t) strata::prefill::mmq::q8_bytes(max_T, widest));
    add((uint64_t) max_T * (std::max(H, NOPE)) / 32 * 36);   // EX_XQ (block_q8_1 rows), plus the head q8 coverage
    add((uint64_t) strata::kernels::native_expert_scratch_bytes(max_T * k, g.n_ff));
    add((uint64_t) (g.n_expert + 1) * 4 * 3);       // GRP_START, ENT_DST, ENT_TOK (ENT_* are max_T*k though)
    add((uint64_t) max_T * k * 4 * 2);              // ENT_DST, ENT_TOK
    add((uint64_t) (g.n_expert) * 8);               // GRP_PTR
    add(16);                                        // BOUNDS (2 x i32)
    add((uint64_t) max_T * 4);                      // IDS_ID
    return n + 4096;
}

bool ds2_prefill_scratch_init(const ModelGeometry& g, int64_t max_T, int64_t k, int64_t n_ff_dense, void* base,
                              Ds2PrefillScratch& s) {
    if (max_T <= 0 || k <= 0 || n_ff_dense <= 0 || base == nullptr) return false;
    const int64_t H = g.n_embd, LQ = g.n_lora_q, KV = g.n_lora_kv, ROT = g.n_rot, NH = g.n_head, HD = g.head_dim;
    const int64_t NOPE = HD - ROT;
    const int64_t widest = std::max(std::max(H, LQ), std::max(NH * HD, n_ff_dense));
    Cursor c{(uint8_t*) base};
    s.max_T = max_T;
    s.k = k;
    s.n_layers = g.n_layers;
    s.n_ff_dense = n_ff_dense;
    s.X = (float*) c.take((uint64_t) max_T * H * 4);
    s.XN = (float*) c.take((uint64_t) max_T * H * 4);
    s.QA = (float*) c.take((uint64_t) max_T * LQ * 4);
    s.QAN = (float*) c.take((uint64_t) max_T * LQ * 4);
    s.Q = (float*) c.take((uint64_t) max_T * NH * HD * 4);
    s.QNOPE = (float*) c.take((uint64_t) NH * max_T * NOPE * 4);
    s.QPE = (float*) c.take((uint64_t) max_T * NH * ROT * 4);
    s.KVA = (float*) c.take((uint64_t) max_T * (KV + ROT) * 4);
    s.KLAT = (float*) c.take((uint64_t) max_T * KV * 4);
    s.KPE = (float*) c.take((uint64_t) max_T * ROT * 4);
    s.QABS = (float*) c.take((uint64_t) max_T * NH * KV * 4);
    s.ALAT = (float*) c.take((uint64_t) max_T * NH * KV * 4);
    s.ATTN = (float*) c.take((uint64_t) max_T * NH * HD * 4);
    s.OUT = (float*) c.take((uint64_t) max_T * H * 4);
    s.SH = (float*) c.take((uint64_t) max_T * H * 4);
    s.LOGITS = (float*) c.take((uint64_t) max_T * g.n_expert * 4);
    s.WEIGHTS = (float*) c.take((uint64_t) max_T * k * 4);
    s.IDS = (int32_t*) c.take((uint64_t) max_T * k * 4);
    s.PARTS = (float*) c.take((uint64_t) max_T * k * H * 4);
    s.DGATE = (float*) c.take((uint64_t) max_T * n_ff_dense * 4);
    s.DUP = (float*) c.take((uint64_t) max_T * n_ff_dense * 4);
    s.XBF16 = c.take((uint64_t) max_T * H * 2);
    s.XQ = c.take(strata::prefill::mmq::q8_bytes(max_T, widest));
    s.EX_XQ = c.take((uint64_t) max_T * (H / 32) * 36 + 4096);
    s.EX_SCRATCH = c.take(strata::kernels::native_expert_scratch_bytes(max_T * k, g.n_ff));
    s.GRP_START = (int32_t*) c.take((uint64_t) (g.n_expert + 1) * 4);
    s.GRP_N = (int32_t*) c.take(4);
    s.ENT_DST = (int32_t*) c.take((uint64_t) max_T * k * 4);
    s.ENT_TOK = (int32_t*) c.take((uint64_t) max_T * k * 4);
    s.GRP_PTR = c.take((uint64_t) std::max<int64_t>(g.n_expert, 1) * 8);
    s.BOUNDS = (int32_t*) c.take(8);
    s.IDS_ID = (int32_t*) c.take((uint64_t) max_T * 4);
    if (hipHostAlloc((void**) &s.HOST_IDS, (size_t) max_T * k * 4, hipHostMallocMapped | hipHostMallocPortable) !=
        hipSuccess) {
        s.HOST_IDS = nullptr;
        return false;
    }
    // identity row map for the plain GEMMs
    std::vector<int32_t> id((size_t) max_T);
    for (int64_t i = 0; i < max_T; ++i) id[(size_t) i] = (int32_t) i;
    hipMemcpy(s.IDS_ID, id.data(), (size_t) max_T * 4, hipMemcpyHostToDevice);
    const int32_t b[2] = {0, (int32_t) max_T};
    hipMemcpy(s.BOUNDS, b, sizeof b, hipMemcpyHostToDevice);
    s.MMQ_CTX = new strata::prefill::mmq::Context();
    return true;
}

bool ds2_prefill(const WeightTable& tables, const ModelGeometry& g, int64_t pos0, int64_t T, SessionState& sess,
                 Ds2PrefillScratch& sc, ExpertSource& src, void* stream, std::string& err) {
    using namespace strata::kernels;
    namespace mmq = strata::prefill::mmq;
    if (!mmq::built()) {
        err = "ds2_prefill: this build has no MMQ (the batched quantized GEMM)";
        return false;
    }
    if (g.mla == 0 || T <= 0 || T > sc.max_T) {
        err = "ds2_prefill: the geometry or chunk length is not usable";
        return false;
    }
    if (pos0 < 0 || pos0 + T > sess.max_cells) {
        err = "ds2_prefill: the chunk is outside the sequence";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) {
        err = "ds2_prefill: the batched prefill needs a native pack (GGUF expert blocks)";
        return false;
    }
    hipStream_t cs = (hipStream_t) stream;
    const int64_t H = g.n_embd, LQ = g.n_lora_q, KV = g.n_lora_kv, ROT = g.n_rot, NH = g.n_head;
    const int64_t HD = g.head_dim, NOPE = HD - ROT;
    const int64_t NE = g.n_expert, K = sc.k;
    const float scale = 1.0f / std::sqrt((float) HD);
    auto* ctx = (mmq::Context*) sc.MMQ_CTX;
    const int32_t bounds_ok[2] = {0, (int32_t) T};
    if (!ck(hipMemcpyAsync(sc.BOUNDS, bounds_ok, sizeof bounds_ok, hipMemcpyHostToDevice, cs), "bounds", err))
        return false;

    auto gemm = [&](const void* w, int type, int64_t n_in, int64_t n_out, const float* x, int64_t ld, float* y,
                    int64_t ld_dst) {
        mmq::quantize(x, nullptr, sc.XQ, type, n_in, ld, T, cs);
        mmq::Product p;
        p.w = w;
        p.type = type;
        p.w_rows = n_out;   // the weight has one row per output value
        p.w_cols = n_in;    // the contraction width
        p.expert_bytes = mmq::matrix_bytes(type, n_out, n_in);
        p.n = 1;
        p.xq = sc.XQ;
        p.bounds = sc.BOUNDS;
        p.ids = sc.IDS_ID;
        p.total_rows = T;
        p.max_rows = T;
        p.dst = y;
        p.ld_dst = ld_dst;
        ctx->run(p, cs);
    };

    static thread_local std::vector<int32_t> count, startpos, order;
    count.resize((size_t) NE);
    startpos.resize((size_t) NE);
    order.resize((size_t) (sc.max_T * K));

    for (int64_t l = 0; l < g.n_layers; ++l) {
        const LayerView v(tables, l);
        const WeightRef* wan = v.get("attn_norm.weight");
        const WeightRef* wfn = v.get("ffn_norm.weight");
        if (wan == nullptr || wfn == nullptr || wan->kind != WeightKind::F32 || wfn->kind != WeightKind::F32) {
            err = v.name("attn_norm/ffn_norm") + " is missing or not F32";
            return false;
        }
        MlaWeights w;
        if (!mla_layer_weights(tables, g, l, sess.ds2_norm_eps, sess.ds2_rope_base, w, err)) return false;

        // ---- attention half: X += MLA(rms(X)) ----
        ds2pf::rms_strided(sc.X, (const float*) wan->data, sc.XN, T, H, H, sess.ds2_norm_eps, cs);
        gemm(w.wq_a, w.wq_a_type, H, LQ, sc.XN, H, sc.QA, LQ);
        ds2pf::rms_strided(sc.QA, w.q_a_norm, sc.QAN, T, LQ, LQ, sess.ds2_norm_eps, cs);
        gemm(w.wq_b, w.wq_b_type, LQ, NH * HD, sc.QAN, LQ, sc.Q, NH * HD);
        gemm(w.wkv_a_mqa, w.wkv_a_type, H, KV + ROT, sc.XN, H, sc.KVA, KV + ROT);
        ds2pf::rms_strided(sc.KVA, w.kv_a_norm, sc.KLAT, T, KV, KV + ROT, sess.ds2_norm_eps, cs);
        ds2pf::rope_slice(sc.Q, sc.QPE, T * NH, HD, ROT, NOPE, ROT, w.rope_freq_base, pos0, NH, cs);
        ds2pf::rope_slice(sc.KVA, sc.KPE, T, KV + ROT, ROT, KV, ROT, w.rope_freq_base, pos0, 1, cs);
        MlaState& st = sess.mla_states[l];
        ds2pf::write_kv(st.kv, st.v, sc.KLAT, sc.KPE, pos0, T, KV, ROT, cs);
        ds2pf::gather_qnope(sc.Q, sc.QNOPE, T, NH, NOPE, HD, cs);
        const size_t wk_hb = native_mmvq_weight_bytes(w.wk_b_type, (int) NOPE, (int) KV);
        for (int64_t h = 0; h < NH; ++h) {
            mmq::quantize(sc.QNOPE + h * T * NOPE, nullptr, sc.XQ, w.wk_b_type, NOPE, NOPE, T, cs);
            mmq::Product p;
            p.w = (const uint8_t*) w.wk_b + h * wk_hb;
            p.type = w.wk_b_type;
            p.w_rows = KV;
            p.w_cols = NOPE;
            p.expert_bytes = mmq::matrix_bytes(w.wk_b_type, KV, NOPE);
            p.n = 1;
            p.xq = sc.XQ;
            p.bounds = sc.BOUNDS;
            p.ids = sc.IDS_ID;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = sc.QABS + h * KV;
            p.ld_dst = NH * KV;
            ctx->run(p, cs);
        }
        ds2pf::attention(sc.QABS, sc.QPE, st.kv, st.v, pos0, T, NH, KV, ROT, scale, sc.ALAT, cs);
        const size_t wv_hb = native_mmvq_weight_bytes(w.wv_b_type, (int) KV, (int) HD);
        for (int64_t h = 0; h < NH; ++h) {
            mmq::quantize(sc.ALAT + h * KV, nullptr, sc.XQ, w.wv_b_type, KV, NH * KV, T, cs);
            mmq::Product p;
            p.w = (const uint8_t*) w.wv_b + h * wv_hb;
            p.type = w.wv_b_type;
            p.w_rows = HD;
            p.w_cols = KV;
            p.expert_bytes = mmq::matrix_bytes(w.wv_b_type, HD, KV);
            p.n = 1;
            p.xq = sc.XQ;
            p.bounds = sc.BOUNDS;
            p.ids = sc.IDS_ID;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = sc.ATTN + h * HD;
            p.ld_dst = NH * HD;
            ctx->run(p, cs);
        }
        gemm(w.wo, w.wo_type, NH * HD, H, sc.ATTN, NH * HD, sc.OUT, H);
        add_inplace(sc.X, sc.OUT, T * H, cs);

        // ---- ffn half: X += FFN(rms(X)) ----
        ds2pf::rms_strided(sc.X, (const float*) wfn->data, sc.XN, T, H, H, sess.ds2_norm_eps, cs);
        if (l < sess.ds2_dense_lead) {
            const WeightRef* wg = v.get("ffn_gate.weight");
            const WeightRef* wu = v.get("ffn_up.weight");
            const WeightRef* wd = v.get("ffn_down.weight");
            if (!wg || !wu || !wd || !wg->native_data || !wu->native_data || !wd->native_data) {
                err = v.name("ffn_gate/up/down") + ": the dense FFN is not served natively";
                return false;
            }
            const int64_t FF = wg->ne1;
            if (FF > sc.n_ff_dense) { err = "ds2_prefill: the dense FFN is wider than the scratch"; return false; }
            gemm(wg->native_data, wg->native_type, H, FF, sc.XN, H, sc.DGATE, FF);
            gemm(wu->native_data, wu->native_type, H, FF, sc.XN, H, sc.DUP, FF);
            swiglu_mul(sc.DGATE, sc.DUP, sc.DGATE, T * FF, cs);
            gemm(wd->native_data, wd->native_type, FF, H, sc.DGATE, FF, sc.OUT, H);
        } else {
            const WeightRef* w_router = v.get("ffn_gate_inp.weight");
            const WeightRef* w_bias = v.get("exp_probs_b.bias");
            if (w_router == nullptr || w_bias == nullptr) {
                err = v.name("ffn_gate_inp.weight/exp_probs_b.bias") + " is missing";
                return false;
            }
            // router: one batched top-k over the chunk, with the token path's own BF16 rounding
            f32_to_bf16_bulk(sc.XN, (uint16_t*) sc.XBF16, T * H, cs);
            for (int64_t t = 0; t < T; ++t)
                bf16_gemv_split((const uint16_t*) sc.XBF16 + t * H, (const uint16_t*) w_router->data,
                                sc.LOGITS + t * NE, H, NE, 32, cs);
            ds2_router(sc.LOGITS, (const float*) w_bias->data, (int) T, (int) NE, (int) K, sess.ds2_expert_scale,
                       sess.ds2_expert_norm, sc.IDS, sc.WEIGHTS, cs);
            if (!ck(hipMemcpyAsync(sc.HOST_IDS, sc.IDS, (size_t) T * K * 4, hipMemcpyDeviceToHost, cs), "ids d2h",
                    err))
                return false;
            if (!ck(hipStreamSynchronize(cs), "ids sync", err)) return false;
            const int32_t* ids = (const int32_t*) sc.HOST_IDS;
            std::memset(count.data(), 0, (size_t) NE * sizeof(int32_t));
            for (int64_t i = 0; i < T * K; ++i) {
                const int32_t e = ids[i];
                if (e < 0 || e >= NE) { err = "ds2_prefill: a routed expert id is out of range"; return false; }
                ++count[(size_t) e];
            }
            static thread_local std::vector<unsigned long long> h_ptr;
            static thread_local std::vector<int32_t> h_start, h_dst, h_tok;
            h_ptr.resize((size_t) std::max<int64_t>(NE, 1));
            h_start.resize((size_t) NE + 1);
            h_dst.resize((size_t) (sc.max_T * K));
            h_tok.resize((size_t) (sc.max_T * K));
            int32_t acc = 0, ng = 0;
            for (int64_t e = 0; e < NE; ++e) {
                startpos[(size_t) e] = acc;
                if (count[(size_t) e] > 0) {
                    const uint8_t* d = src.device_alias(l, e);
                    if (d == nullptr) {
                        err = "ds2_prefill: the expert source has no device alias for the GPU grouping";
                        return false;
                    }
                    h_ptr[(size_t) ng] = (unsigned long long) d;
                    h_start[(size_t) ng] = acc;
                    ++ng;
                }
                acc += count[(size_t) e];
            }
            h_start[(size_t) ng] = acc;
            for (int64_t i = 0; i < T * K; ++i) order[(size_t) i] = 0;
            for (int64_t i = 0; i < T * K; ++i) {
                const int32_t e = ids[i];
                order[(size_t) startpos[(size_t) e]++] = (int32_t) i;
            }
            for (int64_t i = 0; i < T * K; ++i) {
                h_dst[(size_t) i] = order[(size_t) i];
                h_tok[(size_t) i] = order[(size_t) i] / (int32_t) K;
            }
            if (!ck(hipMemcpyAsync(sc.GRP_PTR, h_ptr.data(), (size_t) ng * 8, hipMemcpyHostToDevice, cs), "grp_ptr",
                    err))
                return false;
            if (!ck(hipMemcpyAsync(sc.GRP_START, h_start.data(), (size_t) (ng + 1) * 4, hipMemcpyHostToDevice, cs),
                    "grp_start", err))
                return false;
            if (!ck(hipMemcpyAsync(sc.GRP_N, &ng, 4, hipMemcpyHostToDevice, cs), "grp_n", err)) return false;
            if (!ck(hipMemcpyAsync(sc.ENT_DST, h_dst.data(), (size_t) T * K * 4, hipMemcpyHostToDevice, cs), "ent_dst",
                    err))
                return false;
            if (!ck(hipMemcpyAsync(sc.ENT_TOK, h_tok.data(), (size_t) T * K * 4, hipMemcpyHostToDevice, cs), "ent_tok",
                    err))
                return false;
            quantize_q8_1_rows(sc.XN, T, H, sc.EX_XQ, cs);
            const NativeExpertLayout L = native_expert_layout(lay.fmt[(size_t) l].gu_type, lay.fmt[(size_t) l].d_type,
                                                              H, g.n_ff);
            native_expert_grouped(L, (const unsigned long long*) sc.GRP_PTR, sc.GRP_START, sc.GRP_N, sc.ENT_DST,
                                  sc.ENT_TOK, ng, T * K, sc.EX_XQ, sc.EX_SCRATCH, sc.PARTS, cs);
            // the ungated shared expert, batched
            const WeightRef* wg = v.get("ffn_gate_shexp.weight");
            const WeightRef* wu = v.get("ffn_up_shexp.weight");
            const WeightRef* wd = v.get("ffn_down_shexp.weight");
            if (!wg || !wu || !wd || !wg->native_data || !wu->native_data || !wd->native_data) {
                err = v.name("ffn_*_shexp") + ": the shared expert is not served natively";
                return false;
            }
            const int64_t FF = wg->ne1;
            if (FF > sc.n_ff_dense) { err = "ds2_prefill: the shared expert is wider than the scratch"; return false; }
            gemm(wg->native_data, wg->native_type, H, FF, sc.XN, H, sc.DGATE, FF);
            gemm(wu->native_data, wu->native_type, H, FF, sc.XN, H, sc.DUP, FF);
            swiglu_mul(sc.DGATE, sc.DUP, sc.DGATE, T * FF, cs);
            gemm(wd->native_data, wd->native_type, FF, H, sc.DGATE, FF, sc.SH, H);
            ds2pf::combine(sc.PARTS, sc.WEIGHTS, sc.SH, sc.OUT, T, K, H, cs);
        }
        add_inplace(sc.X, sc.OUT, T * H, cs);
    }
    return true;
}

}  // namespace strata::core
