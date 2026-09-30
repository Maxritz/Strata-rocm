#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "hip/hip_runtime.h"
// src/kernels/mla_parity.cpp - the MLA forward of docs/DEEPSEEK.md section 5, against a scalar reference.
//
// Two gates:
//   * the MLA CORE kernels (`mla_split_q`, `mla_rope`, `mla_write_kv`, `mla_attention`) against scalar references
//     at their own arithmetic tolerances - these are the only new device code, so they are the strict gate;
//   * the WHOLE `mla_layer` forward against a double reference that models the Q8_0 weights and the Q8_1
//     activation quantization `native_mmvq` performs.  The chained projections quantize their own inputs, so a
//     reference that re-derives those inputs can disagree at an int8 rounding boundary; that is why the whole
//     layer's tolerance is the contract's, not the core kernels'.
#include "strata/core/layer.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mla.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
struct Dev {
    T* p = nullptr;
    Dev() = default;
    explicit Dev(size_t n) { alloc(n); }
    ~Dev() { if (p) hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void alloc(size_t n) {
        if (p) { hipFree(p); p = nullptr; }
        if (n) check(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
    }
    void put(const std::vector<T>& v) {
        if (!p) alloc(v.size());
        check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "H2D");
    }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "D2H");
        return v;
    }
};

int g_bad = 0;
void require(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-58s %-6s %s\n", name.c_str(), ok ? "ok" : "*** BAD ***", detail.c_str());
    if (!ok) ++g_bad;
}
void report(const std::string& name, double v, const char* unit) {
    std::printf("  %-58s %s %.3e\n", name.c_str(), unit, v);
}
double l1_rel(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs((double) a[i] - b[i]);
        den += std::fabs(b[i]);
    }
    return num / (den > 1e-30 ? den : 1e-30);
}
double max_rel(const std::vector<float>& a, const std::vector<double>& b, double floor_mag) {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - b[i]);
        worst = std::max(worst, d / std::max(b[i] < 0 ? -b[i] : b[i], floor_mag));
    }
    return worst;
}

/// `w` is [n_out][n_in] row-major (row i is output i, n_in contiguous).  Produces the GGUF Q8_0 block layout
/// `native_mmvq` reads: n_out rows, each of n_in/32 blocks of { half d; int8 qs[32]; }.
std::vector<uint8_t> make_q8_0(const std::vector<float>& w, int n_in, int n_out) {
    std::vector<uint8_t> out;
    out.reserve((size_t) n_out * (n_in / 32) * 34);
    for (int r = 0; r < n_out; ++r) {
        for (int b = 0; b < n_in / 32; ++b) {
            const float* row = &w[(size_t) r * n_in + (size_t) b * 32];
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(row[j]));
            const float d = amax / 127.0f;
            const uint16_t dh = f16_from_f32(d);
            out.push_back((uint8_t) (dh & 0xff));
            out.push_back((uint8_t) (dh >> 8));
            for (int j = 0; j < 32; ++j) {
                const float q = d == 0.0f ? 0.0f : std::round(row[j] / d);
                out.push_back((uint8_t) (int8_t) q);
            }
        }
    }
    return out;
}

/// The scalar reference for one native MMVQ: Q8_0 weight dequant times the Q8_1-quantized activation, in double.
std::vector<double> ref_mmvq(const std::vector<uint8_t>& w, int n_in, int n_out, const std::vector<float>& x) {
    std::vector<float> xq((size_t) n_in);
    for (int b = 0; b < n_in / 32; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[(size_t) b * 32 + j]));
        const float d = amax / 127.0f;
        const float dh = f32_from_f16(f16_from_f32(d));
        for (int j = 0; j < 32; ++j) {
            const float q = d == 0.0f ? 0.0f : std::round(x[(size_t) b * 32 + j] / d);
            xq[(size_t) b * 32 + j] = dh * q;
        }
    }
    std::vector<double> y((size_t) n_out, 0.0);
    for (int r = 0; r < n_out; ++r) {
        double acc = 0;
        for (int b = 0; b < n_in / 32; ++b) {
            const uint8_t* blk = w.data() + ((size_t) r * (n_in / 32) + b) * 34;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float dw = f32_from_f16(dh);
            const int8_t* qs = (const int8_t*) (blk + 2);
            for (int j = 0; j < 32; ++j) acc += (double) (dw * (float) qs[j]) * (double) xq[(size_t) b * 32 + j];
        }
        y[r] = acc;
    }
    return y;
}

void ref_rms(std::vector<float>& x, const std::vector<float>& w, float eps) {
    double ss = 0;
    for (float v : x) ss += (double) v * v;
    const double inv = 1.0 / std::sqrt(ss / (double) x.size() + (double) eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((double) x[i] * inv * (double) w[i]);
}

void ref_rope(std::vector<float>& x, int n_rot, float base, int pos) {
    const int half = n_rot / 2;
    const float ts = std::pow(base, -2.0f / (float) n_rot);
    for (int i = 0; i < half; ++i) {
        const float th = (float) pos * std::pow(ts, (float) i);
        const float c = std::cos(th), s = std::sin(th);
        const float a = x[i], bb = x[i + half];
        x[i] = a * c - bb * s;
        x[i + half] = a * s + bb * c;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: mla_parity [--selftest]\n"); return 2; }
    }
    (void) selftest;

    hipStream_t cs = nullptr;
    check(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking), "stream");

    constexpr int64_t N = 2048;      // n_embd
    constexpr int64_t NH = 20;       // n_head
    constexpr int64_t HD = 256;      // head_dim (key_length_mla)
    constexpr int64_t NROT = 64;
    constexpr int64_t NOPE = HD - NROT;
    constexpr int64_t KV = 512;      // n_lora_kv
    constexpr int64_t QL = 768;      // n_lora_q
    constexpr int64_t MAXC = 8;
    constexpr float EPS = 1e-5f;
    constexpr float BASE = 1e6f;

    strata::core::ModelGeometry g;
    g.n_embd = N;
    g.n_layers = 1;
    g.mla = 1;
    g.n_head = NH;
    g.n_head_kv = 1;
    g.head_dim = HD;
    g.n_rot = NROT;
    g.n_lora_q = QL;
    g.n_lora_kv = KV;
    g.n_expert = 64;
    g.n_ff = 1536;
    g.qsa_interval = 4;
    g.hc = 1;

    std::printf("mla_parity: n_embd %lld  n_head %lld  head_dim %lld (nope %lld, rot %lld)  kv_lora %lld  q_lora "
                "%lld\n\n",
                (long long) N, (long long) NH, (long long) HD, (long long) NOPE, (long long) NROT, (long long) KV,
                (long long) QL);

    std::mt19937 rng(20261001u);
    std::normal_distribution<float> gauss(0.0f, 0.05f);
    auto rnd = [&]() { return gauss(rng); };
    std::vector<float> wq_a((size_t) N * QL), wq_b((size_t) QL * NH * HD), wkv((size_t) N * (KV + NROT));
    std::vector<float> wk_b((size_t) NOPE * KV * NH), wv_b((size_t) KV * HD * NH), wo((size_t) (NH * HD) * N);
    for (auto& v : wq_a) v = rnd();
    for (auto& v : wq_b) v = rnd();
    for (auto& v : wkv) v = rnd();
    for (auto& v : wk_b) v = rnd();
    for (auto& v : wv_b) v = rnd();
    for (auto& v : wo) v = rnd();
    std::vector<float> q_a_norm((size_t) QL), kv_a_norm((size_t) KV);
    for (auto& v : q_a_norm) v = 1.0f + 0.1f * rnd() * 20.0f;
    for (auto& v : kv_a_norm) v = 1.0f + 0.1f * rnd() * 20.0f;

    const std::vector<uint8_t> bq_a = make_q8_0(wq_a, (int) N, (int) QL);
    const std::vector<uint8_t> bq_b = make_q8_0(wq_b, (int) QL, (int) (NH * HD));
    const std::vector<uint8_t> bkv = make_q8_0(wkv, (int) N, (int) (KV + NROT));
    std::vector<uint8_t> bwkb, bwvb;
    for (int64_t h = 0; h < NH; ++h) {
        std::vector<float> m(begin(wk_b) + (size_t) h * NOPE * KV, begin(wk_b) + (size_t) (h + 1) * NOPE * KV);
        const auto bytes = make_q8_0(m, (int) NOPE, (int) KV);
        bwkb.insert(bwkb.end(), bytes.begin(), bytes.end());
    }
    for (int64_t h = 0; h < NH; ++h) {
        std::vector<float> m(begin(wv_b) + (size_t) h * KV * HD, begin(wv_b) + (size_t) (h + 1) * KV * HD);
        const auto bytes = make_q8_0(m, (int) KV, (int) HD);
        bwvb.insert(bwvb.end(), bytes.begin(), bytes.end());
    }
    const std::vector<uint8_t> bwo = make_q8_0(wo, (int) (NH * HD), (int) N);

    Dev<uint8_t> dq_a, dq_b, dkv, dwkb, dwvb, dwo;
    Dev<float> dnorm_q, dnorm_kv;
    dq_a.put(bq_a);
    dq_b.put(bq_b);
    dkv.put(bkv);
    dwkb.put(bwkb);
    dwvb.put(bwvb);
    dwo.put(bwo);
    dnorm_q.put(q_a_norm);
    dnorm_kv.put(kv_a_norm);

    const int64_t max_in = std::max({N, QL, KV, NOPE, NH * HD});
    Dev<uint8_t> dq8_1((size_t) strata::kernels::native_q8_1_bytes((int) max_in));
    Dev<uint8_t> dst_kv, dst_v;
    Dev<int32_t> dpos;

    const uint64_t state_bytes = strata::core::mla_state_bytes(g, MAXC);
    const uint64_t buf_bytes = strata::core::mla_buffers_bytes(g, MAXC);
    void* sraw = nullptr;
    void* braw = nullptr;
    check(hipMalloc(&sraw, state_bytes), "state alloc");
    check(hipMalloc(&braw, buf_bytes), "buffers alloc");
    strata::core::MlaState st;
    strata::core::MlaBuffers b;
    strata::core::mla_state_init(g, MAXC, sraw, st);
    strata::core::mla_buffers_init(g, MAXC, braw, b);
    strata::core::mla_state_zero(st, (void*) cs);
    check(hipStreamSynchronize(nullptr), "state zero");

    // ================= core: mla_split_q =================
    {
        std::vector<float> q((size_t) NH * HD), qn((size_t) NH * NOPE), qp((size_t) NH * NROT);
        for (auto& v : q) v = rnd() * 20.0f;
        Dev<float> dq, dqn((size_t) NH * NOPE), dqp((size_t) NH * NROT);
        dq.put(q);
        strata::kernels::mla_split_q(dq.p, dqn.p, dqp.p, NH, NOPE, NROT, (void*) nullptr);
        const auto gn = dqn.get((size_t) NH * NOPE), gp = dqp.get((size_t) NH * NROT);
        int bad = 0;
        for (int64_t h = 0; h < NH; ++h)
            for (int64_t d = 0; d < NOPE; ++d) if (gn[(size_t) h * NOPE + d] != q[(size_t) h * HD + d]) ++bad;
        for (int64_t h = 0; h < NH; ++h)
            for (int64_t d = 0; d < NROT; ++d)
                if (gp[(size_t) h * NROT + d] != q[(size_t) h * HD + NOPE + d]) ++bad;
        require("mla_split_q: the nope|pe split is bit-exact", bad == 0, std::to_string(bad) + " wrong");
    }

    // ================= core: mla_rope =================
    for (int pos : {0, 1, 137}) {
        std::vector<float> x((size_t) NH * NROT), want;
        for (auto& v : x) v = rnd() * 10.0f;
        want = x;
        for (int64_t h = 0; h < NH; ++h) {
            std::vector<float> row(want.begin() + (size_t) h * NROT, want.begin() + (size_t) (h + 1) * NROT);
            ref_rope(row, (int) NROT, BASE, pos);
            std::copy(row.begin(), row.end(), want.begin() + (size_t) h * NROT);
        }
        Dev<float> dx, dout((size_t) NH * NROT);
        Dev<int32_t> dp(1);
        std::vector<int32_t> pv = {pos};
        dp.put(pv);
        dx.put(x);
        strata::kernels::mla_rope(dx.p, dout.p, NH, NROT, BASE, dp.p, (void*) nullptr);
        const auto got = dout.get((size_t) NH * NROT);
        // The rotation's two outputs subtract/cancel, so the denominator is the magnitude of the TERMS
        // (|a*cos| + |b*sin|), not of the result - the same reason `qsa_parity` uses the term magnitude.
        const int half = (int) (NROT / 2);
        const float ts = std::pow(BASE, -2.0f / (float) NROT);
        double worst = 0;
        for (int64_t h = 0; h < NH; ++h)
            for (int i = 0; i < half; ++i) {
                const float th = (float) pos * std::pow(ts, (float) i);
                const float c = std::cos(th), s = std::sin(th);
                const float a = x[(size_t) h * NROT + i], bb = x[(size_t) h * NROT + half + i];
                const double t0 = std::fabs(a * c) + std::fabs(bb * s);
                const double t1 = std::fabs(a * s) + std::fabs(bb * c);
                worst = std::max(worst, std::fabs(got[(size_t) h * NROT + i] - want[(size_t) h * NROT + i]) / std::max(t0, 1e-6));
                worst = std::max(worst, std::fabs(got[(size_t) h * NROT + half + i] - want[(size_t) h * NROT + half + i]) / std::max(t1, 1e-6));
            }
        report("mla_rope pos " + std::to_string(pos) + " vs the f32 reference", worst, "rel/|terms|");
        // 1e-5, not 1e-6: at a large position the f32 `powf`/`cosf`/`sinf` argument reduction is the residual,
        // and the reference computes the same formula in f64.  The rotation itself is exact (pos 0 is 0.0).
        require("mla_rope pos " + std::to_string(pos) + " within 1e-5", worst <= 1e-5);
    }

    // ================= core: mla_write_kv =================
    {
        std::vector<float> latent((size_t) KV), kpe((size_t) NROT);
        for (auto& v : latent) v = rnd() * 10.0f;
        for (auto& v : kpe) v = rnd() * 10.0f;
        Dev<float> dl, dk;
        dl.put(latent);
        dk.put(kpe);
        Dev<uint16_t> dkv((size_t) MAXC * (KV + NROT)), dv((size_t) MAXC * KV);
        strata::kernels::mla_write_kv(dkv.p, dv.p, dl.p, dk.p, 3, KV, NROT, (void*) nullptr);
        const auto gkv = dkv.get((size_t) MAXC * (KV + NROT)), gv = dv.get((size_t) MAXC * KV);
        int bad = 0;
        for (int64_t d = 0; d < KV; ++d) {
            if (gkv[(size_t) 3 * (KV + NROT) + d] != f16_from_f32(latent[(size_t) d])) ++bad;
            if (gv[(size_t) 3 * KV + d] != f16_from_f32(latent[(size_t) d])) ++bad;
        }
        for (int64_t d = 0; d < NROT; ++d)
            if (gkv[(size_t) 3 * (KV + NROT) + KV + d] != f16_from_f32(kpe[(size_t) d])) ++bad;
        require("mla_write_kv: the cell is bit-exact", bad == 0, std::to_string(bad) + " wrong bits");
    }

    // ================= core: mla_attention vs a double softmax =================
    {
        const int64_t cells = 7;
        std::vector<float> q_abs((size_t) NH * KV), q_pe((size_t) NH * NROT);
        std::vector<uint16_t> kv((size_t) cells * (KV + NROT)), v((size_t) cells * KV);
        std::vector<float> kvf((size_t) cells * (KV + NROT)), vf((size_t) cells * KV);
        for (auto& x : q_abs) x = rnd() * 8.0f;
        for (auto& x : q_pe) x = rnd() * 8.0f;
        for (size_t i = 0; i < kv.size(); ++i) { kvf[i] = rnd() * 8.0f; kv[i] = f16_from_f32(kvf[i]); }
        for (size_t i = 0; i < v.size(); ++i) { vf[i] = rnd() * 8.0f; v[i] = f16_from_f32(vf[i]); }
        Dev<float> dqa, dqp;
        Dev<uint16_t> dkv, dv;
        Dev<float> dsc((size_t) NH * cells), dat((size_t) NH * KV);
        dqa.put(q_abs);
        dqp.put(q_pe);
        dkv.put(kv);
        dv.put(v);
        const float scale = 1.0f / std::sqrt((float) HD);
        strata::kernels::mla_attention(dqa.p, dqp.p, dkv.p, dv.p, cells, NH, KV, NROT, scale, dsc.p, dat.p,
                                       (void*) nullptr);
        const auto got = dat.get((size_t) NH * KV);
        std::vector<double> want((size_t) NH * KV, 0.0), terms((size_t) NH * KV, 0.0);
        for (int64_t h = 0; h < NH; ++h) {
            std::vector<double> sc((size_t) cells);
            for (int64_t j = 0; j < cells; ++j) {
                double dot = 0;
                for (int64_t d = 0; d < KV; ++d) dot += (double) q_abs[(size_t) h * KV + d] * (double) f32_from_f16(kv[(size_t) j * (KV + NROT) + d]);
                for (int64_t d = 0; d < NROT; ++d) dot += (double) q_pe[(size_t) h * NROT + d] * (double) f32_from_f16(kv[(size_t) j * (KV + NROT) + KV + d]);
                sc[(size_t) j] = dot * scale;
            }
            const double mx = *std::max_element(sc.begin(), sc.end());
            double sum = 0;
            for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
            for (int64_t d = 0; d < KV; ++d) {
                double acc = 0, tm = 0;
                for (int64_t j = 0; j < cells; ++j) {
                    const double wj = sc[(size_t) j] / sum;
                    const double vv = (double) f32_from_f16(v[(size_t) j * KV + d]);
                    acc += wj * vv;
                    tm += std::fabs(wj * vv);
                }
                want[(size_t) h * KV + d] = acc;
                terms[(size_t) h * KV + d] = tm;
            }
        }
        // The weighted sum over cells cancels, so the denominator is the magnitude of its TERMS.
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, std::fabs((double) got[i] - want[i]) / std::max(terms[i], 1e-6));
        report("mla_attention over 7 cells vs a double softmax", worst, "rel/|terms|");
        require("mla_attention within 1e-6", worst <= 1e-6);
    }

    // ================= whole layer vs the scalar reference =================
    strata::core::MlaWeights w;
    w.wq_a = dq_a.p; w.wq_a_type = 8;
    w.wq_b = dq_b.p; w.wq_b_type = 8;
    w.wkv_a_mqa = dkv.p; w.wkv_a_type = 8;
    w.wk_b = dwkb.p; w.wk_b_type = 8;
    w.wv_b = dwvb.p; w.wv_b_type = 8;
    w.wo = dwo.p; w.wo_type = 8;
    w.q_a_norm = dnorm_q.p;
    w.kv_a_norm = dnorm_kv.p;
    w.q8_1 = dq8_1.p;
    w.norm_eps = EPS;
    w.rope_freq_base = BASE;

    double worst_l1 = 0;
    for (int32_t pos = 0; pos < 3; ++pos) {
        std::vector<float> x((size_t) N);
        for (auto& v : x) v = rnd() * 8.0f;
        Dev<float> dx, dout((size_t) N);
        dx.put(x);
        std::string err;
        if (!strata::core::mla_layer(g, w, st, b, dx.p, dout.p, pos, (void*) cs, err)) {
            std::printf("mla_layer failed: %s\n", err.c_str());
            return 1;
        }
        check(hipStreamSynchronize(cs), "mla_layer sync");
        const auto got = dout.get((size_t) N);

        std::vector<float> qa = [&] { std::vector<double> d = ref_mmvq(bq_a, (int) N, (int) QL, x); return std::vector<float>(d.begin(), d.end()); }();
        ref_rms(qa, q_a_norm, EPS);
        std::vector<float> q = [&] { std::vector<double> d = ref_mmvq(bq_b, (int) QL, (int) (NH * HD), qa); return std::vector<float>(d.begin(), d.end()); }();
        std::vector<float> kv_a = [&] { std::vector<double> d = ref_mmvq(bkv, (int) N, (int) (KV + NROT), x); return std::vector<float>(d.begin(), d.end()); }();
        std::vector<float> lat(kv_a.begin(), kv_a.begin() + KV), kpe(kv_a.begin() + KV, kv_a.end());
        ref_rms(lat, kv_a_norm, EPS);
        for (int64_t h = 0; h < NH; ++h) {
            std::vector<float> qn(q.begin() + (size_t) h * HD, q.begin() + (size_t) h * HD + NOPE);
            std::vector<float> qp(q.begin() + (size_t) h * HD + NOPE, q.begin() + (size_t) (h + 1) * HD);
            ref_rope(qp, (int) NROT, BASE, pos);
            std::copy(qp.begin(), qp.end(), q.begin() + (size_t) h * HD + NOPE);
            (void) qn;
        }
        ref_rope(kpe, (int) NROT, BASE, pos);
        // The reference's own cache of the cells written so far, forwarded across the pos iterations.
        static std::vector<std::vector<float>> ref_kv(MAXC), ref_v(MAXC);
        std::vector<float> q_abs((size_t) NH * KV);
        for (int64_t h = 0; h < NH; ++h) {
            std::vector<float> qn(q.begin() + (size_t) h * HD, q.begin() + (size_t) h * HD + NOPE);
            const std::vector<uint8_t> slice(bwkb.begin() + (size_t) h * (NOPE / 32 * 34) * KV,
                                             bwkb.begin() + (size_t) (h + 1) * (NOPE / 32 * 34) * KV);
            std::vector<double> a = ref_mmvq(slice, (int) NOPE, (int) KV, qn);
            for (int64_t d = 0; d < KV; ++d) q_abs[(size_t) h * KV + d] = (float) a[(size_t) d];
        }
        // cache the current cell (fp16)
        std::vector<float> cell((size_t) (KV + NROT));
        for (int64_t d = 0; d < KV; ++d) { cell[(size_t) d] = f32_from_f16(f16_from_f32(lat[(size_t) d])); }
        for (int64_t d = 0; d < NROT; ++d) cell[(size_t) KV + d] = f32_from_f16(f16_from_f32(kpe[(size_t) d]));
        ref_kv[(size_t) pos] = cell;
        std::vector<float> vcell((size_t) KV);
        for (int64_t d = 0; d < KV; ++d) vcell[(size_t) d] = cell[(size_t) d];
        ref_v[(size_t) pos] = vcell;
        // attention over cells 0..pos
        std::vector<float> attn((size_t) NH * HD);
        for (int64_t h = 0; h < NH; ++h) {
            std::vector<float> qp(q.begin() + (size_t) h * HD + NOPE, q.begin() + (size_t) (h + 1) * HD);
            const double scale = 1.0 / std::sqrt((double) HD);
            std::vector<double> sc((size_t) pos + 1);
            for (int32_t j = 0; j <= pos; ++j) {
                double dot = 0;
                for (int64_t d = 0; d < KV; ++d) dot += (double) q_abs[(size_t) h * KV + d] * (double) ref_kv[(size_t) j][(size_t) d];
                for (int64_t d = 0; d < NROT; ++d) dot += (double) qp[(size_t) d] * (double) ref_kv[(size_t) j][(size_t) KV + d];
                sc[(size_t) j] = dot * scale;
            }
            const double mx = *std::max_element(sc.begin(), sc.end());
            double sum = 0;
            for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
            std::vector<float> al((size_t) KV, 0.0f);
            for (int64_t d = 0; d < KV; ++d) {
                double acc = 0;
                for (int32_t j = 0; j <= pos; ++j) acc += sc[(size_t) j] / sum * (double) ref_v[(size_t) j][(size_t) d];
                al[(size_t) d] = (float) acc;
            }
            const std::vector<uint8_t> slice(bwvb.begin() + (size_t) h * (KV / 32 * 34) * HD,
                                             bwvb.begin() + (size_t) (h + 1) * (KV / 32 * 34) * HD);
            std::vector<double> a = ref_mmvq(slice, (int) KV, (int) HD, al);
            for (int64_t d = 0; d < HD; ++d) attn[(size_t) h * HD + d] = (float) a[(size_t) d];
        }
        std::vector<double> outref = ref_mmvq(bwo, (int) (NH * HD), (int) N, attn);
        const double l1 = l1_rel(got, outref);
        const double mx = max_rel(got, outref, 1e-2);
        std::printf("  pos %d: L1 rel %.3e   max rel %.3e\n", pos, l1, mx);
        worst_l1 = std::max(worst_l1, l1);
    }
    report("whole mla_layer vs the scalar reference (worst L1)", worst_l1, "rel");
    require("mla_layer within the contract's 1e-3 L1", worst_l1 <= 1e-3);

    hipFree(sraw);
    hipFree(braw);
    std::printf("\nmla_parity: %s (%d failing checks)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
    return g_bad == 0 ? 0 : 1;
}

