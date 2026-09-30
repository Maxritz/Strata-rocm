#include "hip/hip_runtime.h"
// src/kernels/ds2_prefill_parity.cpp - the batched MLA companions against their single-token namesakes.
//   build/ds2_prefill_parity
// T = 1 must reproduce `mla_attention` to within float rounding; the chunk T > 1 is checked against a
// per-token loop of the same kernel.
#include "strata/kernels/ds2_prefill.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mla.hpp"
#include "strata/kernels/native_gr_norm.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace pf = strata::kernels::ds2pf;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipStream_t s;
    hipStreamCreate(&s);
    const int NH = 4, KV = 64, ROT = 16, W = KV + ROT, MAXC = 12, T = 4, POS0 = 0;
    const float scale = 0.35f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qabs((size_t) T * NH * KV), qpe((size_t) T * NH * ROT);
    for (auto& x : qabs) x = nd(rng);
    for (auto& x : qpe) x = nd(rng);
    std::vector<uint16_t> kv((size_t) MAXC * W), v((size_t) MAXC * KV);
    for (auto& x : kv) x = k::f16_from_f32(0.3f * nd(rng));
    for (auto& x : v) x = k::f16_from_f32(0.3f * nd(rng));
    float *d_qabs, *d_qpe, *d_attn, *d_ref, *d_scr;
    uint16_t *d_kv, *d_v;
    hipMalloc(&d_qabs, qabs.size() * 4);
    hipMalloc(&d_qpe, qpe.size() * 4);
    hipMalloc(&d_attn, (size_t) T * NH * KV * 4);
    hipMalloc(&d_ref, (size_t) T * NH * KV * 4);
    hipMalloc(&d_scr, (size_t) NH * (POS0 + T) * 4);
    hipMalloc(&d_kv, kv.size() * 2);
    hipMalloc(&d_v, v.size() * 2);
    hipMemcpy(d_qabs, qabs.data(), qabs.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_qpe, qpe.data(), qpe.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_kv, kv.data(), kv.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(d_v, v.data(), v.size() * 2, hipMemcpyHostToDevice);
    // batched
    pf::attention(d_qabs, d_qpe, d_kv, d_v, POS0, T, NH, KV, ROT, scale, d_attn, s);
    // per-token reference
    for (int t = 0; t < T; ++t)
        k::mla_attention(d_qabs + (size_t) t * NH * KV, d_qpe + (size_t) t * NH * ROT, d_kv, d_v, POS0 + t + 1, NH,
                         KV, ROT, scale, d_scr, d_ref + (size_t) t * NH * KV, s);
    hipStreamSynchronize(s);
    std::vector<float> got((size_t) T * NH * KV), ref((size_t) T * NH * KV);
    hipMemcpy(got.data(), d_attn, got.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(ref.data(), d_ref, ref.size() * 4, hipMemcpyDeviceToHost);
    const double r = rel(got, ref);
    std::printf("attention batch vs mla_attention: rel %.3e  %s\n", r, r < 1e-5 ? "ok" : "FAIL");
    int failures = (r < 1e-5) ? 0 : 1;

    // ---- rope_slice on the q_pe half of a [T, NH, HD] buffer vs mla_rope per token ----
    {
        const int HD = 192 + 64, NOPE = 192;
        std::vector<float> x((size_t) T * NH * HD);
        for (auto& a : x) a = nd(rng);
        std::vector<float> rop((size_t) T * NH * 64);
        float *d_x, *d_r;
        int32_t* d_pos;
        hipMalloc(&d_x, x.size() * 4);
        hipMalloc(&d_r, rop.size() * 4);
        hipMalloc(&d_pos, 4);
        hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice);
        pf::rope_slice(d_x, d_r, T * NH, HD, 64, NOPE, 64, 1000000.0f, POS0, NH, s);
        // reference: per token, gather the q_pe slice to [NH, 64] and mla_rope at that position
        std::vector<float> qp_ref((size_t) T * NH * 64);
        for (int t = 0; t < T; ++t) {
            std::vector<float> sl((size_t) NH * 64);
            for (int h = 0; h < NH; ++h)
                std::memcpy(sl.data() + h * 64, x.data() + ((size_t) t * NH + h) * HD + NOPE, 64 * 4);
            float* d_sl;
            hipMalloc(&d_sl, sl.size() * 4);
            hipMemcpy(d_sl, sl.data(), sl.size() * 4, hipMemcpyHostToDevice);
            const int32_t p = POS0 + t;
            hipMemcpy(d_pos, &p, 4, hipMemcpyHostToDevice);
            k::mla_rope(d_sl, d_sl, NH, 64, 1000000.0f, d_pos, s);
            hipMemcpy(qp_ref.data() + (size_t) t * NH * 64, d_sl, sl.size() * 4, hipMemcpyDeviceToHost);
            hipFree(d_sl);
        }
        hipMemcpy(rop.data(), d_r, rop.size() * 4, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < rop.size(); ++i)
            if (std::fabs(rop[i] - qp_ref[i]) > 1e-3) {
                std::printf("  [dbg] first mismatch at %zu (row %zu d %zu): got %g ref %g\n", i, i / 64, i % 64,
                            rop[i], qp_ref[i]);
                break;
            }
        const double rr = rel(rop, qp_ref);
        std::printf("rope_slice q_pe vs mla_rope: rel %.3e  %s\n", rr, rr < 1e-5 ? "ok" : "FAIL");
        if (!(rr < 1e-5)) ++failures;
        hipFree(d_x);
        hipFree(d_r);
        hipFree(d_pos);
    }

    // ---- rms_strided vs native_gr_rms_norm_weighted on the gathered rows ----
    {
        const int cols = 48, stride = 80;
        std::vector<float> x((size_t) T * stride), w(cols);
        for (auto& a : x) a = nd(rng);
        for (auto& a : w) a = 0.5f + 0.5f * nd(rng);
        float *d_xi, *d_wi, *d_o;
        hipMalloc(&d_xi, x.size() * 4);
        hipMalloc(&d_wi, w.size() * 4);
        hipMalloc(&d_o, (size_t) T * cols * 4);
        hipMemcpy(d_xi, x.data(), x.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(d_wi, w.data(), w.size() * 4, hipMemcpyHostToDevice);
        pf::rms_strided(d_xi, d_wi, d_o, T, cols, stride, 1e-5f, s);
        // reference: per-row RMS with a per-COLUMN gamma broadcast, in double
        std::vector<float> gotr((size_t) T * cols), rr_ref((size_t) T * cols);
        for (int t = 0; t < T; ++t) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) {
                const double v = x[(size_t) t * stride + c];
                acc += v * v;
            }
            const double inv = 1.0 / std::sqrt(acc / cols + 1e-5);
            for (int c = 0; c < cols; ++c)
                rr_ref[(size_t) t * cols + c] =
                    (float) (x[(size_t) t * stride + c] * inv * (double) w[c]);
        }
        hipMemcpy(gotr.data(), d_o, gotr.size() * 4, hipMemcpyDeviceToHost);
        const double rr = rel(gotr, rr_ref);
        std::printf("rms_strided vs per-row double RMS: rel %.3e  %s\n", rr, rr < 1e-5 ? "ok" : "FAIL");
        if (!(rr < 1e-5)) ++failures;
        hipFree(d_xi);
        hipFree(d_wi);
        hipFree(d_o);
    }

    std::printf("ds2_prefill_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
