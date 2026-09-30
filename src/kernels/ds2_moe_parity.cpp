#include "hip/hip_runtime.h"
// src/kernels/ds2_moe_parity.cpp - the deepseek2 router and SwiGLU against a scalar reference.
//
// The router is the new arithmetic: SIGMOID over the logits, `exp_probs_b` added for SELECTION only (the weights
// come from the unbiased sigmoid), a stable descending top-k with ties by ascending index, then the
// `expert_weights_norm` / `scale` epilogue.  Every reading a transcription gets wrong is asserted OBSERVABLE
// apart below, not just present.
#include "strata/kernels/ds2_moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
struct Dev {
    T* p = nullptr;
    ~Dev() { if (p) hipFree(p); }
    void put(const std::vector<T>& v) {
        if (!p) check(hipMalloc(&p, v.size() * sizeof(T)), "hipMalloc");
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

struct Ref {
    std::vector<int32_t> ids;
    std::vector<double> weights;
};
Ref ref_router(const std::vector<float>& logits, const std::vector<float>& bias, int n_expert, int k, float scale,
               bool norm) {
    std::vector<double> probs((size_t) n_expert), sel((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) logits[(size_t) e]));
        sel[(size_t) e] = probs[(size_t) e] + (double) bias[(size_t) e];
    }
    std::vector<int32_t> order((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) order[(size_t) e] = e;
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sel[(size_t) a] > sel[(size_t) b]; });
    Ref r;
    for (int i = 0; i < k; ++i) {
        r.ids.push_back(order[(size_t) i]);
        r.weights.push_back(probs[(size_t) order[(size_t) i]]);
    }
    if (norm) {
        double s = 0;
        for (double w : r.weights) s += w;
        const double sc = std::fmax(s, 6.103515625e-05);
        for (auto& w : r.weights) w /= sc;
    }
    if (scale != 0.0f && scale != 1.0f)
        for (auto& w : r.weights) w *= (double) scale;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a != "--selftest") { std::fprintf(stderr, "usage: ds2_moe_parity [--selftest]\n"); return 2; }
    }

    const int NE = 64, K = 4;
    std::mt19937 rng(20261001u);
    std::normal_distribution<float> gauss(0.0f, 2.0f);
    hipStream_t cs = nullptr;
    check(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking), "stream");

    for (int seed = 0; seed < 8; ++seed) {
        std::vector<float> logits((size_t) NE), bias((size_t) NE);
        for (auto& v : logits) v = gauss(rng);
        for (auto& v : bias) v = 0.5f * gauss(rng);
        const float scale = (seed % 2) ? 1.8f : 1.0f;
        const bool norm = (seed % 3) != 0;
        Dev<float> dl, db;
        Dev<int32_t> did;
        Dev<float> dw;
        dl.put(logits);
        db.put(bias);
        check(hipMalloc(&did.p, (size_t) K * 4), "ids");
        check(hipMalloc(&dw.p, (size_t) K * 4), "weights");
        strata::kernels::ds2_router(dl.p, db.p, 1, NE, K, scale, norm, did.p, dw.p, (void*) cs);
        check(hipStreamSynchronize(cs), "sync");
        const auto gid = did.get((size_t) K);
        const auto gw = dw.get((size_t) K);
        const Ref ref = ref_router(logits, bias, NE, K, scale, norm);
        const bool ids_ok = std::equal(gid.begin(), gid.end(), ref.ids.begin());
        double worst = 0;
        for (int i = 0; i < K; ++i) {
            const double den = std::fabs(ref.weights[(size_t) i]);
            worst = std::max(worst, std::fabs((double) gw[(size_t) i] - ref.weights[(size_t) i]) / (den > 1e-12 ? den : 1e-12));
        }
        require("router seed " + std::to_string(seed) + " (scale " + std::to_string(scale) + ", norm " +
                    std::to_string((int) norm) + "): ids match",
                ids_ok);
        require("router seed " + std::to_string(seed) + ": weights within 1e-6", worst <= 1e-6,
                "worst " + std::to_string(worst));
    }

    // The SELECTION BIAS must be observable: with a bias large enough to reorder, the biased and unbiased
    // selections differ.  A kernel that dropped the bias would still produce a well-formed top-4.
    {
        std::vector<float> logits((size_t) NE, 0.0f), bias((size_t) NE, 0.0f);
        for (int e = 0; e < NE; ++e) logits[(size_t) e] = -0.001f * (float) e;   // expert 0 is the largest
        bias[(size_t) (NE - 1)] = 100.0f;                                         // and the bias promotes it
        Dev<float> dl, db;
        Dev<int32_t> did;
        Dev<float> dw;
        dl.put(logits);
        db.put(bias);
        check(hipMalloc(&did.p, (size_t) K * 4), "ids");
        check(hipMalloc(&dw.p, (size_t) K * 4), "weights");
        strata::kernels::ds2_router(dl.p, db.p, 1, NE, K, 1.0f, false, did.p, dw.p, (void*) cs);
        check(hipStreamSynchronize(cs), "sync");
        const auto gid = did.get((size_t) K);
        require("the selection bias reorders the top-k", gid[0] == NE - 1, "first id " + std::to_string(gid[0]));
        const auto gw = dw.get((size_t) K);
        const Ref ref = ref_router(logits, bias, NE, K, 1.0f, false);
        require("the weight is the UNBIASED sigmoid", std::fabs((double) gw[0] - ref.weights[0]) <= 1e-6,
                "got " + std::to_string(gw[0]) + " want " + std::to_string(ref.weights[0]));
    }

    // ================= swiglu_mul =================
    {
        const int64_t n = 4096;
        std::vector<float> g((size_t) n), u((size_t) n);
        for (auto& v : g) v = gauss(rng);
        for (auto& v : u) v = gauss(rng);
        Dev<float> dg, du, dout;
        dg.put(g);
        du.put(u);
        check(hipMalloc(&dout.p, (size_t) n * 4), "out");
        strata::kernels::swiglu_mul(dg.p, du.p, dout.p, n, (void*) cs);
        check(hipStreamSynchronize(cs), "sync");
        const auto got = dout.get((size_t) n);
        double worst = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double want = ((double) g[(size_t) i] / (1.0 + std::exp(-(double) g[(size_t) i]))) * (double) u[(size_t) i];
            const double den = std::fabs(want) > 1e-9 ? std::fabs(want) : 1e-9;
            worst = std::max(worst, std::fabs((double) got[(size_t) i] - want) / den);
        }
        require("swiglu_mul within 1e-6", worst <= 1e-6, "worst " + std::to_string(worst));
    }

    check(hipStreamDestroy(cs), "stream destroy");
    std::printf("\nds2_moe_parity: %s (%d failing checks)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
    return g_bad == 0 ? 0 : 1;
}
