// src/kernels/native_mmvq_heads_parity.cpp - the head-batched MMVQ against the per-head loop it replaces.
//
// `native_mmvq_heads` exists because the MLA absorption (`q_abs[h] = wk_b[h] @ q_nope[h]`) and up-projection
// (`attn[h] = wv_b[h] @ attn_latent[h]`) are `n_head` independent matrix-vector products with a different
// weight AND a different activation per head, which the `ncols` batching cannot express.  The per-head host
// loop was 2*`n_head` launches and measured as exposed per-layer GPU time; the batched kernel is a bitwise
// mirror with a `blockIdx.z` head offset.
//
// THE GATE: every head's output from `native_mmvq_heads` must be BITWISE equal to the same head's output from
// a single `native_mmvq` call, for every supported type and both the small-K and large-K kernel selection.  A
// synthetic test, no model.  The end-to-end GLM run also exercises Q6_K through this path, but a checked
// mismatch here is what makes a future refactor safe; the first version of the kernel passed the Q8_0-only
// MLA selftest and still produced wrong GLM tokens because the Q6_K activation stride was `blocks_per_row`
// instead of `n_in/32`.
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

int g_failures = 0;

#define CHECK_HEADS(cond, ...)                                                       \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::printf("  FAIL: " __VA_ARGS__);                                     \
            std::printf("\n");                                                       \
            ++g_failures;                                                            \
        } else {                                                                     \
            std::printf("  ok:   " __VA_ARGS__);                                     \
            std::printf("\n");                                                       \
        }                                                                            \
    } while (0)

bool hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::printf("  FAIL: %s: %s\n", what, hipGetErrorString(e));
        ++g_failures;
        return false;
    }
    return true;
}

// One type's shape: the two MLA projections as GLM stores them.  `n_in` is the contraction (contiguous) width
// and `n_out` the row count, in GGUF order, exactly as `mla_layer` calls `native_mmvq`.
struct Shape {
    int type;
    const char* name;
    int n_in;
    int n_out;
};

bool check_shape(const Shape& sh, int n_head) {
    const std::size_t w_head = strata::kernels::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
    const std::size_t w_bytes = w_head * (std::size_t) n_head;
    const std::size_t q8_blocks = (std::size_t) (sh.n_in / 32);
    const std::size_t q8_bytes = strata::kernels::native_q8_1_bytes(sh.n_in) * (std::size_t) n_head;

    // Weights: arbitrary bytes are a valid block stream (the decode is fixed-point), so a fixed seed is enough
    // and the comparison is heads-vs-loop on the SAME bytes.
    std::mt19937 rng(0xC0FFEEu ^ (unsigned) sh.type);
    std::vector<uint8_t> h_w(w_bytes);
    for (uint8_t& b : h_w) b = (uint8_t) (rng() & 0xff);
    std::vector<float> h_x((std::size_t) n_head * sh.n_in);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (float& v : h_x) v = dist(rng);

    uint8_t* d_w = nullptr;
    uint8_t* d_q8 = nullptr;
    float* d_x = nullptr;
    float* d_heads = nullptr;
    float* d_loop = nullptr;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&d_w), w_bytes), "malloc w")) return false;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&d_q8), q8_bytes), "malloc q8")) return false;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&d_x), (std::size_t) n_head * sh.n_in * 4), "malloc x"))
        return false;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&d_heads), (std::size_t) n_head * sh.n_out * 4), "malloc heads"))
        return false;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&d_loop), (std::size_t) n_head * sh.n_out * 4), "malloc loop"))
        return false;

    hipStream_t stream = nullptr;
    if (!hip_ok(hipStreamCreate(&stream), "stream")) return false;
    if (!hip_ok(hipMemcpy(d_w, h_w.data(), w_bytes, hipMemcpyHostToDevice), "copy w")) return false;
    if (!hip_ok(hipMemcpy(d_x, h_x.data(), (std::size_t) n_head * sh.n_in * 4, hipMemcpyHostToDevice), "copy x"))
        return false;
    // The activation is ONE contiguous run of `n_head * n_in` values (exactly `mla_layer`'s `b.act_q8`): one
    // quantize fills every head's Q8_1 blocks in order, head h at block h*(n_in/32).
    strata::kernels::native_quantize_q8_1(d_x, d_q8, (int) ((std::size_t) n_head * sh.n_in), 1, stream);

    strata::kernels::native_mmvq_heads(sh.type, d_w, d_q8, d_heads, sh.n_in, sh.n_out, n_head, stream);
    for (int h = 0; h < n_head; ++h)
        strata::kernels::native_mmvq(sh.type, d_w + (std::size_t) h * w_head,
                                     d_q8 + (std::size_t) h * q8_blocks * 36, d_loop + (std::size_t) h * sh.n_out,
                                     sh.n_in, sh.n_out, 1, stream);
    if (!hip_ok(hipStreamSynchronize(stream), "sync")) return false;

    std::vector<float> heads((std::size_t) n_head * sh.n_out), loop((std::size_t) n_head * sh.n_out);
    if (!hip_ok(hipMemcpy(heads.data(), d_heads, heads.size() * 4, hipMemcpyDeviceToHost), "read heads")) return false;
    if (!hip_ok(hipMemcpy(loop.data(), d_loop, loop.size() * 4, hipMemcpyDeviceToHost), "read loop")) return false;

    // BITWISE, not numeric: random weight bytes can contain a NaN `d` scale, and a NaN must compare equal to a
    // NaN when what is under test is whether the two kernels compute the SAME bits.  `!=` would flag every
    // NaN-as-NaN pair, and `fabs(NaN - NaN) > worst` is always false.
    int wrong = 0;
    for (std::size_t i = 0; i < heads.size(); ++i) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &heads[i], 4);
        std::memcpy(&b, &loop[i], 4);
        if (a != b) {
            ++wrong;
            if (wrong == 1)
                std::printf("  first mismatch at [%zu]: heads %08x loop %08x\n", i, a, b);
        }
    }
    CHECK_HEADS(wrong == 0, "%s (%d) n_head %d [%d -> %d]: %d of %zu float bits differ",
                sh.name, sh.type, n_head, sh.n_in, sh.n_out, wrong, heads.size());

    hipStreamDestroy(stream);
    hipFree(d_w);
    hipFree(d_q8);
    hipFree(d_x);
    hipFree(d_heads);
    hipFree(d_loop);
    return wrong == 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) continue;
        std::printf("usage: %s [--selftest]\n", argv[0]);
        return 2;
    }
    std::printf("native_mmvq_heads_parity: head-batched MMVQ vs the per-head loop, bitwise\n");
    // GLM's real shapes: wk_b Q8_0 [192 -> 512], wv_b Q6_K [512 -> 256], n_head 20.
    const Shape shapes[] = {
        {8, "Q8_0 wk_b", 192, 512},
        {14, "Q6_K wv_b", 512, 256},
        // A large-K Q6_K case (n_in / 256 >= WARPS * WARP / 32 = 4) to exercise the other kernel selection.
        {14, "Q6_K large-K", 2048, 128},
    };
    for (const Shape& sh : shapes) {
        check_shape(sh, 20);
    }
    if (g_failures == 0) std::printf("native_mmvq_heads_parity: PASS\n");
    else std::printf("native_mmvq_heads_parity: %d FAILING\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
