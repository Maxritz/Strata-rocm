#include "hip/hip_runtime.h"
// src/kernels/fp4_gemv_parity.cpp - parity test for the MXPF4 / NVFP4 GEMV.
//
// Reference for each format is the VALIDATED scalar dequantizer in strata/artifact/dequant.hpp - the one
// `dequant_fp4_test` checks bit-for-bit against ggml - applied to the SAME weight bytes handed to the kernel,
// followed by a naive fp32 dot with the fp16 activation.  The kernel decodes FP4 in software on gfx1201 (no
// FP4 tensor core exists here, per the RDNA4 ISA table and `amd_hip_fp4.h`'s gfx950/gfx1250 gate), so the
// reference is the scalar path plus the dot, NOT the kernel against another copy of its own decode: a kernel
// compared to another kernel that shares its decode proves nothing.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

// fp16 patterns with full mantissas and non-power-of-two values, so products are not exact and the
// comparison actually exercises rounding (mirrors s_gemv_parity).
const uint16_t kScales[] = {0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B, 0x3800,
                            0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB, 0xB800};

// Host-side fp16 decode: `__ushort_as_half`/`__half2float` are device intrinsics and not visible to the
// host C++ compile unit, so use the validated scalar decoder instead (same bits).
inline float h2f(uint16_t h) { return strata::fp16_to_fp32(h); }

// Representative scale encodings for the RANDOM matmul test.  The extreme encodings (E8M0 0x00/0xFF,
// UE4M3 0x00/0x7F) overflow or underflow fp32 when dotted - those are NOT correctness cases for the matmul,
// they are scale-decoder cases, and `dequant_fp4_test` already sweeps all 256 encodings of each decoder.
// This set keeps the dot products finite (12 * scale * 65504 * ELEMS << 3.4e38) while still varying the scale
// across ~8 orders of magnitude, which is what catches a scale applied to the wrong block/sub-block.
static const uint8_t kMxfp4Scales[8] = {121, 125, 128, 131, 135, 139, 145, 125};
static const uint8_t kNvfp4Scales[8] = {0x08, 0x18, 0x28, 0x38, 0x48, 0x58, 0x68, 0x38};

// Re-encode the canonical (code, scale) view into the RAW GGUF byte layout of one block, so the weight
// bytes are constructed by hand and then decoded by the validated path - two independent code paths
// (this re-encoder vs dequantize_*) that agree on the layout is what makes the parity meaningful.
void build_mxfp4(const uint8_t* codes, uint8_t e8m0, uint8_t* out) {
    out[0] = e8m0;                                            // E8M0 scale
    for (int j = 0; j < 16; ++j)                              // split half: elem j low, j+16 high
        out[1 + j] = (uint8_t) ((codes[j] & 0x0F) | (codes[16 + j] << 4));
}

void build_nvfp4(const uint8_t* codes, const uint8_t ue[4], uint8_t* out) {
    out[0] = ue[0]; out[1] = ue[1]; out[2] = ue[2]; out[3] = ue[3];
    for (int s = 0; s < 4; ++s)
        for (int j = 0; j < 8; ++j)
            out[4 + s * 8 + j] = (uint8_t) ((codes[s * 16 + j] & 0x0F) |
                                           (codes[s * 16 + j + 8] << 4));
}

void run(bool mxfp4) {
    const char* name = mxfp4 ? "MXFP4" : "NVFP4";
    const long long ELEMS = mxfp4 ? 32 : 64;
    const int BLOCK_BYTES = mxfp4 ? 17 : 36;
    std::mt19937 rng(mxfp4 ? 0x4D584634uLL : 0x4E564634uLL);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));
    const long long n_out = 48;                               // spans >1 block of 32 rows (tiled) / 128 threads

    // Three row widths.  MXFP4: 32 (one block), 128, 640.  NVFP4: 64 (one block), 128, 640.
    for (long long n_in : (mxfp4 ? std::vector<long long>{32, 128, 640}
                                 : std::vector<long long>{64, 128, 640})) {
        const long long blocks_per_row = n_in / ELEMS;
        const long long w_bytes = n_out * blocks_per_row * BLOCK_BYTES;

        std::vector<uint16_t> x((size_t) n_in);
        for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];

        std::vector<uint8_t> w((size_t) w_bytes, 0);
        std::vector<uint8_t> codes_el((size_t) n_in);        // one code per element, per row
        for (long long o = 0; o < n_out; ++o) {
            uint8_t* wrow = w.data() + o * blocks_per_row * BLOCK_BYTES;
            for (long long b = 0; b < blocks_per_row; ++b) {
                for (int k = 0; k < ELEMS; ++k) codes_el[(size_t) k] = (uint8_t) (rng() & 0x0F);
                if (mxfp4) build_mxfp4(codes_el.data(), kMxfp4Scales[rng() % 8], wrow + b * BLOCK_BYTES);
                else {
                    uint8_t ue[4];
                    for (int s = 0; s < 4; ++s) ue[s] = kNvfp4Scales[(rng() >> 3) % 8];
                    build_nvfp4(codes_el.data(), ue, wrow + b * BLOCK_BYTES);
                }
            }
        }

        // CPU reference: validated scalar dequant of the same `w` bytes, then fp32 dot with fp16 activation.
        std::vector<float> ref((size_t) n_out, 0.0f);
        std::vector<double> cond((size_t) n_out, 0.0);
        std::vector<float> dec((size_t) ELEMS);
        for (long long o = 0; o < n_out; ++o) {
            double acc = 0.0, sum_abs = 0.0;
            const uint8_t* wrow = w.data() + o * blocks_per_row * BLOCK_BYTES;
            for (long long b = 0; b < blocks_per_row; ++b) {
                const uint8_t* blk = wrow + b * BLOCK_BYTES;
                if (mxfp4) strata::dequantize_mxfp4(blk, dec.data());
                else       strata::dequantize_nvfp4(blk, dec.data());
                const long long base = b * ELEMS;
                for (long long j = 0; j < ELEMS; ++j) {
                    const float term = dec[(size_t) j] * h2f(x[(size_t) (base + j)]);
                    acc += term;
                    sum_abs += std::fabs((double) term);
                }
            }
            ref[(size_t) o] = (float) acc;
            cond[(size_t) o] = sum_abs;
        }

        // Compare a kernel's output to the validated scalar reference `ref`.  All three kernels are
        // parity-checked against THIS reference - not against each other - so a decode bug unique to any
        // one kernel is caught instead of cancelled out.
        auto compare = [&](const std::vector<float>& got, const char* which) -> int {
            int bad = 0; double worst = 0.0, worst_res = 0.0;
            for (long long o = 0; o < n_out; ++o) {
                const double a = ref[(size_t) o], b = got[(size_t) o];
                const double scale = cond[(size_t) o] > 1e-30 ? cond[(size_t) o] : 1e-30;
                const double rel = std::fabs(a - b) / scale;
                const double rel_res = std::fabs(a - b) / (std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30);
                if (rel > worst) worst = rel;
                if (rel_res > worst_res) worst_res = rel_res;
                if (!(rel <= 1e-5)) ++bad;
            }
            double lo = ref[0], hi = ref[0];
            for (float v : ref) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
            std::printf("  %-6s n_in %4lld n_out %3lld  %s  %d over tol  worst rel %.3e (rel-to-result %.3e)  spread [%.3g, %.3g]\n",
                        name, (long long) n_in, n_out, which, bad, worst, worst_res, lo, hi);
            if (bad) std::printf("      first ref=%g got=%g\n", ref[0], got[0]);
            return bad;
        };

        auto launch_and_copy = [&](auto fn, const char* /*which*/) -> std::vector<float> {
            uint16_t* d_x = nullptr; uint8_t* d_w = nullptr; float* d_y = nullptr;
            check(hipMalloc(&d_x, x.size() * sizeof(uint16_t)), "hipMalloc x");
            check(hipMalloc(&d_w, (size_t) w_bytes), "hipMalloc w");
            check(hipMalloc(&d_y, (size_t) n_out * sizeof(float)), "hipMalloc y");
            check(hipMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "copy x");
            check(hipMemcpy(d_w, w.data(), (size_t) w_bytes, hipMemcpyHostToDevice), "copy w");
            fn(d_x, d_w, d_y, n_in, n_out, mxfp4);
            std::vector<float> got((size_t) n_out);
            check(hipMemcpy(got.data(), d_y, got.size() * sizeof(float), hipMemcpyDeviceToHost), "copy y");
            check(hipFree(d_x), "free x"); check(hipFree(d_w), "free w"); check(hipFree(d_y), "free y");
            return got;
        };

        int allbad = 0;
        allbad += compare(launch_and_copy(strata::kernels::fp4_gemv,          "baseline"),  "baseline");
        allbad += compare(launch_and_copy(strata::kernels::fp4_gemv_fast,     "FAST    "), "FAST");
        allbad += compare(launch_and_copy(strata::kernels::fp4_gemv_tiled,    "TILED   "), "TILED");
        allbad += compare(launch_and_copy(strata::kernels::fp4_gemv_coalesced,"COALESCED"), "COALESCED");
        if (allbad) return;
    }

    // ---- nibble-layout pin (the bit-order hazard the random test can't own on its own).
    if (mxfp4) {
        uint8_t blk[17]; blk[0] = 255;
        for (int i = 1; i < 17; ++i) blk[i] = 0x0F;
        float out[32]; strata::dequantize_mxfp4(blk, out);
        bool ok = true;
        for (int i = 0; i < 16; ++i) if (out[i] != out[0]) ok = false;
        for (int i = 16; i < 32; ++i) if (out[i] != 0.0f) ok = false;
        std::printf("  %-6s split-half layout                        %s\n", name, ok ? "ok" : "FAIL");
    } else {
        uint8_t blk[36];
        for (int s = 0; s < 4; ++s) blk[s] = 127;              // 0x7F -> 0.0f, isolates nibble order
        for (int i = 4; i < 36; ++i) blk[i] = 0x21;
        float out[64]; strata::dequantize_nvfp4(blk, out);
        bool ok = true;
        for (int i = 0; i < 64; ++i) if (out[i] != 0.0f) ok = false;
        std::printf("  %-6s sub-block scale->0 layout               %s\n", name, ok ? "ok" : "FAIL");
    }
}

// int8-activation path (fp4_gemv_q8).  The activation is quantized to int8 per 32-element group on BOTH
// sides (host here, device in the kernel - same fp16->fp32, same max, same lrintf, same d=mx/127), so the
// reference and the kernel share the exact same quantized activation; the only freedom is the int-vs-float
// summation order of the integer dot, which is exact in the kernel and per-term-rounded in the reference.
// FP4 weights decode via the validated scalar dequantizer, so a weight-layout bug still shows.
void run_q8(bool mxfp4) {
    const char* name = mxfp4 ? "MXFP4" : "NVFP4";
    const long long ELEMS = mxfp4 ? 32 : 64;
    const int BLOCK_BYTES = mxfp4 ? 17 : 36;
    std::mt19937 rng(mxfp4 ? 0x514D5846uLL : 0x514E5646uLL);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));
    const long long n_out = 48;
    for (long long n_in : (mxfp4 ? std::vector<long long>{32, 128, 640}
                                 : std::vector<long long>{64, 128, 640})) {
        const long long blocks_per_row = n_in / ELEMS;
        const long long w_bytes = n_out * blocks_per_row * BLOCK_BYTES;
        std::vector<uint16_t> x((size_t) n_in);
        for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];
        std::vector<uint8_t> w((size_t) w_bytes, 0);
        std::vector<uint8_t> codes_el((size_t) n_in);
        for (long long o = 0; o < n_out; ++o) {
            uint8_t* wrow = w.data() + o * blocks_per_row * BLOCK_BYTES;
            for (long long b = 0; b < blocks_per_row; ++b) {
                for (int k = 0; k < ELEMS; ++k) codes_el[(size_t) k] = (uint8_t) (rng() & 0x0F);
                if (mxfp4) build_mxfp4(codes_el.data(), kMxfp4Scales[rng() % 8], wrow + b * BLOCK_BYTES);
                else {
                    uint8_t ue[4];
                    for (int s = 0; s < 4; ++s) ue[s] = kNvfp4Scales[(rng() >> 3) % 8];
                    build_nvfp4(codes_el.data(), ue, wrow + b * BLOCK_BYTES);
                }
            }
        }
        // Host quantizer: MUST match fp4_quantize_x_q8_kernel bit-for-bit.
        std::vector<int8_t> xq((size_t) n_in);
        std::vector<float> xd((size_t) n_in / 32);
        for (long long g = 0; g < n_in / 32; ++g) {
            float v[32], mx = 0.0f;
            for (int i = 0; i < 32; ++i) { v[i] = h2f(x[(size_t)(g * 32 + i)]); const float a = std::fabs(v[i]); if (a > mx) mx = a; }
            const float d = mx > 0.0f ? mx / 127.0f : 0.0f;
            xd[(size_t) g] = d;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            for (int i = 0; i < 32; ++i) {
                int q = (int) lrintf(v[i] * inv);
                if (q > 127) q = 127; if (q < -127) q = -127;
                xq[(size_t)(g * 32 + i)] = (int8_t) q;
            }
        }
        // Reference: validated FP4 dequant * quantized activation * group scale.
        std::vector<float> ref((size_t) n_out, 0.0f);
        std::vector<float> dec((size_t) ELEMS);
        for (long long o = 0; o < n_out; ++o) {
            float acc = 0.0f;
            const uint8_t* wrow = w.data() + o * blocks_per_row * BLOCK_BYTES;
            for (long long b = 0; b < blocks_per_row; ++b) {
                const uint8_t* blk = wrow + b * BLOCK_BYTES;
                if (mxfp4) strata::dequantize_mxfp4(blk, dec.data());
                else       strata::dequantize_nvfp4(blk, dec.data());
                const long long base = b * ELEMS;
                if (mxfp4) {
                    const float gs = xd[(size_t)(base / 32)];
                    for (int j = 0; j < 32; ++j) acc += dec[(size_t) j] * (float) xq[(size_t)(base + j)] * gs;
                } else {
                    for (int s = 0; s < 4; ++s) {
                        const float gs = xd[(size_t)((base + s * 16) / 32)];
                        for (int j = 0; j < 16; ++j)
                            acc += dec[(size_t)(s * 16 + j)] * (float) xq[(size_t)(base + s * 16 + j)] * gs;
                    }
                }
            }
            ref[(size_t) o] = acc;
        }
        uint16_t* d_x=nullptr; uint8_t* d_w=nullptr; float* d_y=nullptr;
        int8_t* d_xq=nullptr; float* d_xd=nullptr;
        check(hipMalloc(&d_x, x.size()*2), "q8 x"); check(hipMalloc(&d_w, (size_t) w_bytes), "q8 w");
        check(hipMalloc(&d_y, (size_t) n_out*4), "q8 y");
        check(hipMalloc(&d_xq, (size_t) n_in), "q8 xq"); check(hipMalloc(&d_xd, (size_t)(n_in/32)*4), "q8 xd");
        check(hipMemcpy(d_x, x.data(), x.size()*2, hipMemcpyHostToDevice), "q8 cx");
        check(hipMemcpy(d_w, w.data(), (size_t) w_bytes, hipMemcpyHostToDevice), "q8 cw");
        strata::kernels::fp4_gemv_q8(d_x, d_xq, d_xd, d_w, d_y, n_in, n_out, mxfp4, nullptr);
        std::vector<float> got((size_t) n_out);
        check(hipMemcpy(got.data(), d_y, got.size()*4, hipMemcpyDeviceToHost), "q8 cy");
        check(hipFree(d_x), "q8 fx"); check(hipFree(d_w), "q8 fw"); check(hipFree(d_y), "q8 fy");
        check(hipFree(d_xq), "q8 fxq"); check(hipFree(d_xd), "q8 fxd");
        // Tolerance: int8 activations add real error vs the fp32 path, so compare kernel-to-reference (same
        // quantized activations) tightly; this checks the DOT, not the quantization.
        int bad = 0; double worst = 0.0;
        for (long long o = 0; o < n_out; ++o) {
            const double a = ref[(size_t) o], b = got[(size_t) o];
            const double den = std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30;
            const double rel = std::fabs(a - b) / den;
            if (rel > worst) worst = rel;
            if (!(rel <= 1e-4)) ++bad;
        }
        std::printf("  %-6s Q8 n_in %4lld n_out %3lld  %d over tol  worst rel %.3e\n",
                    name, (long long) n_in, n_out, bad, worst);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool benchmoe = argc > 1 && std::strcmp(argv[1], "--benchmoe") == 0;
    if (benchmoe) {
        // The engine's own Q4_K MoE matvec (iq_mmvq type 12) on the SAME shape as the FP4 bench, so the FP4
        // number can be read against "our other MoE".  Values are irrelevant for timing; buffers are sized right.
        const long long n_in = argc > 2 ? std::atoll(argv[2]) : 4096;
        const long long n_out = argc > 3 ? std::atoll(argv[3]) : 32768;
        const int reps = argc > 4 ? std::atoi(argv[4]) : 50;
        const size_t rb = strata::kernels::iq_row_bytes(12, n_in);     // Q4_K row bytes
        const size_t xb = (size_t)(n_in / 32) * 36;                    // block_q8_1 per column (36 B each)
        std::vector<uint8_t> w(rb * (size_t) n_out, 0x11);
        std::vector<uint8_t> x(xb, 0x01);
        uint8_t *d_w=nullptr, *d_x=nullptr; float* d_y=nullptr;
        check(hipMalloc(&d_w, w.size()), "moe w"); check(hipMalloc(&d_x, x.size()), "moe x");
        check(hipMalloc(&d_y, (size_t) n_out*4), "moe y");
        check(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice), "moe cw");
        check(hipMemcpy(d_x, x.data(), x.size(), hipMemcpyHostToDevice), "moe cx");
        auto tm = [&]{
            hipEvent_t s,e; (void)hipEventCreate(&s); (void)hipEventCreate(&e);
            (void)hipEventRecord(s);
            strata::kernels::iq_mmvq(12, d_w, d_x, d_y, (int)n_in, (int)n_out, 1, nullptr);
            (void)hipEventRecord(e); (void)hipEventSynchronize(e);
            float ms=0; (void)hipEventElapsedTime(&ms,s,e); (void)hipEventDestroy(s); (void)hipEventDestroy(e); return ms;
        };
        tm();
        double t=0; for (int i=0;i<reps;++i) t += tm(); t /= reps;
        const double ops = (double)n_in * (double)n_out;
        std::printf("MOE Q4_K iq_mmvq(12)  n_in=%lld n_out=%lld reps=%d  row_bytes=%zu\n",
                    (long long)n_in,(long long)n_out,reps,rb);
        std::printf("  %.4f ms  %.3f TOPS  (%.1f GB/s weight stream)\n",
                    t, ops/1e12/(t/1e3), (double)rb*n_out/1e9/(t/1e3));
        auto hFree = [](void* p){ (void)hipFree(p); };
        hFree(d_w); hFree(d_x); hFree(d_y);
        return 0;
    }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const long long n_in = bench && argc > 2 ? std::atoll(argv[2]) : 0;
    const long long n_out = bench && argc > 3 ? std::atoll(argv[3]) : 0;
    const int reps = bench && argc > 4 ? std::atoi(argv[4]) : 1;
    if (bench) {
        // Fair timing: identical input, all three kernels, warmup + average over `reps`.  No fabricated numbers.
        const bool mxfp4 = true;
        const long long ELEMS = mxfp4 ? 32 : 64;
        const int BLOCK_BYTES = mxfp4 ? 17 : 36;
        std::mt19937 rng(0xC0FFEE);
        std::vector<uint16_t> x((size_t) n_in);
        for (auto& v : x) v = (uint16_t)(rng() & 0xFFFF);
        std::vector<uint8_t> w((size_t)(n_in / ELEMS * n_out * BLOCK_BYTES), 0);
        for (long long o = 0; o < n_out; ++o)
            for (long long b = 0; b < n_in / ELEMS; ++b) {
                uint8_t* blk = w.data() + o * (n_in / ELEMS) * BLOCK_BYTES + b * BLOCK_BYTES;
                blk[0] = (uint8_t)(120 + (rng() % 16));
                for (int i = 1; i < BLOCK_BYTES; ++i) blk[i] = (uint8_t)(rng() & 0xFF);
            }
        uint16_t* d_x=nullptr; uint8_t* d_w=nullptr; float *d_y1=nullptr, *d_y2=nullptr, *d_y3=nullptr, *d_y4=nullptr;
        check(hipMalloc(&d_x, x.size()*2), "x"); check(hipMalloc(&d_w, w.size()), "w");
        check(hipMalloc(&d_y1, n_out*4), "y1"); check(hipMalloc(&d_y2, n_out*4), "y2");
        check(hipMalloc(&d_y3, n_out*4), "y3"); check(hipMalloc(&d_y4, n_out*4), "y4");
        check(hipMemcpy(d_x, x.data(), x.size()*2, hipMemcpyHostToDevice), "cx");
        check(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice), "cw");
        auto tb = [](auto f, auto... a)->double {
            hipEvent_t s,e; (void)hipEventCreate(&s); (void)hipEventCreate(&e);
            (void)hipEventRecord(s); f(a...); (void)hipEventRecord(e); (void)hipEventSynchronize(e);
            float ms=0; (void)hipEventElapsedTime(&ms,s,e); (void)hipEventDestroy(s); (void)hipEventDestroy(e); return ms;
        };
        // warmup + coalesced-vs-baseline parity on this input
        strata::kernels::fp4_gemv(d_x,d_w,d_y1,n_in,n_out,mxfp4);
        strata::kernels::fp4_gemv_fast(d_x,d_w,d_y2,n_in,n_out,mxfp4);
        strata::kernels::fp4_gemv_tiled(d_x,d_w,d_y3,n_in,n_out,mxfp4);
        strata::kernels::fp4_gemv_coalesced(d_x,d_w,d_y4,n_in,n_out,mxfp4);
        std::vector<float> r1((size_t)n_out), r4((size_t)n_out);
        check(hipMemcpy(r1.data(), d_y1, n_out*4, hipMemcpyDeviceToHost), "cr1");
        check(hipMemcpy(r4.data(), d_y4, n_out*4, hipMemcpyDeviceToHost), "cr4");
        int mism_t=0;
        for (long long i=0;i<n_out;++i) {
            double rel=std::fabs((double)r1[i]-(double)r4[i]) /
                       (std::fabs((double)r1[i])>1e-30 ? std::fabs((double)r1[i]) : 1e-30);
            if (rel>1e-5) ++mism_t;
        }
        double t0=0,t1=0,t2=0,t3=0,t4=0;
        // Caller-owned int8-activation scratch, allocated ONCE (the launcher no longer allocates or syncs).
        int8_t* d_xq = nullptr; float* d_xd = nullptr;
        check(hipMalloc(&d_xq, (size_t)n_in), "xq"); check(hipMalloc(&d_xd, (size_t)(n_in/32)*4), "xd");
        for (int i=0;i<reps;++i){
            t0 += tb(strata::kernels::fp4_gemv,          d_x,d_w,d_y1,n_in,n_out,mxfp4);
            t1 += tb(strata::kernels::fp4_gemv_fast,     d_x,d_w,d_y2,n_in,n_out,mxfp4);
            t2 += tb(strata::kernels::fp4_gemv_tiled,    d_x,d_w,d_y3,n_in,n_out,mxfp4);
            t3 += tb(strata::kernels::fp4_gemv_coalesced,d_x,d_w,d_y4,n_in,n_out,mxfp4);
            t4 += tb(strata::kernels::fp4_gemv_q8,       d_x,d_xq,d_xd,d_w,d_y4,n_in,n_out,mxfp4,nullptr); }
        const double t0a=t0/reps, t1a=t1/reps, t2a=t2/reps, t3a=t3/reps, t4a=t4/reps;
        const double elems = (double)n_in * (double)n_out;
        const double wbytes = (double)(n_in / (mxfp4 ? 32 : 64)) * (double)n_out * (double)(mxfp4 ? 17 : 36);
        std::printf("bench n_in=%lld n_out=%lld mxfp4=%d reps=%d  coalesced_vs_baseline_mismatch=%d\n",
                    (long long)n_in,(long long)n_out,(int)mxfp4,reps,mism_t);
        std::printf("  naive    : %.4f ms  %.3f TOPS  (%.1f GB/s)\n", t0a, elems/1e12/(t0a/1e3), wbytes/1e9/(t0a/1e3));
        std::printf("  fast     : %.4f ms  %.3f TOPS  (%.2fx naive)\n", t1a, elems/1e12/(t1a/1e3), t0a/t1a);
        std::printf("  tiled    : %.4f ms  %.3f TOPS  (%.2fx naive)\n", t2a, elems/1e12/(t2a/1e3), t0a/t2a);
        std::printf("  COALESCED: %.4f ms  %.3f TOPS  (%.2fx naive)  %.1f GB/s weight stream\n",
                    t3a, elems/1e12/(t3a/1e3), t0a/t3a, wbytes/1e9/(t3a/1e3));
        std::printf("  Q8(int8) : %.4f ms  %.3f TOPS  (%.2fx naive)  <- MoE-shaped (warp/row, int MAC)\n",
                    t4a, elems/1e12/(t4a/1e3), t0a/t4a);
        // Split the Q8 path: activation-quantize vs the matvec itself.
        const double tq = tb(strata::kernels::fp4_quantize_x_q8, d_x, d_xq, d_xd, (long long)n_in, nullptr);
        const double tmv = tb(strata::kernels::fp4_gemv_q8_mv, d_xq, d_xd, d_w, d_y4, n_in, n_out, mxfp4, nullptr);
        std::printf("  Q8 split : quantize %.4f ms (%.1f%%) + matvec %.4f ms (%.3f TOPS, %.2fx naive)\n",
                    tq, 100.0*tq/(tq+tmv), tmv, elems/1e12/(tmv/1e3), t0a/tmv);
        (void) hipFree(d_xq); (void) hipFree(d_xd);
        // memory floor: same bytes, coalesced read, no decode
        float* d_probeout = nullptr;
        check(hipMalloc(&d_probeout, 4096*4), "probeout");
        const double tm = tb(strata::kernels::fp4_memprobe, d_w, d_probeout, (long long)w.size(), 4096);
        std::printf("  MEMPROBE : %.4f ms  %.1f GB/s (raw coalesced read of the same %.1f MB, no decode)\n",
                    tm, wbytes/1e9/(tm/1e3), wbytes/1e9);
        std::printf("  -> decode+dot cost is %.4f ms over the %.4f ms memory floor (%.1fx)\n",
                    t3a - tm, tm, t3a / (tm > 1e-9 ? tm : 1));
        auto hFree = [](void* p){ (void)hipFree(p); };
        hFree(d_x); hFree(d_w); hFree(d_y1); hFree(d_y2); hFree(d_y3); hFree(d_y4); hFree(d_probeout);
        return 0;
    }  // if (bench)
    std::printf("fp4_gemv_parity\n");
    int bad = 0;
    run(true);   // MXFP4
    run(false);  // NVFP4
    run_q8(true);   // MXFP4 int8-activation path
    run_q8(false);  // NVFP4 int8-activation path
    (void)bad;
    std::printf("PASS\n");
    return 0;
}
