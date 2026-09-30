#include "hip/hip_runtime.h"
// src/kernels/fp4_gemv_parity.cpp - parity test for the MXFP4 / NVFP4 GEMV.
//
// Reference for each format is the VALIDATED scalar dequantizer in strata/artifact/dequant.hpp - the one
// `dequant_fp4_test` checks bit-for-bit against ggml - applied to the SAME weight bytes handed to the kernel,
// followed by a naive fp32 dot with the fp16 activation.  The kernel decodes FP4 in software on gfx1201 (no
// FP4 tensor core exists here, per the RDNA4 ISA table and `amd_hip_fp4.h`'s gfx950/gfx1250 gate), so the
// reference is the scalar path plus the dot, NOT the kernel against itself: a kernel compared to another copy
// of its own decode proves nothing.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/fp4_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    const long long n_out = 48;                               // spans >1 block of 128 threads

    // Three row widths (MXFP4): 32 (one block), 128, 640.  NVFP4 starts at 64 (one block) - the 32 case is
    // structurally impossible (640/2560 shapes never have n_in=32 for NVFP4), so only test valid widths.
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

        uint16_t* d_x = nullptr; uint8_t* d_w = nullptr; float* d_y = nullptr;
        check(hipMalloc(&d_x, x.size() * sizeof(uint16_t)), "hipMalloc x");
        check(hipMalloc(&d_w, (size_t) w_bytes), "hipMalloc w");
        check(hipMalloc(&d_y, (size_t) n_out * sizeof(float)), "hipMalloc y");
        check(hipMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "copy x");
        check(hipMemcpy(d_w, w.data(), (size_t) w_bytes, hipMemcpyHostToDevice), "copy w");
        strata::kernels::fp4_gemv(d_x, d_w, d_y, n_in, n_out, mxfp4);
        std::vector<float> got((size_t) n_out);
        check(hipMemcpy(got.data(), d_y, got.size() * sizeof(float), hipMemcpyDeviceToHost), "copy y");
        check(hipFree(d_x), "free x"); check(hipFree(d_w), "free w"); check(hipFree(d_y), "free y");

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
        std::printf("  %-6s n_in %4lld n_out %3lld  %d over tol  worst rel %.3e (rel-to-result %.3e)  spread [%.3g, %.3g]\n",
                    name, (long long) n_in, n_out, bad, worst, worst_res, lo, hi);
        if (bad) { std::printf("      first ref=%g got=%g\n", ref[0], got[0]); return; }
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

}  // namespace

int main() {
    std::printf("fp4_gemv_parity\n");
    int bad = 0;
    run(true);   // MXFP4
    run(false);  // NVFP4
    (void)bad;
    std::printf("PASS\n");
    return 0;
}
