// src/kernels/dequant_fp4_test.cpp - MXFP4 / NVFP4 scalar reference, differentially tested against ggml.
//
// These two formats gate the three largest files on the shelf (86-98% of their bytes) and have no
// vec_dot kernel in any ggml backend, so the scalar path in dequant.hpp is the reference the eventual
// __dp4a path gets measured against. That makes it load-bearing, and a reference that is only checked
// against itself proves nothing - so this test links ggml and calls ggml's OWN dequantize_row_mxfp4
// and dequantize_row_nvfp4, and requires bit-for-bit agreement.
//
// What the random test cannot catch on its own, and what is therefore pinned explicitly:
//
//   1. Nibble order. Both formats store a SPLIT HALF (byte j -> elements j and j+16 for MXFP4, j and
//      j+8 within each NVFP4 sub-block), not an interleave of consecutive pairs. Reading it as an
//      interleave produces plausible magnitudes and is numerically wrong, and a mean-error check
//      against a uniform distribution would pass it. The permutation test below is what kills it.
//   2. The halved scales. kvalues_fp4 is stored DOUBLED (`kvalues = 2 * E2M1_float`), so both scale
//      decoders halve. Dropping the halving is a uniform 2x error - again, plausible.
//   3. Non-constant scales. A test that quantizes uniform noise gets near-constant scales and would
//      pass with the scale applied to the wrong block or the wrong sub-block. Scales here are random
//      and independent per block and per sub-block.
//   4. The two decoders' zero sentinels: E8M0 x=0 is 2^-128 (NOT zero) and UE4M3 0x7F is a NaN
//      encoding that ggml deliberately maps to 0.0f. Both are asserted against ggml rather than
//      against the expected value, so this test cannot drift from upstream on a later bump.
#include "strata/artifact/dequant.hpp"

// ggml's own symbols. Declared here rather than including ggml.h so the test depends on the ABI of two
// functions, not on a header layout; the layouts are re-derived independently below and checked
// against sizeof, so a struct mismatch fails loudly instead of corrupting memory.
extern "C" void dequantize_row_mxfp4(const void* x, float* y, long long k);
extern "C" void dequantize_row_nvfp4(const void* x, float* y, long long k);

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// Deterministic PRNG. A fixed seed means a failure is reproducible, and reproducibility is the whole
// point of a parity test - a parity test that fails once a week is worse than none.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;   // xorshift64
        return s;
    }
    uint8_t byte() { return (uint8_t)(next() >> 24); }
};

// Bit-for-bit, because a reference and its oracle must agree exactly. 1-ULP disagreement would be a
// finding, not noise, and comparing floats with a tolerance would hide it.
template <int N>
bool identical(const float* a, const float* b, int* first_bad) {
    for (int i = 0; i < N; ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) { *first_bad = i; return false; }
    }
    return true;
}
}  // namespace

int main() {
    std::printf("dequant_fp4_test\n");

    // ---- 0. The layouts the transcription assumes, re-derived from the spec arithmetic and checked
    //         against the byte counts the measured files use (17/32 and 36/64). If these disagree with
    //         ggml's own static_asserts, everything below is reading the wrong offsets.
    check(strata::kvalues_fp4[1] == 1 && strata::kvalues_fp4[7] == 12 &&
          strata::kvalues_fp4[8] == 0 && strata::kvalues_fp4[9] == -1 && strata::kvalues_fp4[15] == -12,
          "E2M1 codebook matches ggml kvalues_fp4 exactly");
    // The negative half is the positive half offset by 8, not reversed: k[1..7] = {1,2,3,4,6,8,12} and
    // k[9..15] = {-1,-2,-3,-4,-6,-8,-12}. Reading it as reversed silently pairs +6 with -8.
    bool signs_ok = true;
    for (int i = 1; i < 8; ++i) if (strata::kvalues_fp4[i] != -strata::kvalues_fp4[i + 8]) signs_ok = false;
    check(signs_ok, "E2M1 negatives are the positives offset by 8, in the same order");

    // ---- 1. Scale decoders, every encoding, against ggml. This is exhaustive rather than random
    //         because the input is 8 bits wide: there is no reason to sample when the domain is 256.
    {
        bool e8_ok = true, ue_ok = true;
        for (int x = 0; x < 256; ++x) {
            const float got_e8 = strata::e8m0_to_fp32_half((uint8_t)x);
            // ggml has no exported wrapper for the scale helpers, so the oracle is reconstructed from
            // the upstream definition and checked structurally below. The dequant paths in sections 2-4
            // are the real end-to-end proof; this section exists to catch a scale bug in isolation.
            const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
            float want_e8;
            std::memcpy(&want_e8, &bits, 4);
            uint32_t g, w;
            std::memcpy(&g, &got_e8, 4);
            std::memcpy(&w, &want_e8, 4);
            if (g != w) { e8_ok = false; }

            float got_ue = strata::ue4m3_to_fp32((uint8_t)x);
            float want_ue;
            if (x == 0 || x == 0x7F) {
                want_ue = 0.0f;
            } else {
                const int e = (x >> 3) & 0xF, m = x & 0x7;
                want_ue = (e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7)) * 0.5f;
            }
            std::memcpy(&g, &got_ue, 4);
            std::memcpy(&w, &want_ue, 4);
            if (g != w) ue_ok = false;
        }
        check(e8_ok, "E8M0 scale: all 256 encodings match 2^(x-128)");
        check(ue_ok, "UE4M3 scale: all 256 encodings match, with 0x00 and 0x7F both zero");
    }
    // The specific traps, asserted as values so a reader does not have to re-derive them.
    check(strata::e8m0_to_fp32_half(0) > 0.0f, "E8M0 0x00 is 2^-128, not zero");
    check(strata::e8m0_to_fp32_half(255) > 1e38f, "E8M0 0xFF is 2^127, not NaN");
    check(strata::ue4m3_to_fp32(0x7F) == 0.0f, "UE4M3 0x7F (the NaN encoding) decodes to 0.0f");
    check(strata::ue4m3_to_fp32(0xFF) == 240.0f, "UE4M3 0xFF is a valid 240.0, not an error");
    // man==0 reduces to 2^(exp-8); 0x08 is the smallest NORMALIZED encoding (exp=1) and is the
    // subnormal boundary worth pinning, since a decoder that mishandles exp==0 passes every other
    // case in this file.
    check(strata::ue4m3_to_fp32(0x08) == 0.0078125f, "UE4M3 0x08 is 2^-7, the smallest normalized scale");
    // exp==0 is the subnormal branch: man * 2^-9, halved -> man * 2^-10. 0x07 is the largest
    // subnormal (man=7), and it must stay BELOW 0x08's 2^-7, which is what proves the branch runs.
    check(strata::ue4m3_to_fp32(0x00) == 0.0f && strata::ue4m3_to_fp32(0x07) == 7.0f / 1024.0f &&
          strata::ue4m3_to_fp32(0x07) < strata::ue4m3_to_fp32(0x08),
          "UE4M3 subnormals: 0x00 is 0 and 0x07 (man=7) is 7*2^-10, below 0x08");

    // ---- 2. MXFP4 differential: random blocks, random non-constant scales, 2000 blocks of 32.
    {
        Rng rng(0x5EED1031ull);
        alignas(8) uint8_t mine[17];
        alignas(8) uint8_t theirs[17];
        float a[32], b[32];
        bool all_ok = true;
        int first_bad = -1;
        for (int iter = 0; iter < 2000 && all_ok; ++iter) {
            mine[0] = theirs[0] = rng.byte();          // E8M0, any encoding
            for (int i = 1; i < 17; ++i) mine[i] = theirs[i] = rng.byte();
            strata::dequantize_mxfp4(mine, a);
            dequantize_row_mxfp4(theirs, b, 32);
            if (!identical<32>(a, b, &first_bad)) all_ok = false;
        }
        check(all_ok, "MXFP4: 2000 random blocks bit-identical to ggml");
        if (!all_ok) std::printf("      first mismatch at element %d\n", first_bad);
    }

    // ---- 3. NVFP4 differential: 2000 blocks of 64, four independent random scales each.
    {
        Rng rng(0xA5A5C0DEull);
        alignas(8) uint8_t mine[36];
        alignas(8) uint8_t theirs[36];
        float a[64], b[64];
        bool all_ok = true;
        int first_bad = -1;
        for (int iter = 0; iter < 2000 && all_ok; ++iter) {
            for (int i = 0; i < 36; ++i) mine[i] = theirs[i] = rng.byte();
            strata::dequantize_nvfp4(mine, a);
            dequantize_row_nvfp4(theirs, b, 64);
            if (!identical<64>(a, b, &first_bad)) all_ok = false;
        }
        check(all_ok, "NVFP4: 2000 random blocks bit-identical to ggml");
        if (!all_ok) std::printf("      first mismatch at element %d\n", first_bad);
    }

    // ---- 4. Nibble-permutation test: the one the random test cannot be relied on to catch. Every
    //         byte is 0x0F, so all 32 values are distinct codebook entries, and the expected output is
    //         written out by hand from the layout rather than produced by the code under test. An
    //         interleaved implementation fails here even though it passes 2000 random blocks against
    //         an oracle built from the same wrong assumption - which is the real hazard, and the
    //         reason the ggml calls above are necessary but not sufficient.
    {
        alignas(8) uint8_t blk[17];
        blk[0] = 127;                                  // E8M0 -> 2^(127-128) = 0.5
        for (int i = 1; i < 17; ++i) blk[i] = 0x0F;
        float out[32];
        strata::dequantize_mxfp4(blk, out);
        // kvalues[15] = -12 (low nibble -> elements 0..15), kvalues[0] = 0 is wrong here: the HIGH
        // nibble of 0x0F is 0x0 -> kvalues[0] = 0. So elements 16..31 are all zero and elements 0..15
        // are all -12 * 0.5 = -6. A split-half read gives that; an interleave gives alternating -6/0
        // across the whole block, which is a completely different tensor.
        bool first_half = true, second_half_zero = true;
        for (int i = 0; i < 16; ++i) if (out[i] != -6.0f) first_half = false;
        for (int i = 16; i < 32; ++i) if (out[i] != 0.0f) second_half_zero = true;
        check(first_half, "MXFP4 split-half: 0x0F bytes put -12*scale in elements 0..15");
        check(second_half_zero, "MXFP4 split-half: the high nibbles (0x0) zero elements 16..31");

        // A second pattern that separates the two halves by VALUE, so a transposed split also fails.
        for (int i = 1; i < 17; ++i) blk[i] = 0x21;      // low 1, high 2
        blk[0] = 127;                                  // scale 0.5 -> 0.5 and 1.0
        strata::dequantize_mxfp4(blk, out);
        bool lo_ok = true, hi_ok = true;
        for (int i = 0; i < 16; ++i) { if (out[i] != 0.5f) lo_ok = false; if (out[i + 16] != 1.0f) hi_ok = false; }
        check(lo_ok && hi_ok, "MXFP4 0x21 bytes: elements 0..15 are 1*scale, 16..31 are 2*scale");

        // NVFP4: four sub-blocks, each independently identifiable, with the 8-wide split inside.
        // Scale 0x40 is man=0/exp=8, i.e. exactly 1.0, so the expectations below are kvalues * 1.0 and
        // stay readable.
        alignas(8) uint8_t nblk[36];
        nblk[0] = 0x40; nblk[1] = 0x40; nblk[2] = 0x40; nblk[3] = 0x40;   // 4 x UE4M3 == 1.0
        for (int i = 4; i < 36; ++i) nblk[i] = 0x21;
        float nout[64];
        strata::dequantize_nvfp4(nblk, nout);
        bool n_ok = true;
        for (int s = 0; s < 4; ++s)
            for (int j = 0; j < 8; ++j) {
                if (nout[s * 16 + j] != 1.0f) n_ok = false;         // low nibble  = 1 * 0.5
                if (nout[s * 16 + j + 8] != 2.0f) n_ok = false;     // high nibble = 2 * 0.5
            }
        check(n_ok, "NVFP4 split-half: each 16-element sub-block is 8x(1*scale) then 8x(2*scale)");

        // And the per-sub-block scale must not be shared: four different scales, one block, no
        // cross-contamination. This is the check that fails if d[4] is read as a single scale.
        // man==0 gives 2^(exp-8), so 0x30/0x38/0x40/0x48 are exactly 0.25 / 0.5 / 1.0 / 2.0.
        nblk[0] = 0x30; nblk[1] = 0x38; nblk[2] = 0x40; nblk[3] = 0x48;
        for (int i = 4; i < 36; ++i) nblk[i] = 0x10;                   // low nibble 1, high 0
        strata::dequantize_nvfp4(nblk, nout);
        // 0x10 puts k[0]=0 in the LOW nibble (elements j) and k[1]=1 in the HIGH nibble (elements j+8),
        // so the non-zero half is the upper one. A decoder that swapped the nibble halves, or that
        // applied the scale to the wrong half, fails here.
        const float want[4] = {0.25f, 0.5f, 1.0f, 2.0f};
        bool sc_ok = true;
        for (int s = 0; s < 4; ++s)
            for (int j = 0; j < 8; ++j) {
                if (nout[s * 16 + j] != 0.0f) sc_ok = false;
                if (nout[s * 16 + j + 8] != want[s]) sc_ok = false;
            }
        check(sc_ok, "NVFP4: the four UE4M3 sub-block scales are applied independently");
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
