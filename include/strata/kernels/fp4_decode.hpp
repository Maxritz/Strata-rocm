// include/strata/kernels/fp4_decode.hpp - shared FP4 decode helpers for the GPU matmul kernels.
//
// Device-side (and host-side) FP4 decoding that is bit-for-bit the same path the scalar reference in
// dequant.hpp validates against ggml.  Factored here so the naive and the fast matmul kernels cannot drift
// apart: both include this one table and these one decoder, and a decode bug shows up in BOTH parity tests
// at once instead of hiding in whichever kernel wrote its own copy first.
#pragma once

#include <cstdint>

namespace strata::kernels {

// FP4 (microscaling) E2M1 codebook, verbatim from ggml-common.h / dequant.hpp.  Stored as int8 because the
// values are signed and range -12..12 (which overflows signed 4-bit, so the FP4 codes themselves are NEVER
// fed to a 4-bit dot instruction - see docs/TODO.md P1).
__device__ __host__ inline const int8_t* fp4_codebook() {
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return kv;
}

// E8M0 scale -> fp32 = 2^(x-128).  Matches e8m0_to_fp32_half in dequant.hpp exactly.
__device__ __host__ inline float fp4_e8m0_to_fp32(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x)
                              : ((uint32_t)(x - 1) << 23);
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// UE4M3 scale -> fp32, halved because the codebook is stored doubled.  Matches ue4m3_to_fp32.
// 0x00 and 0x7F both map to 0.0f (0x7F is the NaN encoding; ggml maps it to zero).
__device__ __host__ inline float fp4_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = exp == 0 ? ldexpf((float)man, -9)
                              : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// One MXFP4 block (17 bytes: E8M0 scale + 16 bytes = 32 values), split-half nibble order.
__device__ __forceinline__ void decode_mxfp4_block(const uint8_t* __restrict__ blk,
                                                   float* __restrict__ out) {
    const float d = fp4_e8m0_to_fp32(blk[0]);
    const int8_t* k = fp4_codebook();
    const uint8_t* qs = blk + 1;
    for (int j = 0; j < 16; ++j) {
        const uint8_t byte = qs[j];
        out[j + 0]  = (float) k[byte & 0x0F] * d;
        out[j + 16] = (float) k[byte >> 4]   * d;
    }
}

// One NVFP4 block (36 bytes: 4x UE4M3 scales + 32 bytes = 64 values), four 16-element sub-blocks.
__device__ __forceinline__ void decode_nvfp4_block(const uint8_t* __restrict__ blk,
                                                   float* __restrict__ out) {
    const uint8_t* d4 = blk;
    const uint8_t* qs = blk + 4;
    const int8_t* k = fp4_codebook();
    for (int s = 0; s < 4; ++s) {
        const float d = fp4_ue4m3_to_fp32(d4[s]);
        float* yb = out + s * 16;
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            yb[j + 0] = (float) k[byte & 0x0F] * d;
            yb[j + 8] = (float) k[byte >> 4]   * d;
        }
    }
}

}  // namespace strata::kernels
