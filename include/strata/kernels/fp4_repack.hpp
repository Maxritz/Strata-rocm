// include/strata/kernels/fp4_repack.hpp - K3: repack the ggml FP4 block layouts to 16-byte-aligned arrays.
//
// ggml's MXFP4 block is 17 bytes (1 E8M0 scale + 16 data bytes = 32 values) and NVFP4 is 36 bytes (4 UE4M3
// scales + 32 data bytes = 64 values).  The odd strides (17, 36) mean neither the scale nor the 16 data bytes
// of a block is 16-byte aligned, so a 128-bit (`uint4`) load always spans two blocks and the vectorizer gives
// up (docs/CONTRACT.md K3).  The repack below splits each tensor into a SCALE array and a DATA array, so the
// data is contiguous and 16-byte aligned (`uint4`-loadable) and the scales are their own compact stream.
//
// This is a lossless RELAYOUT: the values are copied verbatim, so decode(repacked) == decode(original) bit for
// bit.  `fp4_repack_parity` is that test and it is not optional.
#pragma once

#include "strata/kernels/fp4_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::kernels {

// Bytes per block in each ggml layout (docs/TODO.md P1 #6, measured from ggml-common.h).
constexpr size_t kMxfp4BlockBytes = 17;   // 1 E8M0 scale + 16 data
constexpr size_t kNvfp4BlockBytes = 36;   // 4 UE4M3 scales + 32 data

struct Fp4Repack {
    std::vector<uint8_t> scales;   // nblocks * (1 for MXFP4, 4 for NVFP4)
    std::vector<uint8_t> data;     // nblocks * (16 for MXFP4, 32 for NVFP4), 16-byte aligned
    size_t nblocks = 0;
    bool mxfp4 = true;
};

// MXFP4: src is nblocks * 17 bytes, block i = {scale, 16 data}.  Copy verbatim into scales/data.
inline Fp4Repack repack_mxfp4(const uint8_t* src, size_t nblocks) {
    Fp4Repack r;
    r.nblocks = nblocks;
    r.mxfp4 = true;
    r.scales.resize(nblocks * 1);
    r.data.resize(nblocks * 16);
    for (size_t i = 0; i < nblocks; ++i) {
        r.scales[i] = src[i * kMxfp4BlockBytes];
        std::memcpy(r.data.data() + i * 16, src + i * kMxfp4BlockBytes + 1, 16);
    }
    return r;
}

// NVFP4: src is nblocks * 36 bytes, block i = {4 scales, 32 data}.
inline Fp4Repack repack_nvfp4(const uint8_t* src, size_t nblocks) {
    Fp4Repack r;
    r.nblocks = nblocks;
    r.mxfp4 = false;
    r.scales.resize(nblocks * 4);
    r.data.resize(nblocks * 32);
    for (size_t i = 0; i < nblocks; ++i) {
        std::memcpy(r.scales.data() + i * 4, src + i * kNvfp4BlockBytes, 4);
        std::memcpy(r.data.data() + i * 32, src + i * kNvfp4BlockBytes + 4, 32);
    }
    return r;
}

// Decode one block out of the repacked layout, for the parity test.  32 values for MXFP4, 64 for NVFP4.
inline void decode_repacked_mxfp4(const Fp4Repack& r, size_t i, float* out) {
    const float d = fp4_e8m0_to_fp32(r.scales[i]);
    const int8_t* k = fp4_codebook();
    const uint8_t* qs = r.data.data() + i * 16;
    for (int j = 0; j < 16; ++j) {
        out[j + 0]  = (float) k[qs[j] & 0x0F] * d;
        out[j + 16] = (float) k[qs[j] >> 4]   * d;
    }
}

inline void decode_repacked_nvfp4(const Fp4Repack& r, size_t i, float* out) {
    const int8_t* k = fp4_codebook();
    const uint8_t* sc = r.scales.data() + i * 4;
    const uint8_t* qs = r.data.data() + i * 32;
    for (int s = 0; s < 4; ++s) {
        const float d = fp4_ue4m3_to_fp32(sc[s]);
        float* yb = out + s * 16;
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            yb[j + 0] = (float) k[byte & 0x0F] * d;
            yb[j + 8] = (float) k[byte >> 4]   * d;
        }
    }
}

}  // namespace strata::kernels
