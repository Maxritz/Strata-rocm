#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv.cu - P2.S2: MXFP4 / NVFP4 GEMV (Path A: fp16 x, software FP4 decode, fp32 acc).
//
// See fp4_gemv.hpp.  This is the naive, parity-testable kernel against which the U44 dot path must match.
#include "strata/kernels/fp4_gemv.hpp"

#include "strata/artifact/dequant.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {
namespace {

// ---- device copies of the validated scalar decoders (dequant.hpp).  These are bit-for-bit the host path
// that `dequant_fp4_test` checks against ggml, transcribed to __device__ so a kernel-side scale bug cannot
// diverge from the reference it is parity-tested against.

__device__ __forceinline__ float e8m0_to_fp32_half_dev(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;                                  // 2^(x-128)
}

__device__ __forceinline__ float ue4m3_to_fp32_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;       // 0x7F is the UE4M3 NaN encoding -> mapped to 0 by ggml
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = exp == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;                          // halved, because codebooks here are stored DOUBLED
}

// The FP4 E2M1 codebook, verbatim from ggml-common.h / dequant.hpp: 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12.
// Stored as int8 so the signed kvalues plug straight into a signed int8 dot path.
__device__ static const int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ONE FP4 BLOCK, decoded to `out` (32 for MXFP4, 64 for NVFP4).  Mirrors dequantize_mxfp4 / dequantize_nvfp4
// exactly: split-half nibble order, block-scale applied to every code.
__device__ __forceinline__ void decode_mxfp4_block(const uint8_t* __restrict__ blk, float* __restrict__ out) {
    const float d = e8m0_to_fp32_half_dev(blk[0]);
    const uint8_t* qs = blk + 1;                // 16 bytes -> 32 values
    for (int j = 0; j < 16; ++j) {
        const uint8_t byte = qs[j];
        out[j + 0]  = (float) kFp4[byte & 0x0F] * d;
        out[j + 16] = (float) kFp4[byte >> 4]   * d;
    }
}

__device__ __forceinline__ void decode_nvfp4_block(const uint8_t* __restrict__ blk, float* __restrict__ out) {
    const uint8_t* d4 = blk;                    // 4 UE4M3 scales
    const uint8_t* qs = blk + 4;                // 32 bytes -> 4 sub-blocks of 16
    for (int s = 0; s < 4; ++s) {
        const float d = ue4m3_to_fp32_dev(d4[s]);
        float* yb = out + s * 16;
        for (int j = 0; j < 8; ++j) {
            const uint8_t byte = qs[s * 8 + j];
            yb[j + 0] = (float) kFp4[byte & 0x0F] * d;
            yb[j + 8] = (float) kFp4[byte >> 4]   * d;
        }
    }
}

}  // namespace

__global__ void fp4_gemv_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, float* __restrict__ y,
                                long long n_in, long long n_out, bool mxfp4) {
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    const long long blocks_per_row = mxfp4 ? (n_in / 32) : (n_in / 64);
    const int block_bytes = mxfp4 ? 17 : 36;
    const long long elems_per_block = mxfp4 ? 32 : 64;
    const uint8_t* wrow = w + o * blocks_per_row * block_bytes;

    float acc = 0.0f;
    // Decode one block at a time into registers, dot against the activation.  `n_in` is a multiple of the
    // block size (asserted by the host wrapper), so there is no tail on the block loop.
    float block[64];  // NVFP4's max; MXFP4 uses only the first 32
    for (long long b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk = wrow + b * block_bytes;
        if (mxfp4) decode_mxfp4_block(blk, block);
        else       decode_nvfp4_block(blk, block);
        const long long base = b * elems_per_block;
        for (int j = 0; j < elems_per_block; ++j) {
            // fp16 activation -> fp32; one __half2float per element, same as s_gemv_kernel.
            acc += block[(size_t) j] * __half2float(__ushort_as_half(x[(size_t) (base + j)]));
        }
    }
    y[o] = acc;
}

void fp4_gemv(const uint16_t* x, const uint8_t* w, float* y,
              int64_t n_in, int64_t n_out, bool mxfp4) {
    if (n_in <= 0 || n_out <= 0) return;
    const long long elems_per_block = mxfp4 ? 32 : 64;
    if (n_in % elems_per_block != 0) {
        std::fprintf(stderr, "fp4_gemv: n_in %lld is not a multiple of %lld\n",
                     (long long) n_in, (long long) elems_per_block);
        std::exit(1);
    }
    const int threads = 128;
    const int blocks = (int) ((n_out + threads - 1) / threads);
    fp4_gemv_kernel<<<blocks, threads>>>(x, w, y, n_in, n_out, mxfp4);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        std::fprintf(stderr, "fp4_gemv launch: %s\n", hipGetErrorString(e));
        std::exit(1);
    }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) {
        std::fprintf(stderr, "fp4_gemv: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}

}  // namespace strata::kernels
