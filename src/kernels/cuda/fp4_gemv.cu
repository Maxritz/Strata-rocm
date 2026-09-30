#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv.cu - P2.S2: MXFP4 / NVFP4 GEMV (Path A: fp16 x, software FP4 decode, fp32 acc).
//
// See fp4_gemv.hpp.  This is the naive baseline (one thread per output row) checked against the validated
// scalar dequantizers in fp4_gemv_parity.cpp.  The FP4 decode helpers live in fp4_decode.hpp and are shared
// with fp4_gemv_fast.cu so the two kernels cannot drift; a decode bug shows in BOTH parity tests.
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/fp4_decode.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {

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
