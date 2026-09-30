#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv_q8.cu - P2.S5: MXFP4 / NVFP4 GEMV in the SHAPE THE MOE KERNELS USE.
//
// WHY: fp4_gemv_coalesced (3cee14d) is correct and 3.19x, but it keeps fp16 activations and does fp32 FMA.
// The engine's own MoE kernels (iq_kernels.cu: vec_dot_q4_K_q8_1 + row_dot + mmvq_kernel) are faster for the
// same reason they are fast at all: they QUANTIZE THE ACTIVATION TO Q8 (int8) ONCE and then do an INTEGER
// MAC (sumi += code * (int)y->qs[l]), applying the float scale once per block.  Integer MACs avoid the
// per-element fp16->fp32 convert and the fp32 FMA pipe, and the row is walked WARP-COOPERATIVELY with a
// lane stride, which is coalesced.  This file gives FP4 that same shape.
//
// SHAPE (mirrors iq_kernels.cu:413-437):
//   * one warp per output row; lane k handles blocks k, k+32, k+64, ... of that row (coalesced).
//   * the activation is pre-quantized to int8 per 32-element group (scale per group).
//   * the FP4 codes map through the shared kvalues table to int8 and are MAC'd against the int8 activation;
//     the integer sum is scaled by (fp4_block_scale * activation_group_scale) once per block.
//   * warp-shuffle reduce at the end.  No shared memory, no local array.
//
// NUMERICS: this quantizes the ACTIVATION to int8, exactly as the MoE hit path does.  It is therefore NOT
// bit-comparable to the fp32-activation reference; its parity test models the SAME int8 activation
// quantization (see fp4_gemv_parity.cpp, the q8 section).  FP4 weight decode is unchanged (fp4_decode.hpp).
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/fp4_decode.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {

// ---- activation quantizer: fp16 x[n] -> int8 codes + one fp32 scale per 32-element group.
__global__ void fp4_quantize_x_q8_kernel(const uint16_t* __restrict__ x, int8_t* __restrict__ xq,
                                         float* __restrict__ xd, long long n) {
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long groups = n / 32;
    if (g >= groups) return;
    float mx = 0.0f;
    float v[32];
    for (int i = 0; i < 32; ++i) {
        v[i] = __half2float(__ushort_as_half(x[(size_t)(g * 32 + i)]));
        const float a = fabsf(v[i]);
        if (a > mx) mx = a;
    }
    const float d = mx > 0.0f ? mx / 127.0f : 0.0f;
    xd[(size_t) g] = d;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; ++i) {
        int q = (int) lrintf(v[i] * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        xq[(size_t)(g * 32 + i)] = (int8_t) q;
    }
}

// One warp per row, lane-strided over the row's FP4 blocks, integer MAC against the int8 activation.
constexpr int kQ8Warps = 4;   // rows per block (matches mmvq_kernel's 4)

__global__ void fp4_gemv_q8_kernel(const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                   const uint8_t* __restrict__ w, float* __restrict__ y,
                                   long long n_in, long long n_out, bool mxfp4) {
    const long long ELEMS = mxfp4 ? 32 : 64;
    const int BLOCK_BYTES = mxfp4 ? 17 : 36;
    const long long blocks_per_row = n_in / ELEMS;
    const long long row = (long long) blockIdx.x * kQ8Warps + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* wrow = w + row * blocks_per_row * BLOCK_BYTES;
    const int8_t* kv = fp4_codebook();

    float acc = 0.0f;
    for (long long b = lane; b < blocks_per_row; b += 32) {
        const uint8_t* blk = wrow + b * BLOCK_BYTES;
        const long long base = b * ELEMS;
        long long sumi = 0;
        if (mxfp4) {
            // Split-half: byte qs[j] low -> elem j, high -> elem j+16; x group = b (32 elems/group).
            const uint8_t* qs = blk + 1;
            const int8_t* xg = xq + base;
            for (int j = 0; j < 16; ++j) {
                const uint8_t byte = qs[j];
                sumi += (long long) kv[byte & 0x0Fu]    * xg[j];
                sumi += (long long) kv[(byte >> 4) & 0x0Fu] * xg[16 + j];
            }
            acc += fp4_e8m0_to_fp32(blk[0]) * xd[base / 32] * (float) sumi;
        } else {
            // 4 sub-blocks of 16, each with its own scale; x group (32) = 2b + s/2.
            const uint8_t* qs = blk + 4;
            for (int s = 0; s < 4; ++s) {
                const int8_t* xg = xq + base + s * 16;
                const uint8_t* qsb = qs + s * 8;
                long long ss = 0;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qsb[j];
                    ss += (long long) kv[byte & 0x0Fu]      * xg[j];
                    ss += (long long) kv[(byte >> 4) & 0x0Fu] * xg[8 + j];
                }
                acc += fp4_ue4m3_to_fp32(blk[s]) * xd[(base + s * 16) / 32] * (float) ss;
            }
        }
    }
    // warp reduce
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffffffffffull, acc, o);
    if (lane == 0) y[row] = acc;
}

void fp4_gemv_q8(const uint16_t* x, const uint8_t* w, float* y,
                 int64_t n_in, int64_t n_out, bool mxfp4) {
    if (n_in <= 0 || n_out <= 0) return;
    const long long ELEMS = mxfp4 ? 32 : 64;
    if (n_in % ELEMS != 0 || n_in % 32 != 0) {
        std::fprintf(stderr, "fp4_gemv_q8: n_in %lld must be a multiple of %lld and 32\n",
                     (long long) n_in, (long long) ELEMS);
        std::exit(1);
    }
    int8_t* d_xq = nullptr; float* d_xd = nullptr;
    if (hipMalloc(&d_xq, (size_t) n_in) != hipSuccess ||
        hipMalloc(&d_xd, (size_t)(n_in / 32) * sizeof(float)) != hipSuccess) {
        std::fprintf(stderr, "fp4_gemv_q8: x-scratch malloc failed\n"); std::exit(1);
    }
    fp4_quantize_x_q8_kernel<<<(unsigned)((n_in / 32 + 255) / 256), 256>>>(x, d_xq, d_xd, n_in);
    const int grid = (int) ((n_out + kQ8Warps - 1) / kQ8Warps);
    fp4_gemv_q8_kernel<<<grid, kQ8Warps * 32>>>(d_xq, d_xd, w, y, n_in, n_out, mxfp4);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { std::fprintf(stderr, "fp4_gemv_q8 launch: %s\n", hipGetErrorString(e)); std::exit(1); }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) { std::fprintf(stderr, "fp4_gemv_q8: %s\n", hipGetErrorString(s)); std::exit(1); }
    (void) hipFree(d_xq); (void) hipFree(d_xd);
}

}  // namespace strata::kernels
