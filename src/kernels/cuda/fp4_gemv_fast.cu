#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv_fast.cu - P2.S2: MXFP4 / NVFP4 GEMV (Path B: vectorized decode, same numerics).
//
// Same math as fp4_gemv.cu - one thread per output row, fp16 activations, fp32 accumulation, identical block
// scale and codebook - but the FP4 *decode* is vectorized instead of scalar: a whole 16-byte block is
// loaded as four uint32s and eight codes are unpacked per uint32 via the kFp4 codebook table (a table
// lookup costs ~1 cycle on RDNA4 shared/constant, versus 5+ for `>>`/`&`/`ldexp`-on-fp4 + mul), then the
// exact same dot loop as the baseline runs over the register array.  Numerics are bit-identical aside from
// summation order, so the parity test compares against the validated scalar reference, not the baseline.
//
// NOTE: the codebook values {0,1,2,3,4,6,8,12,…,±12} do NOT fit in 4-bit and the codebook is NONLINEAR, so
// the FP4 CODES are NOT fed to a 4-bit dot instruction.  The int8 UDOT4 path that also quantizes the
// activation to int8 is a SEPARATE item (it changes the numerics).
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/fp4_decode.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {

namespace {
__device__ static const int8_t kFp4_dev[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
}

__global__ void fp4_gemv_fast_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                     float* __restrict__ y, long long n_in, long long n_out, bool mxfp4) {
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;

    const long long blocks_per_row = mxfp4 ? (n_in / 32) : (n_in / 64);
    const int block_bytes = mxfp4 ? 17 : 36;
    const long long elems_per_block = mxfp4 ? 32 : 64;
    const uint8_t* wrow = w + o * blocks_per_row * block_bytes;
    const int8_t* kv = kFp4_dev;

    float acc = 0.0f;
    float vblock[64];  // holds the decoded block (32 for MXFP4); the dot loop is block-size-agnostic.
    for (long long b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk = wrow + b * block_bytes;
        const long long base = b * elems_per_block;

        if (mxfp4) {
            const float d = fp4_e8m0_to_fp32(blk[0]);
            const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 1);  // 16 bytes = 4 uint32
            // Split-half MXFP4 layout (matches decode_mxfp4_block): byte qs[j]'s LOW nibble -> elem j (0..15),
            // HIGH nibble -> elem 16+j (16..31).  So loading 4 bytes as one uint32 puts this uint32's four
            // LOW nibbles at elems [4*w32 .. 4*w32+3] (contiguous) and its four HIGH nibbles at elems
            // [16+4*w32 .. 16+4*w32+3] (contiguous).  Writing them that way keeps vblock[j] == out[j] exactly.
            for (int w32 = 0; w32 < 4; ++w32) {
                const uint32_t v = q[w32];
                const int lo = w32 * 4;
                const int hi = 16 + w32 * 4;
                vblock[(size_t)(lo + 0)] = (float) kv[v & 0x0Fu]          * d;
                vblock[(size_t)(lo + 1)] = (float) kv[(v >> 8) & 0x0Fu]   * d;
                vblock[(size_t)(lo + 2)] = (float) kv[(v >> 16) & 0x0Fu]  * d;
                vblock[(size_t)(lo + 3)] = (float) kv[(v >> 24) & 0x0Fu]  * d;
                vblock[(size_t)(hi + 0)] = (float) kv[(v >> 4) & 0x0Fu]   * d;
                vblock[(size_t)(hi + 1)] = (float) kv[(v >> 12) & 0x0Fu]  * d;
                vblock[(size_t)(hi + 2)] = (float) kv[(v >> 20) & 0x0Fu]  * d;
                vblock[(size_t)(hi + 3)] = (float) kv[(v >> 28) & 0x0Fu]  * d;
            }
        } else {
            const uint8_t* d4 = blk;                    // 4 UE4M3 sub-block scales
            const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 4);  // 32 bytes = 8 uint32
            // NVFP4 per-sub-block split (matches decode_nvfp4_block): byte qs[s*8+j] LOW -> elem s*16+j,
            // HIGH -> elem s*16+8+j.  Two uint32s per sub-block (q+s*2), four bytes each -> contiguous low
            // group [s*16+4*w32 .. s*16+4*w32+3] and contiguous high group [s*16+8+4*w32 .. +3].
            for (int s = 0; s < 4; ++s) {
                const float d = fp4_ue4m3_to_fp32(d4[s]);
                const uint32_t* qs = q + s * 2;
                for (int w32 = 0; w32 < 2; ++w32) {
                    const uint32_t v = qs[w32];
                    const int lo = s * 16 + w32 * 4;
                    const int hi = s * 16 + 8 + w32 * 4;
                    vblock[(size_t)(lo + 0)] = (float) kv[v & 0x0Fu]          * d;
                    vblock[(size_t)(lo + 1)] = (float) kv[(v >> 8) & 0x0Fu]   * d;
                    vblock[(size_t)(lo + 2)] = (float) kv[(v >> 16) & 0x0Fu]  * d;
                    vblock[(size_t)(lo + 3)] = (float) kv[(v >> 24) & 0x0Fu]  * d;
                    vblock[(size_t)(hi + 0)] = (float) kv[(v >> 4) & 0x0Fu]   * d;
                    vblock[(size_t)(hi + 1)] = (float) kv[(v >> 12) & 0x0Fu]  * d;
                    vblock[(size_t)(hi + 2)] = (float) kv[(v >> 20) & 0x0Fu]  * d;
                    vblock[(size_t)(hi + 3)] = (float) kv[(v >> 28) & 0x0Fu]  * d;
                }
            }
        }
        // Dot the decoded values with the fp16 activations, in natural element order - identical to the
        // baseline's loop, so the only difference is summation order, not the pairing.
        for (int j = 0; j < elems_per_block; ++j) {
            acc += vblock[(size_t) j] * __half2float(__ushort_as_half(x[(size_t) (base + j)]));
        }
    }
    y[o] = acc;
}

void fp4_gemv_fast(const uint16_t* x, const uint8_t* w, float* y,
                   int64_t n_in, int64_t n_out, bool mxfp4) {
    if (n_in <= 0 || n_out <= 0) return;
    const long long elems_per_block = mxfp4 ? 32 : 64;
    if (n_in % elems_per_block != 0) {
        std::fprintf(stderr, "fp4_gemv_fast: n_in %lld is not a multiple of %lld\n",
                     (long long) n_in, (long long) elems_per_block);
        std::exit(1);
    }
    const int threads = 128;
    const int blocks = (int) ((n_out + threads - 1) / threads);
    fp4_gemv_fast_kernel<<<blocks, threads>>>(x, w, y, n_in, n_out, mxfp4);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        std::fprintf(stderr, "fp4_gemv_fast launch: %s\n", hipGetErrorString(e));
        std::exit(1);
    }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) {
        std::fprintf(stderr, "fp4_gemv_fast: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}

}  // namespace strata::kernels
