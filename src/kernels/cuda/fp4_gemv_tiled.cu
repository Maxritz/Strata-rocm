#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv_tiled.cu - P2.S3: MXPF4 / NVFP4 GEMV (Path B+: shared-memory x-tile).
//
// WHY THIS EXISTS: fp4_gemv_fast (2dbf569) vectorizes the decode but is still 0.99x on a single GEMV
// because the kernel is activation-bandwidth-bound - every thread re-reads the full `x` from global memory.
// In the MoE pattern (one activation vector per token, many expert rows) `x` is REUSED across rows, so
// tiling `x` once into shared memory per 32-row tile cuts activation traffic ~32x and moves the kernel
// off the bandwidth ceiling.
//
// CONTRACT (matches fp4_gemv): `x` is a SINGLE vector of length `n_in` reused across all `n_out` rows.
// `w` is `n_out` rows packed as [block_bytes * (n_in/ELEMS)] each.  One thread per output row; a 32-row
// block loads `x` into shared memory once and every row-thread dots its own weights against the shared
// activations.  The dot runs in natural element order (j = 0..n_in) per row, so output is bit-identical to
// the scalar reference up to fp32 accumulation - parity-tested against dequantize_mxfp4 / dequantize_nvfp4,
// NOT against the naive kernel.
//
// Same numerics-correctness rule as the FP4 tile: the FP4 codebook {0,1,2,3,4,6,8,12,…,±12} is nonlinear
// and overflows 4-bit, so FP4 codes are decoded via the shared codebook table (fp4_codebook()) into fp32
// then dotted with the fp16 activation - no raw-4-bit/dot, no activation quantization.
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/fp4_decode.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {

// Rows per output block - also the activation-reuse fanout (one x-tile shared by 32 rows).
constexpr int kMoeTileRows = 32;
// x loaded in chunks of kXTile elements (fp16) = 8 KB shared, comfortably under 32 KB.  Must be a multiple
// of the block size (32 / 64) so chunk boundaries align to weight-block boundaries per row.
constexpr int kXTile = 4096;

__global__ void fp4_gemv_tiled_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                      float* __restrict__ y, long long n_in, long long n_out, bool mxfp4) {
    const long long row = (long long) blockIdx.x * kMoeTileRows + threadIdx.x;
    const long long ELEMS = mxfp4 ? 32 : 64;
    const int BLOCK_BYTES = mxfp4 ? 17 : 36;
    const long long blocks_per_row = n_in / ELEMS;
    const uint8_t* wrow = w + row * blocks_per_row * BLOCK_BYTES;  // dead rows index past the buffer but never read it
    const int8_t* kv = fp4_codebook();
    const bool live = row < n_out;  // guard the per-row work, NOT the shared-load/sync below

    // Shared tile of the activation vector.  Reused by all 32 rows in this block.
    __shared__ uint16_t xshared[kXTile];

    float acc = 0.0f;
    // Walk x in aligned chunks so each chunk is a whole number of weight blocks per row.
    const long long n_chunks = (n_in + kXTile - 1) / kXTile;
    for (long long c = 0; c < n_chunks; ++c) {
        const long long seg0 = c * kXTile;
        const long long segN = seg0 + ((n_in - seg0 > kXTile) ? (long long)kXTile : (n_in - seg0));
        const long long seg_elems = segN - seg0;
        const long long seg_blocks = seg_elems / ELEMS;   // whole weight blocks inside this chunk

        // Load this x-chunk into shared memory (one uint16_t per thread, strided).  ALL 32 threads load so the
        // __syncthreads below is safe even for a trailing partial block (dead rows still participate here).
        for (long long i = threadIdx.x; i < seg_elems; i += kMoeTileRows)
            xshared[(size_t) i] = x[(size_t)(seg0 + i)];
        __syncthreads();

        // Each LIVE row-thread dots its own weights against the shared activations, natural element order.
        if (live) {
            for (long long b = 0; b < seg_blocks; ++b) {
                const uint8_t* blk = wrow + b * BLOCK_BYTES;
                const long long belem = seg0 + b * ELEMS;  // first element index of this weight block
                float vb[64];  // decoded block (ELEMS wide)
                if (mxfp4) {
                    const float d = fp4_e8m0_to_fp32(blk[0]);
                    const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 1);
                    for (int w32 = 0; w32 < 4; ++w32) {
                        const uint32_t v = q[w32];
                        const int lo = w32 * 4, hi = 16 + w32 * 4;
                        vb[lo + 0] = (float) kv[v & 0x0Fu]          * d;
                        vb[lo + 1] = (float) kv[(v >> 8) & 0x0Fu]   * d;
                        vb[lo + 2] = (float) kv[(v >> 16) & 0x0Fu]  * d;
                        vb[lo + 3] = (float) kv[(v >> 24) & 0x0Fu]  * d;
                        vb[hi + 0] = (float) kv[(v >> 4) & 0x0Fu]   * d;
                        vb[hi + 1] = (float) kv[(v >> 12) & 0x0Fu]  * d;
                        vb[hi + 2] = (float) kv[(v >> 20) & 0x0Fu]  * d;
                        vb[hi + 3] = (float) kv[(v >> 28) & 0x0Fu]  * d;
                    }
                } else {
                    const uint8_t* d4 = blk;
                    const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 4);
                    for (int s = 0; s < 4; ++s) {
                        const float d = fp4_ue4m3_to_fp32(d4[s]);
                        const uint32_t* qs = q + s * 2;
                        for (int w32 = 0; w32 < 2; ++w32) {
                            const uint32_t v = qs[w32];
                            const int lo = s * 16 + w32 * 4, hi = s * 16 + 8 + w32 * 4;
                            vb[lo + 0] = (float) kv[v & 0x0Fu]          * d;
                            vb[lo + 1] = (float) kv[(v >> 8) & 0x0Fu]   * d;
                            vb[lo + 2] = (float) kv[(v >> 16) & 0x0Fu]  * d;
                            vb[lo + 3] = (float) kv[(v >> 24) & 0x0Fu]  * d;
                            vb[hi + 0] = (float) kv[(v >> 4) & 0x0Fu]   * d;
                            vb[hi + 1] = (float) kv[(v >> 12) & 0x0Fu]  * d;
                            vb[hi + 2] = (float) kv[(v >> 20) & 0x0Fu]  * d;
                            vb[hi + 3] = (float) kv[(v >> 28) & 0x0Fu]  * d;
                        }
                    }
                }
                const long long xbase = belem - seg0;  // index into the shared tile
                for (int j = 0; j < (int) ELEMS; ++j)
                    acc += vb[j] * __half2float(__ushort_as_half(xshared[(size_t)(xbase + j)]));
            }
        }
        __syncthreads();  // every thread (incl. dead rows) waits before the next chunk reuses xshared
    }
    if (live) y[row] = acc;
}

void fp4_gemv_tiled(const uint16_t* x, const uint8_t* w, float* y,
                    int64_t n_in, int64_t n_out, bool mxfp4) {
    if (n_in <= 0 || n_out <= 0) return;
    const long long ELEMS = mxfp4 ? 32 : 64;
    if (n_in % ELEMS != 0) {
        std::fprintf(stderr, "fp4_gemv_tiled: n_in %lld is not a multiple of %lld\n",
                     (long long) n_in, (long long) ELEMS);
        std::exit(1);
    }
    if (n_in > kXTile) {
        // The chunked loop handles n_in > kXTile, but shared mem is fixed at kXTile; guard explicitly.
        std::fprintf(stderr, "fp4_gemv_tiled: n_in %lld exceeds kXTile %d (raise kXTile)\n",
                     (long long) n_in, kXTile);
        std::exit(1);
    }
    const int blocks = (int) ((n_out + kMoeTileRows - 1) / kMoeTileRows);
    fp4_gemv_tiled_kernel<<<blocks, kMoeTileRows>>>(x, w, y, n_in, n_out, mxfp4);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { std::fprintf(stderr, "fp4_gemv_tiled launch: %s\n", hipGetErrorString(e)); std::exit(1); }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) { std::fprintf(stderr, "fp4_gemv_tiled: %s\n", hipGetErrorString(s)); std::exit(1); }
}

}  // namespace strata::kernels
