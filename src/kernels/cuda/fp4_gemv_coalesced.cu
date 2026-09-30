#include "hip/hip_runtime.h"
// src/kernels/cuda/fp4_gemv_coalesced.cu - P2.S4: MXPF4 / NVFP4 GEMV (bandwidth-coalesced).
//
// WHY: fp4_gemv_fast/tiled were correct but stuck at ~65 GB/s of ~640 GB/s on the RX 9070 XT (measured
// 4096x32768: 1.09 ms for a 71.3 MB weight stream).  That is a COALESCING failure, not compute: one thread
// per row means a warp's 32 threads read rows 2176 bytes apart, so every load touches 32 distinct cache
// lines and pulls ~8x the useful bytes.  The GPU is starved on memory transactions, not flops.
//
// FIX: a block cooperatively stages ONE row's weights into shared memory with COALESCED byte loads
// (consecutive threads -> consecutive addresses), then decodes from shared (bank-parallel, no coalescing
// constraint) and block-reduces.  The activation vector is staged once per block and reused across every
// row the block processes.  Decode is the SAME shared decoder as fp4_gemv_fast/tiled (fp4_decode.hpp), so
// parity is against the validated scalar reference, not against another kernel.
//
// CONTRACT (matches fp4_gemv): `x` is a single vector of length `n_in` reused across all `n_out` rows.
#include "strata/kernels/fp4_gemv.hpp"
#include "strata/kernels/fp4_decode.hpp"

#include <hip/hip_fp16.h>

namespace strata::kernels {

constexpr int kCoalescedThreads = 128;

__global__ void fp4_gemv_coalesced_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                          float* __restrict__ y, long long n_in, long long n_out, bool mxfp4) {
    const long long ELEMS = mxfp4 ? 32 : 64;
    const int BLOCK_BYTES = mxfp4 ? 17 : 36;
    const long long blocks_per_row = n_in / ELEMS;
    const long long row_bytes = blocks_per_row * BLOCK_BYTES;

    extern __shared__ uint8_t smem[];
    uint16_t* xsh = reinterpret_cast<uint16_t*>(smem);          // x staged once per block (fp16, 8 KB)
    uint8_t*  rowbuf = smem + n_in * sizeof(uint16_t);          // one row's weights, coalesced in
    float*    red = reinterpret_cast<float*>(rowbuf + row_bytes); // block reduction scratch

    // Stage the activation vector ONCE and reuse it for every row this block handles.  (Converting x to fp32
    // here was tried and REGRESSED: doubling shared to 16 KB halved occupancy and cost more than the per-element
    // __half2float it saved - measured 0.885 ms -> 1.588 ms.  Keep fp16 in shared.)
    for (long long i = threadIdx.x; i < n_in; i += kCoalescedThreads)
        xsh[(size_t) i] = x[(size_t) i];
    __syncthreads();

    const long long row_u4 = row_bytes / 16;               // 2176/16=136 (MXFP4), 1152/16=72 (NVFP4)

    for (long long r = blockIdx.x; r < n_out; r += gridDim.x) {
        const uint8_t* wrow = w + r * row_bytes;
        // Coalesced 16-byte staging of the row: consecutive threads read consecutive 16-byte chunks.
        if (row_u4 > 0) {
            const uint4* ws = reinterpret_cast<const uint4*>(wrow);
            uint4* rd = reinterpret_cast<uint4*>(rowbuf);
            for (long long i = threadIdx.x; i < row_u4; i += kCoalescedThreads) rd[i] = ws[i];
        }
        for (long long i = row_u4 * 16 + threadIdx.x; i < row_bytes; i += kCoalescedThreads)
            rowbuf[(size_t) i] = wrow[(size_t) i];         // tail bytes (row_bytes not a multiple of 16)
        __syncthreads();

        float partial = 0.0f;
        const int8_t* kv = fp4_codebook();
        for (long long b = threadIdx.x; b < blocks_per_row; b += kCoalescedThreads) {
            // FUSED decode + dot: no `float dec[ELEMS]` array (dynamically indexed -> local memory, a load per
            // element).  Decoding inline keeps every value in a register.
            const long long base = b * ELEMS;
            if (mxfp4) {
                const uint8_t* blk = rowbuf + b * BLOCK_BYTES;
                const uint8_t* qs = blk + 1;                       // 16 code bytes, split-half
                const float d = fp4_e8m0_to_fp32(blk[0]);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t byte = qs[j];
                    partial += (float) kv[byte & 0x0Fu]    * d * __half2float(__ushort_as_half(xsh[(size_t)(base + j)]));
                    partial += (float) kv[(byte >> 4) & 0x0Fu] * d * __half2float(__ushort_as_half(xsh[(size_t)(base + 16 + j)]));
                }
            } else {
                const uint8_t* blk = rowbuf + b * BLOCK_BYTES;
                const uint8_t* qs = blk + 4;                       // 32 code bytes, 4 sub-blocks
                for (int s = 0; s < 4; ++s) {
                    const float d = fp4_ue4m3_to_fp32(blk[s]);
                    const uint8_t* qsb = qs + s * 8;
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t byte = qsb[j];
                        partial += (float) kv[byte & 0x0Fu]      * d * __half2float(__ushort_as_half(xsh[(size_t)(base + s * 16 + j)]));
                        partial += (float) kv[(byte >> 4) & 0x0Fu] * d * __half2float(__ushort_as_half(xsh[(size_t)(base + s * 16 + 8 + j)]));
                    }
                }
            }
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int s = kCoalescedThreads / 2; s > 0; s >>= 1) {
            if ((int) threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[r] = red[0];
        __syncthreads();  // rowbuf is reused next iteration
    }
}

// ---- memory probe: same coalesced 16-byte read pattern as the GEMV staging, NO decode.  Separates
// "memory-bound" from "decode-bound" by timing the raw weight read alone.
__global__ void fp4_memprobe_kernel(const uint8_t* __restrict__ w, float* __restrict__ out, long long bytes) {
    unsigned long long acc = 0;
    const long long n4 = bytes / 16;
    const uint4* ws = reinterpret_cast<const uint4*>(w);
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (long long) gridDim.x * blockDim.x) {
        const uint4 v = ws[i];
        acc += (unsigned long long) v.x + v.y + v.z + v.w;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = (float) acc;
}

void fp4_memprobe(const uint8_t* w, float* out, int64_t bytes, int grid) {
    fp4_memprobe_kernel<<<grid, kCoalescedThreads>>>(w, out, bytes);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { std::fprintf(stderr, "fp4_memprobe launch: %s\n", hipGetErrorString(e)); std::exit(1); }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) { std::fprintf(stderr, "fp4_memprobe: %s\n", hipGetErrorString(s)); std::exit(1); }
}

void fp4_gemv_coalesced(const uint16_t* x, const uint8_t* w, float* y,
                        int64_t n_in, int64_t n_out, bool mxfp4) {
    if (n_in <= 0 || n_out <= 0) return;
    const long long ELEMS = mxfp4 ? 32 : 64;
    if (n_in % ELEMS != 0) {
        std::fprintf(stderr, "fp4_gemv_coalesced: n_in %lld is not a multiple of %lld\n",
                     (long long) n_in, (long long) ELEMS);
        std::exit(1);
    }
    const long long row_bytes = (n_in / ELEMS) * (mxfp4 ? 17 : 36);
    const size_t smem = (size_t) n_in * sizeof(uint16_t) + (size_t) row_bytes + kCoalescedThreads * sizeof(float);
    // Cap grid so the activation staging stays amortised; grid-stride covers the rows regardless.
    const int grid = (int) (n_out < 2048 ? n_out : 2048);
    fp4_gemv_coalesced_kernel<<<grid, kCoalescedThreads, smem>>>(x, w, y, n_in, n_out, mxfp4);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) { std::fprintf(stderr, "fp4_gemv_coalesced launch: %s\n", hipGetErrorString(e)); std::exit(1); }
    const hipError_t s = hipDeviceSynchronize();
    if (s != hipSuccess) { std::fprintf(stderr, "fp4_gemv_coalesced: %s\n", hipGetErrorString(s)); std::exit(1); }
}

}  // namespace strata::kernels
