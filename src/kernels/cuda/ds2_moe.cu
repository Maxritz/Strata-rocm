#include "hip/hip_runtime.h"
// src/kernels/cuda/ds2_moe.cu - the deepseek2 router and SwiGLU.  See include/strata/kernels/ds2_moe.hpp.
#include "strata/kernels/ds2_moe.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int DS2_MAX_THREADS = 512;
constexpr int DS2_MAX_EXPERT = 4096;

void launch_check() {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("ds2_moe launch: ") + hipGetErrorString(e));
}

__global__ void ds2_router_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                  int k, float scale, int norm, int32_t* __restrict__ ids,
                                  float* __restrict__ weights) {
    extern __shared__ unsigned char raw[];
    float* s_probs = reinterpret_cast<float*>(raw);
    float* s_sel = s_probs + n_expert;
    unsigned char* s_taken = reinterpret_cast<unsigned char*>(s_sel + n_expert);
    __shared__ float red[DS2_MAX_THREADS / 32];
    __shared__ int rid[DS2_MAX_THREADS / 32];

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const float* l = logits + (size_t) t * n_expert;

    for (int e = tid; e < n_expert; e += nt) {
        const float p = 1.0f / (1.0f + expf(-l[e]));
        s_probs[e] = p;
        s_sel[e] = bias ? p + bias[e] : p;
        s_taken[e] = 0;
    }
    __syncthreads();

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float v = s_sel[e];
            if (v > bv) { bv = v; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffffffffffull, bv, off);
            const int oi = __shfl_down_sync(0xffffffffffffffffull, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { red[tid >> 5] = bv; rid[tid >> 5] = bi; }
        __syncthreads();
        if (tid < 32) {
            const int nw = (nt + 31) >> 5;
            float v = tid < nw ? red[tid] : -INFINITY;
            int ix = tid < nw ? rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffffffffffull, v, off);
                const int oi = __shfl_down_sync(0xffffffffffffffffull, ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(size_t) t * k + i] = ix;
                weights[(size_t) t * k + i] = s_probs[ix];
                s_taken[ix] = 1;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
        if (norm) {
            const double sc = fmax(s, 6.103515625e-05);
            for (int i = 0; i < k; ++i) weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
        }
        if (scale != 0.0f && scale != 1.0f)
            for (int i = 0; i < k; ++i) weights[(size_t) t * k + i] *= scale;
    }
}

__global__ void swiglu_mul_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                  float* __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

}  // namespace

void ds2_router(const float* logits, const float* bias, int n_tokens, int n_expert, int k, float scale, bool norm,
                int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0 || k > 64 || n_expert > DS2_MAX_EXPERT)
        throw std::invalid_argument("ds2_router: bad shape");
    if (!logits || !ids || !weights) throw std::invalid_argument("ds2_router: null buffer");
    if (!std::isfinite(scale)) throw std::invalid_argument("ds2_router: non-finite scale");
    int threads = (n_expert + 31) & ~31;
    if (threads > DS2_MAX_THREADS) threads = DS2_MAX_THREADS;
    const size_t smem = (size_t) n_expert * (sizeof(float) * 2) + (size_t) n_expert;
    ds2_router_kernel<<<(unsigned) n_tokens, threads, smem, (hipStream_t) stream>>>(
        logits, bias, n_expert, k, scale, norm ? 1 : 0, ids, weights);
    launch_check();
    if (stream == nullptr) {
        const hipError_t s = hipDeviceSynchronize();
        if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
    }
}

void swiglu_mul(const float* gate, const float* up, float* out, int64_t n, void* stream) {
    if (n <= 0) return;
    if (!gate || !up || !out) throw std::invalid_argument("swiglu_mul: null buffer");
    const long long total = (long long) n;
    const unsigned blocks = (unsigned) ((total + 255) / 256);
    swiglu_mul_kernel<<<blocks, 256, 0, (hipStream_t) stream>>>(gate, up, out, total);
    launch_check();
    if (stream == nullptr) {
        const hipError_t s = hipDeviceSynchronize();
        if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
    }
}

}  // namespace strata::kernels
