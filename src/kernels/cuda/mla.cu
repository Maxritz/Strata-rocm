#include "hip/hip_runtime.h"
// src/kernels/cuda/mla.cu - the MLA core kernels.  See include/strata/kernels/mla.hpp for the contract.
#include "strata/kernels/mla.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int MLA_THREADS = 256;

void check_launch() {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("mla launch: ") + hipGetErrorString(e));
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("mla: ") + what);
}

__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffffffffffull, v, off));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = (int) ((blockDim.x + 31) >> 5);
    if ((int) threadIdx.x < 32) {
        float t = (int) threadIdx.x < nw ? red[threadIdx.x] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) t = fmaxf(t, __shfl_down_sync(0xffffffffffffffffull, t, off));
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = (int) ((blockDim.x + 31) >> 5);
    if ((int) threadIdx.x < 32) {
        float t = (int) threadIdx.x < nw ? red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffffffffffull, t, off);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

__global__ void split_q_kernel(const float* __restrict__ q, float* __restrict__ q_nope, float* __restrict__ q_pe,
                               int n_head, int nope, int n_rot) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    const int head_dim = nope + n_rot;
    if (i >= n_head * head_dim) return;
    const int h = i / head_dim, d = i % head_dim;
    if (d < nope) q_nope[(size_t) h * nope + d] = q[i];
    else q_pe[(size_t) h * n_rot + (d - nope)] = q[i];
}

__global__ void rope_kernel(const float* __restrict__ x, float* __restrict__ out, int rows, int n_rot,
                            float theta_scale, const int* __restrict__ pos) {
    const int row = blockIdx.y;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = n_rot / 2;
    if (row >= rows || pair >= half) return;
    const size_t start = (size_t) row * n_rot;
    const float theta = (float) pos[0] * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = x[start + pair], b = x[start + pair + half];
    out[start + pair] = a * c - b * s;
    out[start + pair + half] = a * s + b * c;
}

__global__ void write_kv_kernel(uint16_t* __restrict__ kv, uint16_t* __restrict__ v, const float* __restrict__ latent,
                                const float* __restrict__ k_pe, int n_lora_kv, int n_rot) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    const int width = n_lora_kv + n_rot;
    if (i >= width) return;
    if (i < n_lora_kv) {
        const uint16_t h = f16_from_f32(latent[i]);
        kv[i] = h;
        v[i] = h;
    } else {
        kv[i] = f16_from_f32(k_pe[i - n_lora_kv]);
    }
}

__global__ void attention_kernel(const float* __restrict__ q_abs, const float* __restrict__ q_pe,
                                 const uint16_t* __restrict__ kv, const uint16_t* __restrict__ v, int n_cells,
                                 int n_lora_kv, int n_rot, float scale, float* __restrict__ scores,
                                 float* __restrict__ attn) {
    __shared__ float red[MLA_THREADS / 32];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int width = n_lora_kv + n_rot;
    const float* qa = q_abs + (size_t) h * n_lora_kv;
    const float* qp = q_pe + (size_t) h * n_rot;
    float* sc = scores + (size_t) h * n_cells;

    for (int j = tid; j < n_cells; j += nt) {
        const uint16_t* k = kv + (size_t) j * width;
        float acc = 0.0f;
        for (int d = 0; d < n_lora_kv; ++d) acc += qa[d] * __half2float(*(const __half*) (k + d));
        for (int d = 0; d < n_rot; ++d) acc += qp[d] * __half2float(*(const __half*) (k + n_lora_kv + d));
        sc[j] = acc * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = tid; j < n_cells; j += nt) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int j = tid; j < n_cells; j += nt) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = tid; j < n_cells; j += nt) sc[j] *= inv;
    __syncthreads();

    float* out = attn + (size_t) h * n_lora_kv;
    for (int d = tid; d < n_lora_kv; d += nt) {
        float acc = 0.0f;
        for (int j = 0; j < n_cells; ++j)
            acc += sc[j] * __half2float(*(const __half*) (v + (size_t) j * n_lora_kv + d));
        out[d] = acc;
    }
}

}  // namespace

void mla_split_q(const float* q, float* q_nope, float* q_pe, int64_t n_head, int64_t nope, int64_t n_rot,
                 void* stream) {
    require(q && q_nope && q_pe, "split_q needs non-null pointers");
    require(n_head > 0 && nope > 0 && n_rot > 0, "split_q needs positive dims");
    const int total = (int) (n_head * (nope + n_rot));
    const unsigned blocks = (unsigned) ((total + MLA_THREADS - 1) / MLA_THREADS);
    split_q_kernel<<<blocks, MLA_THREADS, 0, (hipStream_t) stream>>>(q, q_nope, q_pe, (int) n_head, (int) nope,
                                                                    (int) n_rot);
    check_launch();
}

void mla_rope(const float* x, float* out, int64_t rows, int64_t n_rot, float freq_base, const int32_t* pos,
              void* stream) {
    require(x && out && pos, "rope needs non-null pointers");
    require(rows > 0 && n_rot > 0 && n_rot % 2 == 0, "rope needs rows > 0 and an even n_rot");
    require(freq_base > 1.0f && std::isfinite(freq_base), "rope needs a finite base > 1");
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    const int half = (int) (n_rot / 2);
    const dim3 grid((unsigned) ((half + 127) / 128), (unsigned) rows);
    rope_kernel<<<grid, 128, 0, (hipStream_t) stream>>>(x, out, (int) rows, (int) n_rot, theta_scale,
                                                        (const int*) pos);
    check_launch();
}

void mla_write_kv(uint16_t* kv, uint16_t* v, const float* latent, const float* k_pe, int64_t pos, int64_t n_lora_kv,
                  int64_t n_rot, void* stream) {
    require(kv && v && latent && k_pe, "write_kv needs non-null pointers");
    require(pos >= 0 && n_lora_kv > 0 && n_rot > 0, "write_kv needs a nonnegative pos and positive dims");
    const int width = (int) (n_lora_kv + n_rot);
    const unsigned blocks = (unsigned) ((width + MLA_THREADS - 1) / MLA_THREADS);
    write_kv_kernel<<<blocks, MLA_THREADS, 0, (hipStream_t) stream>>>(kv + pos * (size_t) width,
                                                                    v + pos * (size_t) n_lora_kv, latent, k_pe,
                                                                    (int) n_lora_kv, (int) n_rot);
    check_launch();
}

void mla_attention(const float* q_abs, const float* q_pe, const uint16_t* kv, const uint16_t* v, int64_t n_cells,
                   int64_t n_head, int64_t n_lora_kv, int64_t n_rot, float scale, float* scores, float* attn,
                   void* stream) {
    require(q_abs && q_pe && kv && v && scores && attn, "attention needs non-null pointers");
    require(n_cells > 0 && n_head > 0 && n_lora_kv > 0 && n_rot > 0, "attention needs positive dims");
    require(scale > 0.0f && std::isfinite(scale), "attention needs a positive finite scale");
    attention_kernel<<<(unsigned) n_head, MLA_THREADS, 0, (hipStream_t) stream>>>(
        q_abs, q_pe, kv, v, (int) n_cells, (int) n_lora_kv, (int) n_rot, scale, scores, attn);
    check_launch();
}

}  // namespace strata::kernels

