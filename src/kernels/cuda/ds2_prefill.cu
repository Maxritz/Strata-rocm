#include "hip/hip_runtime.h"
// src/kernels/cuda/ds2_prefill.cu - see include/strata/kernels/ds2_prefill.hpp.  Row-wise generalisations of
// the single-token MLA kernels in mla.cu; T = 1 reproduces them exactly.
#include "strata/kernels/ds2_prefill.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels::ds2pf {
namespace {

constexpr int NTHREADS = 256;

void check() {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("ds2_prefill launch: ") + hipGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffffffffffull, v, o);
    return v;
}

__global__ void rope_slice_kernel(const float* __restrict__ x, float* __restrict__ out, long rows, long in_stride,
                                  long out_stride, long rot_off, int n_rot, float theta_scale, long pos0,
                                  long heads_per_pos) {
    const long r = blockIdx.y;
    const int half = n_rot / 2;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || pair >= half) return;
    const long t = r / heads_per_pos;
    const float theta = (float) (pos0 + t) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float* xr = x + r * in_stride + rot_off;
    float* orow = out + r * out_stride;
    const float a = xr[pair], b = xr[pair + half];
    orow[pair] = a * c - b * s;
    orow[pair + half] = a * s + b * c;
}

__global__ void gather_qnope_kernel(const float* __restrict__ q, float* __restrict__ qnope, long T, int n_head,
                                    int nope, int head_dim) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n_head * nope) return;
    const int h = (int) (i / ((long) T * nope));
    const long rem = i % ((long) T * nope);
    const long t = rem / nope;
    const int d = (int) (rem % nope);
    qnope[((long) h * T + t) * nope + d] = q[(t * n_head + h) * head_dim + d];
}

__global__ void rms_strided_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                   float* __restrict__ out, long rows, long cols, long stride, float eps) {
    const long row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* base = in + row * stride;
    float acc = 0.0f;
    for (long c = lane; c < cols; c += 32) acc += base[c] * base[c];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffffffffffull, acc, o);
    float inv = 0.0f;
    if (lane == 0) inv = rsqrtf(acc / (float) cols + eps);
    inv = __shfl_sync(0xffffffffffffffffull, inv, 0);
    float* o = out + row * cols;
    for (long c = lane; c < cols; c += 32) o[c] = base[c] * w[c] * inv;
}

__global__ void write_kv_kernel(uint16_t* __restrict__ kv, uint16_t* __restrict__ v, const float* __restrict__ latent,
                                const float* __restrict__ k_pe, long pos0, long T, int n_lora_kv, int n_rot) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    const int width = n_lora_kv + n_rot;
    if (i >= T * width) return;
    const long t = i / width;
    const int c = (int) (i % width);
    uint16_t* krow = kv + (pos0 + t) * width;
    if (c < n_lora_kv) {
        const uint16_t h = f16_from_f32(latent[t * n_lora_kv + c]);
        krow[c] = h;
        v[(pos0 + t) * n_lora_kv + c] = h;
    } else {
        krow[c] = f16_from_f32(k_pe[t * n_rot + (c - n_lora_kv)]);
    }
}

__global__ void attention_kernel(const float* __restrict__ q_abs, const float* __restrict__ q_pe,
                                 const uint16_t* __restrict__ kv, const uint16_t* __restrict__ v, long pos0,
                                 int n_head, int n_lora_kv, int n_rot, float scale, float* __restrict__ attn) {
    extern __shared__ float sc[];
    __shared__ float red[NTHREADS / 32];
    const int h = blockIdx.x;
    const long t = blockIdx.y;
    const int n_cells = (int) (pos0 + t + 1);
    const int tid = threadIdx.x, nt = blockDim.x;
    const int width = n_lora_kv + n_rot;
    const float* qa = q_abs + ((long) t * n_head + h) * n_lora_kv;
    const float* qp = q_pe + ((long) t * n_head + h) * n_rot;
    for (int j = tid; j < n_cells; j += nt) {
        const uint16_t* k = kv + (long) j * width;
        float acc = 0.0f;
        for (int d = 0; d < n_lora_kv; ++d) acc += qa[d] * __half2float(*(const __half*) (k + d));
        for (int d = 0; d < n_rot; ++d) acc += qp[d] * __half2float(*(const __half*) (k + n_lora_kv + d));
        sc[j] = acc * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = tid; j < n_cells; j += nt) mx = fmaxf(mx, sc[j]);
    {
        const int lane = tid & 31, warp = tid >> 5;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffull, mx, o));
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        const int nw = (int) ((blockDim.x + 31) >> 5);
        if (tid < 32) {
            float tt = tid < nw ? red[tid] : -INFINITY;
            for (int o = 16; o > 0; o >>= 1) tt = fmaxf(tt, __shfl_xor_sync(0xffffffffffffffffull, tt, o));
            if (tid == 0) red[0] = tt;
        }
        __syncthreads();
        mx = red[0];
        __syncthreads();
    }
    float sum = 0.0f;
    for (int j = tid; j < n_cells; j += nt) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    {
        const int lane = tid & 31, warp = tid >> 5;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffffffffffull, sum, o);
        if (lane == 0) red[warp] = sum;
        __syncthreads();
        const int nw = (int) ((blockDim.x + 31) >> 5);
        if (tid < 32) {
            float tt = tid < nw ? red[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) tt += __shfl_xor_sync(0xffffffffffffffffull, tt, o);
            if (tid == 0) red[0] = tt;
        }
        __syncthreads();
        sum = red[0];
        __syncthreads();
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int j = tid; j < n_cells; j += nt) sc[j] *= inv;
    __syncthreads();
    float* out = attn + ((long) t * n_head + h) * n_lora_kv;
    for (int d = tid; d < n_lora_kv; d += nt) {
        float acc = 0.0f;
        for (int j = 0; j < n_cells; ++j) acc += sc[j] * __half2float(*(const __half*) (v + (long) j * n_lora_kv + d));
        out[d] = acc;
    }
}

__global__ void combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                               const float* __restrict__ shared, float* __restrict__ out, long T, int k,
                               int n_embd) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n_embd) return;
    const long t = i / n_embd;
    const int d = (int) (i % n_embd);
    float acc = shared[i];
    for (int j = 0; j < k; ++j) acc += weights[t * k + j] * parts[((long) t * k + j) * n_embd + d];
    out[i] = acc;
}

__global__ void gather_rows_kernel(const float* __restrict__ x, float* __restrict__ out, long rows, long cols,
                                   long in_stride) {
    const long r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    out[r * cols + c] = x[r * in_stride + c];
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

int64_t attention_scratch_bytes() { return 0; }

void rope_slice(const float* x, float* out, int64_t rows, int64_t in_stride, int64_t out_stride, int64_t rot_off,
                int64_t n_rot, float freq_base, int64_t pos0, int64_t heads_per_pos, void* stream) {
    if (rows <= 0) return;
    if (!x || !out || n_rot <= 0 || n_rot % 2 != 0 || heads_per_pos <= 0) {
        throw std::invalid_argument("ds2pf::rope_slice: bad arguments");
    }
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    const int half = (int) (n_rot / 2);
    const dim3 grid((unsigned) ceil_div(half, 128), (unsigned) rows);
    rope_slice_kernel<<<grid, 128, 0, (hipStream_t) stream>>>(x, out, (long) rows, (long) in_stride,
                                                              (long) out_stride, (long) rot_off, (int) n_rot,
                                                              theta_scale, (long) pos0, (long) heads_per_pos);
    check();
}

void gather_qnope(const float* q, float* qnope, int64_t T, int64_t n_head, int64_t nope, int64_t head_dim,
                  void* stream) {
    if (T <= 0) return;
    const long n = (long) (T * n_head * nope);
    gather_qnope_kernel<<<(unsigned) ceil_div(n, NTHREADS), NTHREADS, 0, (hipStream_t) stream>>>(
        q, qnope, (long) T, (int) n_head, (int) nope, (int) head_dim);
    check();
}

void rms_strided(const float* in, const float* w, float* out, int64_t rows, int64_t cols, int64_t stride,
                 float eps, void* stream) {
    if (rows <= 0) return;
    rms_strided_kernel<<<(unsigned) rows, 32, 0, (hipStream_t) stream>>>(in, w, out, (long) rows, (long) cols,
                                                                         (long) stride, eps);
    check();
}

void write_kv(uint16_t* kv, uint16_t* v, const float* latent, const float* k_pe, int64_t pos0, int64_t T,
              int64_t n_lora_kv, int64_t n_rot, void* stream) {
    if (T <= 0) return;
    const long n = (long) (T * (n_lora_kv + n_rot));
    write_kv_kernel<<<(unsigned) ceil_div(n, NTHREADS), NTHREADS, 0, (hipStream_t) stream>>>(
        kv, v, latent, k_pe, (long) pos0, (long) T, (int) n_lora_kv, (int) n_rot);
    check();
}

void attention(const float* q_abs, const float* q_pe, const uint16_t* kv, const uint16_t* v, int64_t pos0,
               int64_t T, int64_t n_head, int64_t n_lora_kv, int64_t n_rot, float scale, float* attn,
               void* stream) {
    if (T <= 0) return;
    if (pos0 < 0 || n_head <= 0 || n_lora_kv <= 0 || n_rot <= 0) throw std::invalid_argument("ds2pf::attention");
    const size_t shbytes = (size_t) (pos0 + T) * sizeof(float);
    const dim3 grid((unsigned) n_head, (unsigned) T);
    attention_kernel<<<grid, NTHREADS, shbytes, (hipStream_t) stream>>>(
        q_abs, q_pe, kv, v, (long) pos0, (int) n_head, (int) n_lora_kv, (int) n_rot, scale, attn);
    check();
}

void combine(const float* parts, const float* weights, const float* shared, float* out, int64_t T, int64_t k,
             int64_t n_embd, void* stream) {
    if (T <= 0) return;
    const long n = (long) (T * n_embd);
    combine_kernel<<<(unsigned) ceil_div(n, NTHREADS), NTHREADS, 0, (hipStream_t) stream>>>(
        parts, weights, shared, out, (long) T, (int) k, (int) n_embd);
    check();
}

void gather_rows(const float* x, float* out, int64_t rows, int64_t cols, int64_t in_stride, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    const dim3 grid((unsigned) ceil_div(cols, NTHREADS), (unsigned) rows);
    gather_rows_kernel<<<grid, NTHREADS, 0, (hipStream_t) stream>>>(x, out, (long) rows, (long) cols, (long) in_stride);
    check();
}

}  // namespace strata::kernels::ds2pf
