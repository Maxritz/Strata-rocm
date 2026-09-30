// bench/micro/wmma_peak.cu - the gfx1201 tensor-core ceiling, measured (step 1 of the WMMA plan).
//
// A GEMV (batch 1) is bandwidth-bound and cannot use the matrix units.  A GEMM (batch B) reuses each weight
// block across B tokens and is compute-bound; THIS is the number that decides whether the 40-90 TOPS band the
// MoE prefill needs is real on RDNA4.  Measures fp16 (f32 accumulate) and int4 (iu4, i32 accumulate) 16x16x16
// WMMA throughput with 4 independent accumulators to expose ILP (a single dependent chain measures latency).
//
//   hipcc -O3 --offload-arch=gfx1201 -o wmma_peak wmma_peak.cu   (or via CMake: target wmma_peak)
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

typedef __attribute__((ext_vector_type(8))) _Float16 f16x8;
typedef __attribute__((ext_vector_type(8))) float f32x8;
typedef __attribute__((ext_vector_type(8))) int32_t i32x8;

// 16x16x16 per instruction = 4096 MACs = 8192 FLOPs.
constexpr double kMacsPerMma = 16.0 * 16.0 * 16.0;

__global__ void wmma_f16_kernel(float* __restrict__ out, long long iters) {
    f16x8 a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (_Float16) (i + 1);
        b[i] = (_Float16) (i + 2);
    }
    f32x8 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (long long it = 0; it < iters; ++it) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
    }
    const float s = c0[0] + c1[1] + c2[2] + c3[3];
    out[(long long) blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void wmma_iu4_kernel(int* __restrict__ out, long long iters) {
    const int32_t a = (int32_t) 0x32103210;   // 8 int4 codes packed, one per lane-slot
    const int32_t b = (int32_t) 0x32103210;
    i32x8 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (long long it = 0; it < iters; ++it) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c3, false);
    }
    const int s = c0[0] + c1[1] + c2[2] + c3[3];
    out[(long long) blockIdx.x * blockDim.x + threadIdx.x] = s;
}

void check(const char* what, hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}

template <typename K, typename P>
double run(const char* name, K kernel, P out, int blocks, int threads, long long iters) {
    // warmup
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, out, iters / 10);
    check("warmup", hipGetLastError());
    check("warmup sync", hipDeviceSynchronize());
    hipEvent_t t0, t1;
    check("ev", hipEventCreate(&t0));
    check("ev", hipEventCreate(&t1));
    check("rec", hipEventRecord(t0, 0));
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, out, iters);
    check(name, hipGetLastError());
    check("rec", hipEventRecord(t1, 0));
    check("sync", hipEventSynchronize(t1));
    float ms = 0.0f;
    check("elapsed", hipEventElapsedTime(&ms, t0, t1));
    const double waves = (double) blocks * (threads / 32.0);
    const double mmas = waves * (double) iters * 4.0;   // 4 MMA per iter
    const double tops = mmas * kMacsPerMma / (ms * 1e-3) / 1e12;
    std::printf("%-6s blocks=%d thr=%d iters=%lld  %.3f ms  %.1f T-MAC/s (TOPS)\n", name, blocks, threads, iters,
                ms, tops);
    hipEventDestroy(t0); hipEventDestroy(t1);
    return tops;
}

}  // namespace

int main() {
    int dev = 0;
    check("get", hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    check("prop", hipGetDeviceProperties(&prop, dev));
    int cus = 0;
    check("cus", hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));
    std::printf("device: %s  CUs=%d\n", prop.name, cus);

    const int threads = 256;
    const int blocks = cus * 8;           // ~8 waves/CU to hide latency
    const long long iters = 200000;
    float* d_f = nullptr;
    int* d_i = nullptr;
    check("malloc", hipMalloc(&d_f, (size_t) blocks * threads * 4));
    check("malloc", hipMalloc(&d_i, (size_t) blocks * threads * 4));

    std::printf("\n--- WMMA 16x16x16 throughput (4 independent accumulators) ---\n");
    run("f16", wmma_f16_kernel, d_f, blocks, threads, iters);
    run("iu4", wmma_iu4_kernel, d_i, blocks, threads, iters);
    std::printf("\n(16x16x16 iu4 = 4096 int4 MACs/instr; fp16 peak on RDNA4 ~48-97 TFLOP = 24-48 T-MAC/s)\n");

    check("free", hipFree(d_f));
    check("free", hipFree(d_i));
    return 0;
}
