// bench/micro/wmma_iu4_gemm.cu - K1b: tiled int4 WMMA GEMM, parity + TOPS (step 2 of the contract's K1).
//
// Same fragment map as K1a (RDNA4 ISA 7.12.2).  One wave computes a 16x16 output tile, looping K in steps of
// 16 with the A/B fragments built from a packed int4 (2 codes/byte, row-major) global buffer = the MoE expert
// weight layout.  Reports parity vs a scalar int4 reference AND the achieved T-MAC/s.
//
// Expert shape (Swift gate/up, one batch): M = tokens, N = 2*n_ff = 1280, K = n_embd = 2560.
//
//   hipcc -O3 --offload-arch=gfx1201 -o wmma_iu4_gemm wmma_iu4_gemm.cu
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

typedef __attribute__((ext_vector_type(8))) int32_t i32x8;

__global__ void iu4_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int* __restrict__ C,
                                int M, int N, int K) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const int tiles_n = N >> 4;
    if (wave >= (M >> 4) * tiles_n) return;
    const int m0 = (wave / tiles_n) << 4;
    const int n0 = (wave % tiles_n) << 4;

    const int a_row = m0 + (lane & 15);            // A fragment: lane L holds A[row][ (L/16)*8 + i ]
    const int b_col = n0 + (lane & 15);            // B fragment: lane L holds B[ (L/16)*8 + i ][col]
    const int half = lane >> 4;                    // 0 or 1 -> the 8-column group

    i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        int a = 0, b = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int ak = k0 + half * 8 + i;       // A column
            const uint8_t ab = A[(long long) a_row * (K / 2) + (ak >> 1)];
            const int acode = (ak & 1) ? (ab >> 4) : (ab & 0xF);
            a |= (acode & 0xF) << (4 * i);

            const int bk = k0 + half * 8 + i;       // B row
            const uint8_t bb = B[(long long) bk * (N / 2) + (b_col >> 1)];
            const int bcode = (b_col & 1) ? (bb >> 4) : (bb & 0xF);
            b |= (bcode & 0xF) << (4 * i);
        }
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c, false);
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) C[(long long) (m0 + half * 8 + v) * N + b_col] = c[v];
}

void check(const char* what, hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}
int s4(uint8_t code) { return (code & 0x8) ? (int) code - 16 : (int) code; }   // signed int4

}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 512;
    const int N = argc > 2 ? std::atoi(argv[2]) : 1280;
    const int K = argc > 3 ? std::atoi(argv[3]) : 2560;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 50;

    std::mt19937 rng(1234);
    std::vector<uint8_t> A((size_t) M * K / 2), B((size_t) K * N / 2);
    for (auto& x : A) x = (uint8_t) (rng() & 0xFF);
    for (auto& x : B) x = (uint8_t) (rng() & 0xFF);

    uint8_t *dA = nullptr, *dB = nullptr;
    int* dC = nullptr;
    check("malloc", hipMalloc(&dA, A.size()));
    check("malloc", hipMalloc(&dB, B.size()));
    check("malloc", hipMalloc(&dC, (size_t) M * N * 4));
    check("copy", hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    check("copy", hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));

    const int waves = (M / 16) * (N / 16);
    const int threads = 256;
    const int blocks = (waves * 32 + threads - 1) / threads;
    iu4_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    check("launch", hipGetLastError());
    check("sync", hipDeviceSynchronize());
    std::vector<int> C((size_t) M * N);
    check("copy", hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));

    // parity: a sparse sample of the output (16x16 blocks at a few positions) vs a scalar int4 dot.
    int bad = 0, worst = 0;
    for (int m = 0; m < M; m += 97)
        for (int n = 0; n < N; n += 89) {
            long long s = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t ab = A[(long long) m * (K / 2) + (k >> 1)];
                const uint8_t bb = B[(long long) k * (N / 2) + (n >> 1)];
                s += (long long) s4((k & 1) ? (ab >> 4) : (ab & 0xF)) * s4((n & 1) ? (bb >> 4) : (bb & 0xF));
            }
            const int got = C[(long long) m * N + n];
            const int diff = (int) std::abs((long long) got - s);
            if (diff > worst) worst = diff;
            if (diff != 0) ++bad;
        }

    hipEvent_t t0, t1;
    check("ev", hipEventCreate(&t0)); check("ev", hipEventCreate(&t1));
    iu4_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);   // warmup
    check("sync", hipDeviceSynchronize());
    check("rec", hipEventRecord(t0, 0));
    for (int i = 0; i < iters; ++i) iu4_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    check("rec", hipEventRecord(t1, 0));
    check("sync", hipEventSynchronize(t1));
    float ms = 0.0f;
    check("elapsed", hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    const double mmas = (double) (M / 16) * (N / 16) * (K / 16);
    const double tops = mmas * 4096.0 / (ms * 1e-3) / 1e12;
    std::printf("iu4 GEMM M=%d N=%d K=%d: %s (%d sampled wrong, worst=%d)  %.3f ms  %.1f T-MAC/s\n",
                M, N, K, bad ? "FAIL" : "PASS", bad, worst, ms, tops);
    check("free", hipFree(dA)); check("free", hipFree(dB)); check("free", hipFree(dC));
    return bad ? 1 : 0;
}
