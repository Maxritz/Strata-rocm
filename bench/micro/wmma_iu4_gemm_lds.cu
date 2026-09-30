// bench/micro/wmma_iu4_gemm_lds.cu - K1b (fixed): LDS-staged int4 WMMA GEMM, parity + TOPS.
//
// The unstaged version read each fragment nibble from global (uncoalesced, ~10 GB/s) and was load-bound at
// 7.5-11 T-MAC/s.  Here a block stages BMxBK + BKxBN int4 tiles into shared with COALESCED loads, and every
// wave reads its fragments from LDS - the standard fix.  Same 16x16x16 iu4 fragment map as K1a.
//
//   hipcc -O3 --offload-arch=gfx1201 -o wmma_iu4_gemm_lds wmma_iu4_gemm_lds.cu
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

typedef __attribute__((ext_vector_type(8))) int32_t i32x8;

constexpr int BM = 64, BN = 64, BK = 32, THREADS = 512;   // 16 waves (4x4) per block

__global__ void iu4_gemm_lds_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                    int* __restrict__ C, int M, int N, int K) {
    __shared__ int8_t As[BM][BK];
    __shared__ int8_t Bs[BK][BN];
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const int wrow = wave >> 2, wcol = wave & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK, kk = idx - r * BK, k = k0 + kk;
            const uint8_t b = A[(long long) (m0 + r) * (K / 2) + (k >> 1)];
            As[r][kk] = (int8_t) ((k & 1) ? (b >> 4) : (b & 0xF));
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int kk = idx / BN, cc = idx - kk * BN, k = k0 + kk;
            const uint8_t b = B[(long long) k * (N / 2) + ((n0 + cc) >> 1)];
            Bs[kk][cc] = (int8_t) (((n0 + cc) & 1) ? (b >> 4) : (b & 0xF));
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            int a = 0, b = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a |= (As[wrow * 16 + (lane & 15)][kk + (lane >> 4) * 8 + i] & 0xF) << (4 * i);
                b |= (Bs[kk + (lane >> 4) * 8 + i][wcol * 16 + (lane & 15)] & 0xF) << (4 * i);
            }
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c, false);
        }
        __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < 8; ++v)
        C[(long long) (m0 + wrow * 16 + (lane >> 4) * 8 + v) * N + (n0 + wcol * 16 + (lane & 15))] = c[v];
}

void check(const char* what, hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}
int s4(uint8_t code) { return (code & 0x8) ? (int) code - 16 : (int) code; }

}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 512;
    const int N = argc > 2 ? std::atoi(argv[2]) : 1280;
    const int K = argc > 3 ? std::atoi(argv[3]) : 2560;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 50;
    if (M % BM || N % BN || K % BK) { std::fprintf(stderr, "dims must be multiples of %d/%d/%d\n", BM, BN, BK); return 2; }

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

    const dim3 grid(N / BN, M / BM);
    iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dB, dC, M, N, K);
    check("launch", hipGetLastError());
    check("sync", hipDeviceSynchronize());
    std::vector<int> C((size_t) M * N);
    check("copy", hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));

    int bad = 0, worst = 0;
    for (int m = 0; m < M; m += 97)
        for (int n = 0; n < N; n += 89) {
            long long s = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t ab = A[(long long) m * (K / 2) + (k >> 1)];
                const uint8_t bb = B[(long long) k * (N / 2) + (n >> 1)];
                s += (long long) s4((k & 1) ? (ab >> 4) : (ab & 0xF)) * s4((n & 1) ? (bb >> 4) : (bb & 0xF));
            }
            const int diff = (int) std::abs((long long) C[(long long) m * N + n] - s);
            if (diff > worst) worst = diff;
            if (diff != 0) ++bad;
        }

    hipEvent_t t0, t1;
    check("ev", hipEventCreate(&t0)); check("ev", hipEventCreate(&t1));
    iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dB, dC, M, N, K);
    check("sync", hipDeviceSynchronize());
    check("rec", hipEventRecord(t0, 0));
    for (int i = 0; i < iters; ++i) iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dB, dC, M, N, K);
    check("rec", hipEventRecord(t1, 0));
    check("sync", hipEventSynchronize(t1));
    float ms = 0.0f;
    check("elapsed", hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    const double mmas = (double) (M / 16) * (N / 16) * (K / 16);
    const double tops = mmas * 4096.0 / (ms * 1e-3) / 1e12;
    std::printf("iu4 LDS GEMM M=%d N=%d K=%d: %s (%d sampled wrong, worst=%d)  %.3f ms  %.1f T-MAC/s\n",
                M, N, K, bad ? "FAIL" : "PASS", bad, worst, ms, tops);
    check("free", hipFree(dA)); check("free", hipFree(dB)); check("free", hipFree(dC));
    return bad ? 1 : 0;
}
