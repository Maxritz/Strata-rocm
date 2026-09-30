// bench/micro/wmma_iu4_gemm_lds.cu - K1b (packed-dword fragments): int4 WMMA GEMM, parity + TOPS.
//
// Root cause of the 5-16 T-MAC/s plateau (found by disassembly, not guessed): the earlier versions stored the
// tiles UNPACKED as int8 in LDS -- staging paid byte->nibble, then every fragment build paid nibble->packed
// (~128 ALU ops per MMA).  Here the tiles live in LDS PACKED (int4), and each MMA fragment is a SINGLE
// ds_read_b32: the 8 nibbles the fragment wants are 8 consecutive codes in the row, i.e. 4 contiguous bytes.
//   * A is row-major packed: fragment = &As[row][ (kk+(lane>>4)*8)/2 ].
//   * B must be N-major (transposed K) so 8 consecutive k are contiguous: fragment = &Bs[n][ same offset ].
// N-major packed B is the native weight layout for the real path (K3 16B repack), so this is representative.
// Staging is now a pure 128-bit copy - no unpacking at all.
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

constexpr int BM = 64, BN = 64, BK = 64, THREADS = 128;   // 4 waves (16 rows each); each wave does 16x64 = 4x 16x16
constexpr int BK2 = BK / 2;

__global__ void iu4_gemm_lds_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                                    int* __restrict__ C, int M, int N, int K) {
    __shared__ alignas(16) uint8_t As[BM][BK2];   // packed int4, row-major  (K-major)
    __shared__ alignas(16) uint8_t Bs[BN][BK2];   // packed int4, N-major    (transposed K - fragment-contiguous)
    const int tid = threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    i32x8 c[4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (tid < BM) {                                       // A row: BK/2 = 32 B = 2 uint4, copied verbatim
            const uint4* ap = reinterpret_cast<const uint4*>(A + (long long) (m0 + tid) * (K / 2) + (k0 >> 1));
            *reinterpret_cast<uint4*>(&As[tid][0]) = ap[0];
            *reinterpret_cast<uint4*>(&As[tid][16]) = ap[1];
        }
        {                                                      // Bt row: 128 threads x 1 uint4 (64 rows x 2 segs)
            const int r = tid >> 1, seg = tid & 1;
            const uint4 v = *reinterpret_cast<const uint4*>(Bt + (long long) (n0 + r) * (K / 2) + (k0 >> 1) + seg * 16);
            *reinterpret_cast<uint4*>(&Bs[r][seg * 16]) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const int coff = (kk + (lane >> 4) * 8) >> 1;      // 8 nibbles = 4 contiguous bytes = one dword
            const uint32_t a = *reinterpret_cast<const uint32_t*>(&As[wave * 16 + (lane & 15)][coff]);
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const uint32_t b = *reinterpret_cast<const uint32_t*>(&Bs[sub * 16 + (lane & 15)][coff]);
                c[sub] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, (int32_t) a, true, (int32_t) b, c[sub], false);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int sub = 0; sub < 4; ++sub)
#pragma unroll
        for (int v = 0; v < 8; ++v)
            C[(long long) (m0 + wave * 16 + (lane >> 4) * 8 + v) * N + (n0 + sub * 16 + (lane & 15))] = c[sub][v];
}

void check(const char* what, hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}
int s4(uint8_t code) { return (code & 0x8) ? (int) code - 16 : (int) code; }
uint8_t code_at(const uint8_t* p, int idx) { return (idx & 1) ? (p[idx >> 1] >> 4) : (p[idx >> 1] & 0xF); }

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
    std::vector<uint8_t> Bt((size_t) N * K / 2);                     // N-major packed (transpose of B)
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const uint8_t c = code_at(B.data(), (long long) k * N + n);
            uint8_t& d = Bt[(long long) n * (K / 2) + (k >> 1)];
            if (k & 1) d |= (uint8_t) (c << 4); else d = (uint8_t) (c & 0xF);
        }

    uint8_t *dA = nullptr, *dBt = nullptr;
    int* dC = nullptr;
    check("malloc", hipMalloc(&dA, A.size()));
    check("malloc", hipMalloc(&dBt, Bt.size()));
    check("malloc", hipMalloc(&dC, (size_t) M * N * 4));
    check("copy", hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    check("copy", hipMemcpy(dBt, Bt.data(), Bt.size(), hipMemcpyHostToDevice));

    const dim3 grid(N / BN, M / BM);
    iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dBt, dC, M, N, K);
    check("launch", hipGetLastError());
    check("sync", hipDeviceSynchronize());
    std::vector<int> C((size_t) M * N);
    check("copy", hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));

    int bad = 0, worst = 0;
    for (int m = 0; m < M; m += 97)
        for (int n = 0; n < N; n += 89) {
            long long s = 0;
            for (int k = 0; k < K; ++k)
                s += (long long) s4(code_at(A.data(), (long long) m * K + k)) * s4(code_at(Bt.data(), (long long) n * K + k));
            const int diff = (int) std::abs((long long) C[(long long) m * N + n] - s);
            if (diff > worst) worst = diff;
            if (diff != 0) ++bad;
        }

    hipEvent_t t0, t1;
    check("ev", hipEventCreate(&t0)); check("ev", hipEventCreate(&t1));
    iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dBt, dC, M, N, K);
    check("sync", hipDeviceSynchronize());
    check("rec", hipEventRecord(t0, 0));
    for (int i = 0; i < iters; ++i) iu4_gemm_lds_kernel<<<grid, THREADS>>>(dA, dBt, dC, M, N, K);
    check("rec", hipEventRecord(t1, 0));
    check("sync", hipEventSynchronize(t1));
    float ms = 0.0f;
    check("elapsed", hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    const double mmas = (double) (M / 16) * (N / 16) * (K / 16);
    const double tops = mmas * 4096.0 / (ms * 1e-3) / 1e12;
    std::printf("iu4 LDS GEMM M=%d N=%d K=%d: %s (%d sampled wrong, worst=%d)  %.3f ms  %.1f T-MAC/s\n",
                M, N, K, bad ? "FAIL" : "PASS", bad, worst, ms, tops);
    check("free", hipFree(dA)); check("free", hipFree(dBt)); check("free", hipFree(dC));
    return bad ? 1 : 0;
}
