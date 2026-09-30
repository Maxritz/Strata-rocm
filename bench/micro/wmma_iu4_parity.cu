// bench/micro/wmma_iu4_parity.cu - K1a: the int4 tensor-core MMA, parity vs a scalar reference.
//
// Locks the two things the WMMA expert GEMM depends on, before any tiling:
//   1. the FRAGMENT LAYOUT (from the RDNA4 ISA section 7.12.2, V_WMMA_I32_16X16X16_IU4), and
//   2. the SIGNED-int4 arithmetic + the accumulate contract.
//
// ISA layout (wave32, dataSize=4):
//   A (16x16 MxK): lane = { col[3], row[3:0] }, vgpr 0, startPosn = col[2:0]
//                  -> lane L holds A[L%16][(L/16)*8 + i] at nibble i (i=0..7)
//   B (16x16 KxN): lane = { row[3], col[3:0] }, vgpr 0, startPosn = row[2:0]
//                  -> lane L holds B[(L/16)*8 + i][L%16] at nibble i
//   C/D (16x16 I32): lane = { row[3], col[3:0] }, vgpr = row[2:0]
//                  -> lane L, vgpr v holds C[(L/16)*8 + v][L%16]
//
//   hipcc -O3 --offload-arch=gfx1201 -o wmma_iu4_parity wmma_iu4_parity.cu
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

typedef __attribute__((ext_vector_type(8))) int32_t i32x8;

__global__ void iu4_mma_kernel(const int* __restrict__ a, const int* __restrict__ b, int* __restrict__ c) {
    const int lane = threadIdx.x & 31;
    const int32_t av = a[lane];
    const int32_t bv = b[lane];
    i32x8 cv = {0, 0, 0, 0, 0, 0, 0, 0};
    // A signed, B signed for IU4 (NEG[0]=A, NEG[1]=B), clamp=false.
    cv = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, av, true, bv, cv, false);
#pragma unroll
    for (int v = 0; v < 8; ++v) c[lane * 8 + v] = cv[v];
}

void check(const char* what, hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}

// A[row][col] -> lane L (row = L%16, base = (L/16)*8), nibble i = col - base.
void pack_a(const int8_t A[16][16], int* out) {
    for (int L = 0; L < 32; ++L) {
        const int row = L % 16, base = (L / 16) * 8;
        int32_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (int32_t) (A[row][base + i] & 0xF) << (4 * i);
        out[L] = v;
    }
}

// B[k][n] -> lane L (n = L%16, base = (L/16)*8), nibble i = k - base.
void pack_b(const int8_t B[16][16], int* out) {
    for (int L = 0; L < 32; ++L) {
        const int n = L % 16, base = (L / 16) * 8;
        int32_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (int32_t) (B[base + i][n] & 0xF) << (4 * i);
        out[L] = v;
    }
}

}  // namespace

int main() {
    int8_t A[16][16], B[16][16];
    int32_t ref[16][16];
    // non-trivial signed int4 values, so the dot exercises negatives and carries.
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 16; ++k) A[m][k] = (int8_t) ((m * 7 + k * 3) % 16 - 8);   // -8..7
    for (int k = 0; k < 16; ++k)
        for (int n = 0; n < 16; ++n) B[k][n] = (int8_t) ((k * 5 + n * 11) % 16 - 8);
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            int32_t s = 0;
            for (int k = 0; k < 16; ++k) s += (int32_t) A[m][k] * (int32_t) B[k][n];
            ref[m][n] = s;
        }

    int ha[32], hb[32], hc[32 * 8];
    pack_a(A, ha);
    pack_b(B, hb);

    int *da = nullptr, *db = nullptr, *dc = nullptr;
    check("malloc a", hipMalloc(&da, sizeof ha));
    check("malloc b", hipMalloc(&db, sizeof hb));
    check("malloc c", hipMalloc(&dc, sizeof hc));
    check("copy a", hipMemcpy(da, ha, sizeof ha, hipMemcpyHostToDevice));
    check("copy b", hipMemcpy(db, hb, sizeof hb, hipMemcpyHostToDevice));
    iu4_mma_kernel<<<1, 32>>>(da, db, dc);
    check("launch", hipGetLastError());
    check("sync", hipDeviceSynchronize());
    check("copy c", hipMemcpy(hc, dc, sizeof hc, hipMemcpyDeviceToHost));

    // unpack: C[(L/16)*8 + v][L%16] = c[L*8 + v]
    int bad = 0, worst = 0;
    for (int L = 0; L < 32; ++L)
        for (int v = 0; v < 8; ++v) {
            const int m = (L / 16) * 8 + v, n = L % 16;
            const int32_t got = hc[L * 8 + v];
            if (got != ref[m][n]) {
                if (bad < 8) std::printf("  MISMATCH C[%d][%d]: ref=%d got=%d\n", m, n, ref[m][n], got);
                ++bad;
            }
            const int d = std::abs(got - ref[m][n]);
            if (d > worst) worst = d;
        }
    std::printf("iu4 16x16x16 MMA parity: %s (%d/256 wrong, worst |diff|=%d)\n", bad ? "FAIL" : "PASS", bad, worst);
    check("free", hipFree(da)); check("free", hipFree(db)); check("free", hipFree(dc));
    return bad ? 1 : 0;
}
