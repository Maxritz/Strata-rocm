// bench/micro/fp4_repack_parity.cu - K3 acceptance: the 16-byte repack is a LOSSLESS relayout.
//
// Decodes every block from the ggml layout (17 B MXFP4 / 36 B NVFP4, odd strides) and from the repacked
// scale/data arrays, on the device, and requires bit-identical floats.  If the repack lost a byte this fails.
//
//   hipcc -O3 --offload-arch=gfx1201 -o fp4_repack_parity fp4_repack_parity.cu
#include "hip/hip_runtime.h"
#include "strata/kernels/fp4_repack.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::kernels::fp4_codebook;
using strata::kernels::fp4_e8m0_to_fp32;
using strata::kernels::fp4_ue4m3_to_fp32;

__global__ void mxfp4_check(const uint8_t* __restrict__ orig, const uint8_t* __restrict__ scales,
                            const uint8_t* __restrict__ data, size_t nblocks, int* bad) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t* blk = orig + i * 17;
    const float d0 = fp4_e8m0_to_fp32(blk[0]);
    const float d1 = fp4_e8m0_to_fp32(scales[i]);
    const int8_t* k = fp4_codebook();
    for (int j = 0; j < 16; ++j) {
        const uint8_t o = blk[1 + j], r = data[i * 16 + j];
        if (k[o & 0x0F] * d0 != k[r & 0x0F] * d1) atomicAdd(bad, 1);
        if (k[o >> 4]   * d0 != k[r >> 4]   * d1) atomicAdd(bad, 1);
    }
}

__global__ void nvfp4_check(const uint8_t* __restrict__ orig, const uint8_t* __restrict__ scales,
                            const uint8_t* __restrict__ data, size_t nblocks, int* bad) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t* blk = orig + i * 36;
    const int8_t* k = fp4_codebook();
    const uint8_t* sc0 = blk, *sc1 = scales + i * 4;
    const uint8_t* q0 = blk + 4, *q1 = data + i * 32;
    for (int s = 0; s < 4; ++s) {
        const float a = fp4_ue4m3_to_fp32(sc0[s]), b = fp4_ue4m3_to_fp32(sc1[s]);
        for (int j = 0; j < 8; ++j) {
            const uint8_t o = q0[s * 8 + j], r = q1[s * 8 + j];
            if (k[o & 0x0F] * a != k[r & 0x0F] * b) atomicAdd(bad, 1);
            if (k[o >> 4]   * a != k[r >> 4]   * b) atomicAdd(bad, 1);
        }
    }
}

static void* up(const std::vector<uint8_t>& v) {
    void* d = nullptr;
    hipMalloc(&d, v.size());
    hipMemcpy(d, v.data(), v.size(), hipMemcpyHostToDevice);
    return d;
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? (size_t) std::atoll(argv[1]) : (size_t) 1 << 20;
    int bad = 0, *d_bad = nullptr;
    hipMalloc(&d_bad, sizeof(int));
    hipMemset(d_bad, 0, sizeof(int));

    std::vector<uint8_t> mx(n * 17), nv(n * 36);
    for (auto& x : mx) x = (uint8_t) (rand() & 0xFF);
    for (auto& x : nv) x = (uint8_t) (rand() & 0xFF);
    // A scale byte of 0xFF (E8M0 -> 2^127) or 0x7F (UE4M3 NaN) is legal and must survive; rand() covers them.

    const auto rx = strata::kernels::repack_mxfp4(mx.data(), n);
    const auto rn = strata::kernels::repack_nvfp4(nv.data(), n);

    void* dmx = up(mx); void* dmxs = up(rx.scales); void* dmxd = up(rx.data);
    void* dnv = up(nv); void* dnvs = up(rn.scales); void* dnvd = up(rn.data);

    const int threads = 256;
    const int blocks = (int) ((n + threads - 1) / threads);
    mxfp4_check<<<blocks, threads>>>((const uint8_t*) dmx, (const uint8_t*) dmxs, (const uint8_t*) dmxd, n, d_bad);
    nvfp4_check<<<blocks, threads>>>((const uint8_t*) dnv, (const uint8_t*) dnvs, (const uint8_t*) dnvd, n, d_bad);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { std::fprintf(stderr, "kernel: %s\n", hipGetErrorString(e)); return 2; }
    hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost);

    const bool aligned = (reinterpret_cast<uintptr_t>(rx.data.data()) & 15) == 0 &&
                         (reinterpret_cast<uintptr_t>(rn.data.data()) & 15) == 0;
    std::printf("fp4 repack: %zu MXFP4 + %zu NVFP4 blocks, %d wrong values, data 16B-aligned %s\n",
                n, n, bad, aligned ? "yes" : "NO");
    std::printf("%s\n", (bad == 0 && aligned) ? "PASS" : "FAIL");
    return (bad == 0 && aligned) ? 0 : 1;
}
