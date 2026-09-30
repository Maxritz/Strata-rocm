# Strata ROCm Porting Guide (CUDA → HIP for RDNA2/RDNA4)

Port target: Strata-rocm engine from CUDA (sm_120, RTX 5090) to ROCm 10.1 (HIP 7.16) on Windows,
targeting gfx1031 (RX 6700 XT / RDNA2) and gfx1201 (RX 9070 XT / RDNA4).

## 1. Environment

| Runtime | Path | Contents | Status |
|---------|------|----------|--------|
| gfx1201 (RDNA4) | G:\ROCM10RT-gfx1201 | Full: HIP 7.16 + rocBLAS + hipBLAS + MIOpen + import .lib files | Ready |
| gfx1031 (RDNA2) | G:\ROCM10RT-gfx1031 | Minimal: mdhip64.lib, hiprtc.lib + DLLs in in\. Missing import libs **copied from gfx1201** | Ready (fixed) |
| Source tree | G:\rocm-10\rocm-libraries\projects | rocBLAS, hipBLAS, MIOpen, rocsolver source | Available if rebuild needed |
| HipKittens | C:\Users\rr\OneDrive\Desktop\AMDS\HipKittens | Official HIP port of ThunderKittens (834 files, CDNA-only) | Reference |
| Findings | C:\Users\rr\OneDrive\Desktop\dxl\knowledge\text\llama_findings_RDNA.md | AMD intrinsic coverage (confirms __dp4a→__builtin_amdgcn_sdot4) | Reference |

### gfx1031 import lib fix (DONE)
The gfx1031 runtime DLL directory (G:\ROCM10RT-gfx1031\bin\) has identical ocblas.dll
(21,053,952 bytes) to gfx1201, but the import .lib files were missing from G:\ROCM10RT-gfx1031\lib\.
Import libraries are architecture-agnostic symbol stubs — they reference DLL exports, not GPU code.
Copied from gfx1201:
`
  G:\ROCM10RT-gfx1201\lib\rocblas.lib  → G:\ROCM10RT-gfx1031\lib\rocblas.lib
  G:\ROCM10RT-gfx1201\lib\hipsolver.lib → G:\ROCM10RT-gfx1031\lib\hipsolver.lib
  G:\ROCM10RT-gfx1201\lib\hipblas.lib   → G:\ROCM10RT-gfx1031\lib\hipblas.lib
`
The .kpack files (las_lib_gfx1031.kpack, 43 MB etc.) in gfx1031 contain the gfx1031-specific
GPU kernels loaded by the DLLs at runtime — no rebuild needed.

## 2. Strata CUDA Dependencies (CMakeLists.txt)

Strata links these CUDA targets (see CMakeLists.txt:124, 251, 259, 263):
- CUDA::cudart — CUDA/HIP runtime (available as mdhip64.lib in both runtimes)
- CUDA::cublas — cuBLAS → rocBLAS (needs ocblas.lib — **copied for gfx1031**)
- CUDA language (nvcc → hipcc for .cu files)

## 3. CUDA → HIP Intrinsic Mapping

Verified against HIP 7.16 headers at G:\ROCM10RT-gfx1201\include\hip\amd_detail\.

### 3.1 DIRECT (hipify-clang handles automatically)

| CUDA | HIP | Header Location | Notes |
|------|-----|-----------------|-------|
| cudaMalloc/cudaFree | hipMalloc/hipFree | hip_runtime.h | |
| cudaMemcpy/Async/2DAsync | hipMemcpy/Async/2DAsync | hip_runtime.h | |
| cudaMemcpyToSymbol | hipMemcpyToSymbol | hip_runtime.h | |
| cudaMemset/Async | hipMemset/Async | hip_runtime.h | |
| cudaMemGetInfo | hipMemGetInfo | hip_runtime.h | |
| cudaHostAlloc/FreeHost | hipHostAlloc/FreeHost | hip_runtime.h | |
| cudaHostRegister/Unregister | hipHostRegister/Unregister | hip_runtime.h | |
| cudaHostGetDevicePointer | hipHostGetDevicePointer | hip_runtime.h | |
| cudaStreamCreate/Destroy/WithFlags | same hip* prefix | hip_runtime.h | hipStreamNonBlocking replaces cudaStreamNonBlocking |
| cudaStreamSynchronize/Query/WaitEvent | same hip* prefix | hip_runtime.h | |
| cudaStreamBeginCapture/EndCapture | same hip* prefix | hip_runtime.h | hipStreamCaptureModeThreadLocal |
| cudaEventCreate/Destroy/WithFlags/Record/Synchronize/Query/ElapsedTime | same hip* prefix | hip_runtime.h | hipEventDisableTiming |
| cudaDeviceSynchronize | hipDeviceSynchronize | hip_runtime.h | |
| cudaGetDevice/GetDeviceCount/SetDevice | same hip* prefix | hip_runtime.h | |
| cudaGetDeviceProperties → cudaDeviceProp | hipGetDeviceProperties → hipDeviceProp_t | hip_runtime.h | Use gcnArchName[256] for arch detection |
| cudaGetLastError/GetErrorString | same hip* prefix | hip_runtime.h | |
| cudaDriverGetVersion/RuntimeGetVersion | same hip* prefix | hip_runtime_api.h | |
| cudaGraphCreate | hipGraphCreate | hip_graph.h | |
| cudaGraphAddKernelNode | hipGraphAddKernelNode | hip_graph.h | |
| cudaGraphAddMemcpyNode/MemsetNode/EventRecordNode/EventWaitNode | same hip* prefix | hip_graph.h | |
| cudaGraphLaunch/Instantiate/GetNodes/AddDependencies | same hip* prefix | hip_graph.h | |
| cudaGraphExecDestroy/Destroy | same hip* prefix | hip_graph.h | |
| cublasCreate → cublasHandle_t | ocblas_create_handle → ocblas_handle | rocblas.h | |
| cublasDestroy | ocblas_destroy_handle | rocblas.h | |
| cublasSetStream | ocblas_set_stream | rocblas.h | |
| __shfl_down_sync(mask, var, delta) | same name, add , width=32 | md_warp_sync_functions.h:296 | AMD wavefront=64, width=32 keeps CUDA semantics |
| __shfl_xor_sync(mask, var, laneMask) | same name, add , width=32 | md_warp_sync_functions.h:306 | |
| __shfl_up_sync(mask, var, delta) | same name, add , width=32 | md_warp_sync_functions.h:286 | |
| __half22float2(h2) | same | md_hip_fp16.h:488 | Returns loat2 |
| __low2float(h2) | same | md_hip_fp16.h:486 | |
| __high2float(h2) | same | md_hip_fp16.h:487 | |
| __byte_perm(x, y, s) | same | md_device_functions.h:257 | |
| __float2bfloat16(f) | same | md_hip_bf16.h | |
| __bfloat162float(h) | same | md_hip_bf16.h | |
| __float2int_rn(f) | same | md_device_functions.h:480 | |
| __float2int_rz(f) | same | md_device_functions.h:482 | |
| __float2half_rn(f) | same | md_hip_fp16.h | |
| __float2half(f) | same | md_hip_fp16.h | |
| __half2float(h) | same | md_hip_fp16.h | |
| __half_as_ushort(h) | same | md_hip_fp16.h | |
| __int_as_float(x) | same | hip_runtime.h | |
| __float_as_uint(f) | same | hip_runtime.h | |
| __expf(x) | same | hip_device_functions.h | |
| __fmul_rn(a, b) | same | compiler builtin | **USER CONFIRMED available on RDNA2/RDNA4** |
| make_float2/make_float4/make_int2 | same | hip_runtime.h | |
| __global__/__device__/__shared__/__forceinline__/__restrict__ | same | hip_runtime.h | |
| <<<>>> kernel launch | same | hip_runtime.h | |
| __syncthreads() / __syncthreads_count() | same | hip_runtime.h | |
| __constant__ | same | hip_runtime.h | |
| #include <cuda_runtime.h> | #include <hip/hip_runtime.h> | header swap | |
| #include <cuda_fp16.h> | #include <hip/hip_fp16.h> | header swap | |
| #include <cuda_bf16.h> | #include <hip/hip_bf16.h> | header swap | |

### 3.2 ADAPT (hipify-clang handles most; manual enum/handle changes needed)

| CUDA | HIP | Manual Fixup | Source File |
|------|-----|--------------|-------------|
| cublasCreate(&handle) | ocblas_create_handle(&handle) | cublasHandle_t → ocblas_handle | gemm.cu |
| cublasDestroy(handle) | ocblas_destroy_handle(handle) | | gemm.cu |
| cublasSetStream(handle, stream) | ocblas_set_stream(handle, stream) | | gemm.cu |
| cublasGemmEx(...) | ocblas_gemm_ex(...) | CUBLAS_OP_T→ROCBLAS_OP_T, CUBLAS_OP_N→ROCBLAS_OP_N, CUDA_R_16F→ocblas_datatype_f16_r, CUDA_R_32F→ocblas_datatype_f32_r, CUDA_R_16BF→ocblas_datatype_bf16_r, CUBLAS_COMPUTE_32F→ocblas_compute_f32 | gemm.cu |
| cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) | omit | rocBLAS uses default math | gemm.cu |
| cublasSetWorkspace(h, ptr, bytes) | omit | rocBLAS manages workspace internally | gemm.cu |
| cudaDeviceProp.name | hipDeviceProp_t.gcnArchName[256] | AMD arch string (e.g. "gfx1031") | device.cu |
| cudaEventDisableTiming | hipEventDisableTiming | hipify handles | prefill.cpp |
| cudaStreamNonBlocking | hipStreamNonBlocking | hipify handles | pinned.cu |
| CUBLAS_STATUS_SUCCESS | ocblas_status_success | Error type changes | gemm.cu |

### 3.3 RETHINK (manual rewrite required — no HIP equivalent)

| CUDA Intrinsic | HIP Replacement | Files | Occurrences |
|---------------|-----------------|-------|-------------|
| __dp4a(a, b, c) | __builtin_amdgcn_sdot4((const char4&)a, (const char4&)b, c) | 
ative_mmoq.cu, iq_kernels.cu | ~25 |
| __vsubss4(a, b) | Manual byte-wise subtraction | 
ative_mmoq.cu | 2 |
| __fmaf_rn(a, b, c) | maf(a, b, c) from <cmath> | 
ative_bf16.cu, 
ative_flash_attn.cu, 
ative_gdn_preprocess.cu, 
ative_gr_postops.cu, 
ative_ple_postops.cu | ~10 |

#### __dp4a → __builtin_amdgcn_sdot4

Confirmed by llama_findings_RDNA.md (llama.cpp uses it at common.cuh:708):
- __dp4a(a, b, c) = c + Σ(int8(a[i]) * int8(b[i])) for 4 bytes — CUDA sm_61+
- __builtin_amdgcn_sdot4 → V_DOT4_I32_I8 ISA on RDNA2 (gfx1031) and RDNA4 (gfx1201)
- Also from md_math_functions.h:45: md_mixed_dot(char4 a, char4 b, int c, bool saturate) wraps __ockl_sdot4

Replacement macro:
`cpp
#ifdef __HIP__
__device__ __forceinline__ int dp4a_compat(int a, int b, int c) {
    return __builtin_amdgcn_sdot4((const char4&)a, (const char4&)b, c);
}
#else
__device__ __forceinline__ int dp4a_compat(int a, int b, int c) {
    return __dp4a(a, b, c);
}
#endif
`
Or per-file, use hipify-clang then search/replace __dp4a → __builtin_amdgcn_sdot4((const char4&), (const char4&), ).

#### __vsubss4 → manual replacement

CUDA: __vsubss4(a, b) — 4× signed byte subtraction, returns int (a and b treated as 4 signed bytes each)
HIP replacement:
`cpp
__device__ __forceinline__ int vsubss4_compat(int a, int b) {
    int result;
    int8_t* rp = (int8_t*)&result;
    const int8_t* ap = (const int8_t*)&a;
    const int8_t* bp = (const int8_t*)&b;
    rp[0] = ap[0] - bp[0];
    rp[1] = ap[1] - bp[1];
    rp[2] = ap[2] - bp[2];
    rp[3] = ap[3] - bp[3];
    return result;
}
`

#### __fmaf_rn → maf

CUDA: __fmaf_rn(a, b, c) — fused multiply-add, round-to-nearest-even (IEEE 754 default)
HIP: maf(a, b, c) from <cmath> — same semantics, same hardware instruction (_fma_f32 on AMD)
Replace #include to ensure <cmath> is available.

## 4. Kernel File-by-File Conversion Matrix

| Kernel File | Shuffles | dp4a/vsubss4 | fmaf/fmul/fadd | byte_perm | half22float2 | Complexity |
|------------|----------|-------------|----------------|-----------|--------------|------------|
| elementwise.cu | __shfl_down_sync | — | __fmul_rn, __fadd_rn | (via int_as_float) | — | adapt |
| f16_gemv.cu | __shfl_down_sync | — | — | — | — | direct |
| cvec.cu | __shfl_xor_sync | — | — | — | — | direct |
| dequant_bf16.cu | — | — | — | — | — | direct |
| dequant_s2.cu | — | — | — | — | — | direct |
| used_gdn.cu | __shfl_xor_sync | — | — | — | — | direct |
| used_gr.cu | — | — | — | — | — | direct |
| gdn.cu | __shfl_down_sync | — | — | — | — | direct |
| gr.cu | — | — | — | — | — | direct |
| iq_kernels.cu | — | __dp4a | — | __byte_perm | — | **rethink** |
| kv_q4.cu | — | — | — | — | — | direct |
| kv_q8.cu | — | — | — | — | — | direct |
| kv_stream.cu | — | — | — | — | — | direct |
| 
ative_bf16.cu | — | — | __fmaf_rn | — | — | adapt |
| 
ative_flash_attn.cu | __shfl_xor_sync | — | __fmaf_rn, __fmul_rn | — | — | direct+adapt |
| 
ative_gdn.cu | __shfl_down_sync | — | — | — | — | direct |
| 
ative_gdn_preprocess.cu | __shfl_xor_sync | — | __fmaf_rn, __fmul_rn, __fadd_rn | — | — | direct+adapt |
| 
ative_gr_norm.cu | __shfl_xor_sync | — | — | — | — | direct |
| 
ative_gr_postops.cu | — | — | __fmaf_rn, __fmul_rn, __fadd_rn | — | — | adapt |
| 
ative_mmoq.cu | __shfl_xor_sync | __dp4a, __vsubss4 | — | __byte_perm | __half22float2, __low2float | **rethink** |
| 
ative_moe.cu | — | — | — | — | — | direct |
| 
ative_ple_postops.cu | __shfl_xor_sync | — | __fmaf_rn, __fmul_rn, __fadd_rn | — | — | direct+adapt |
| 
ative_qsa.cu | — | — | — | — | — | direct |
| 
ative_qsa_indexer.cu | — | — | — | — | — | direct |
| 
ative_qsa_score.cu | — | — | — | — | — | direct |
| 
ative_rope.cu | — | — | — | — | — | direct |
| 
ative_router.cu | — | — | — | — | — | direct |
| ple.cu | — | — | — | — | — | direct |
| qsa.cu | — | — | — | — | — | direct |
| qsa_decode_attn.cu | __shfl_xor_sync | — | — | — | __low2float | direct |
| qsa_select.cu | — | — | — | — | — | direct |
| quantize_act.cu | — | — | — | — | — | direct |
| ope.cu | — | — | — | — | — | direct |
| s2_gemv.cu | — | — | — | — | — | direct |
| s2_gemv_fast.cu | __shfl_down_sync | — | — | __byte_perm | __low2float | direct |
| s2_gemv_q8.cu | — | — | — | — | — | direct |
| s2_gemv_quads.cu | — | — | — | — | __low2float | direct |
| s_gemv.cu | — | — | — | — | __low2float | direct |
| sampler.cu | — | — | — | — | — | direct |
| shared_expert.cu | — | — | — | — | — | direct |
| erify_kernels.cu | — | — | — | — | — | direct |
| outer_top10.cu | — | — | — | — | — | direct |
| s2_expert_grouped.cu | — | — | — | — | — | direct |
| 
ative_qsa.cu | — | — | — | — | — | direct |
| 
ative_rope.cu | — | — | — | — | — | direct |

36 CUDA kernel files in src/kernels/cuda/. 28 are DIRECT, 6 need ADAPT (fmul_rn/fadd_rn/fmaf_rn), 2 need RETHINK (native_mmoq.cu with __dp4a/__vsubss4, iq_kernels.cu with __dp4a).

## 5. Build System Changes (CMakeLists.txt)

### Replace CUDA with HIP
`cmake
# OLD (CUDA):
option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES 120 ...)  # sm_120 enforcement

# NEW (HIP):
option(STRATA_ENABLE_HIP "Build the HIP targets" OFF)
if(STRATA_ENABLE_HIP)
  # Use hipcc compiler via CMake HIP language support
  set(CMAKE_HIP_COMPILER "{ROCM_INSTALL_DIR}/bin/hipcc")
  set(CMAKE_HIP_ARCHITECTURES gfx1031 gfx1201)  # RDNA2 + RDNA4
  # or set to: set(CMAKE_HIP_ARCHITECTURES gfx1000) for native
endif()
`

### Replace CUDA::cublas → rocm-rocmblas
`cmake
# OLD (CUDA):
target_link_libraries(strata_prefill PUBLIC strata_engine CUDA::cublas CUDA::cudart)

# NEW (HIP):
# Option A: Use find_package for rocBLAS
find_package(rocBLAS REQUIRED)
target_link_libraries(strata_prefill PUBLIC strata_engine roc::rocblas )

# Option B: Direct library paths (simpler for Windows)
target_link_libraries(strata_prefill PUBLIC strata_engine
    /lib/rocblas.lib
    /lib/amdhip64.lib  # HIP runtime
)
`

### Replace CUDA::cudart → hip
`cmake
# OLD:
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings CUDA::cudart)

# NEW:
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings hip::hip)
# or: /lib/amdhip64.lib
`

### Replace CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
`cmake
# OLD:
target_include_directories(strata-concurrent PRIVATE )

# NEW:
target_include_directories(strata-concurrent PRIVATE /include/hip)
`

## 6. Conversion Workflow

### Step 1: Run hipify-clang on all CUDA files
`ash
# Use the hipify-clang from either runtime:
G:\ROCM10RT-gfx1201\bin\hipify-clang.exe \
  --cuda-include-dirs=G:\ROCM10RT-gfx1201\include \
  --hip-include-dirs=G:\ROCM10RT-gfx1201\include\hip \
  --print stats \
  -inplace \
  src/kernels/cuda/*.cu \
  src/core/*.cu src/core/*.cpp \
  src/prefill/*.cu src/prefill/*.cpp \
  src/platform/*.cu src/platform/*.cpp \
  src/spec/*.cpp
`
Or use --cuda-to-hip to output to separate files for review:
`ash
G:\ROCM10RT-gfx1201\bin\hipify-clang.exe \
  --cuda-to-hip \
  -o output_dir \
  src/kernels/cuda/native_flash_attn.cu
`

### Step 2: Fix the 3 manual replacements
- **__dp4a**: In 
ative_mmoq.cu and iq_kernels.cu, replace __dp4a(a,b,c) → __builtin_amdgcn_sdot4((const char4&)a, (const char4&)b, c)
- **__vsubss4**: In 
ative_mmoq.cu, replace __vsubss4(a,b) → manual byte-wise subtraction
- **__fmaf_rn**: In 5 files, replace __fmaf_rn(a,b,c) → maf(a,b,c) (ensure #include <cmath>)

### Step 3: Fix cublas → rocblas in gemm.cu
- Replace cublasCreate → ocblas_create_handle
- Replace cublasGemmEx → ocblas_gemm_ex (enum changes: CUBLAS_OP_T→ROCBLAS_OP_T, etc.)
- Replace cublasSetStream → ocblas_set_stream
- Replace cublasDestroy → ocblas_destroy_handle
- Remove cublasSetMathMode and cublasSetWorkspace (not needed in rocBLAS)

### Step 4: Fix arch detection in CMakeLists.txt
Replace sm_120 / CMAKE_CUDA_ARCHITECTURES-based checks with hipDeviceProp_t.gcnArchName[256]:
`cpp
// In device.cpp, replace CUDA arch detection:
// OLD: prop.major, prop.minor (CUDA compute capability)
// NEW: prop.gcnArchName (AMD GCN arch string, e.g. "gfx1031", "gfx1201")
`

### Step 5: Fix header includes
- #include <cuda_runtime.h> → #include <hip/hip_runtime.h>
- #include <cuda_fp16.h> → #include <hip/hip_fp16.h>
- #include <cuda_bf16.h> → #include <hip/hip_bf16.h>
- #include <cublas_v2.h> → #include <hipblas/hipblas.h> or <rocblas/rocblas.h>

### Step 6: Build and test
`ash
# gfx1201 build first (full runtime):
mkdir build && cd build
cmake -DSTRATA_ENABLE_HIP=ON ^
  -DROCM_INSTALL_DIR=G:\ROCM10RT-gfx1201 ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 ^
  ..
cmake --build . --target strata

# gfx1031 build (now that import libs are copied):
cmake -DSTRATA_ENABLE_HIP=ON ^
  -DROCM_INSTALL_DIR=G:\ROCM10RT-gfx1031 ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1031 ^
  ..
cmake --build . --target strata
`

## 7. HipKittens Reference Patterns

HipKittens (C:\Users\rr\OneDrive\Desktop\AMDS\HipKittens) is the official HIP port of ThunderKittens
(834 files, CDNA-only). While CDNA-specific, it provides correct HIP patterns:

- **Build** (kernels/common.mk): hipcc with --offload-arch=gfxXYZ, C++20, GPU_TARGET env var
- **Types** (include/cdna3/common/base_types.cuh): using bf16 = __hip_bfloat16, using half = __half
- **Kernel launch**: hipFuncSetAttribute + <<<>>> (same as CUDA)
- **Shared memory**: extern __shared__ + hipMalloc (same as CUDA)
- **Sync**: __syncthreads() (same), sm volatile("s_waitcnt vmcnt(0)") (AMD-specific)
- **MMA**: mma_ABt<float, mfma>(...) — AMD Matrix Core instructions
- **Wavefront**: WARP_THREADS = 64 (matches RDNA2/RDNA4)
- **Shuffle**: kittens::warp_shuffle_xor(val, laneMask) → uses __shfl_xor_sync
- **Error handling**: hipGetLastError() + hipGetErrorString() (same as CUDA)

Key difference: HipKittens uses CDNA-specific intrinsics (__builtin_amdgcn_s_waitcnt, __builtin_amdgcn_mbcnt,
__builtin_amdgcn_s_barrier) that are NOT available on RDNA2/RDNA4. Strata can reuse the HIP API patterns
(hipMalloc, hipStreamCreate, etc.) but must NOT use CDNA-only intrinsics.

## 8. Shuffle Intrinsic Correctness (B3)

__shfl_down_sync / __shfl_xor_sync with width=32 on AMD wavefronts (64 threads):
- Strata uses shuffles for warp-level reductions with offsets 1, 2, 4, 8, 16 — all < 32
- With width=32 explicitly set, only the first 32 threads participate
- The __shfl_xor_sync(0xffffffffu, v, offset, 32) in native_flash_attn.cu uses XOR mask, same safety
- This is safe on AMD wavefronts — the width parameter restricts participation to 32 threads
- HipKittens uses width=64 (__shfl_down_sync(0xFFFFFFFFu, v, o, 64)) but Strata's 32-width is correct
  for its existing CUDA semantics
- **Verified: no correctness issue** (offsets 1,2,4,8,16 never cross the 32-thread boundary)

## 9. AMD Intrinsic Coverage (from llama_findings_RDNA.md)

| Intrinsic | ISA (RDNA2) | ISA (RDNA4) | Available |
|-----------|-------------|-------------|-----------|
| __builtin_amdgcn_sdot4 | V_DOT4_I32_I8 | V_DOT4_I32_I8 | Yes (dp4a equiv) |
| __builtin_amdgcn_sdot8 | V_DOT8_I32_I4 | V_DOT8_I32_I4 | Yes (8-nibble) |
| __builtin_amdgcn_udot4 | V_DOT4_U32_U8 | V_DOT4_U32_U8 | Yes |
| __builtin_amdgcn_udot8 | V_DOT8_U32_U4 | V_DOT8_U32_U4 | Yes |
| __builtin_amdgcn_sdot2 | V_DOT2_I32_I16 | — | RDNA3+ only |
| __builtin_amdgcn_udot2 | V_DOT2_U32_U16 | — | RDNA3+ only |

**For Strata's __dp4a**: __builtin_amdgcn_sdot4 is the exact match — both compute c + Σ(int8 × int8) for 4 bytes.
The only difference is argument types: __dp4a(int, int, int) vs __builtin_amdgcn_sdot4(char4, char4, int).
Use type punning: (const char4&)a casts the int to char4 without changing the bit pattern.

## 10. Decision Summary

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Use rocBLAS (ocblas_gemm_ex) for GEMM | Matches cublasGemmEx params exactly (verified at rocblas-functions.h:21904) |
| D2 | Use hipify-clang then manual fixups | 28/36 files are direct; only 3 intrinsics need manual replacement |
| D3 | __dp4a → __builtin_amdgcn_sdot4 | Confirmed by llama_findings_RDNA.md (V_DOT4_I32_I8 on RDNA2) |
| D4 | __fmaf_rn → maf() | IEEE 754 round-to-nearest is default; same hardware instruction |
| D5 | __vsubss4 → manual byte subtraction | No AMD intrinsic equivalent |
| D6 | __fmul_rn — no change needed | USER confirmed available as compiler builtin on RDNA2/RDNA4 |
| D7 | __shfl_*_sync — no change needed | Direct in HIP 7.16, same CUDA signature |
| D8 | __half22float2/__low2float/__high2float — no change | Direct in HIP 7.16 (amd_hip_fp16.h:486-488) |
| D9 | __byte_perm — no change | Direct in HIP 7.16 (amd_device_functions.h:257) |
| D10 | Build on gfx1201 first (full runtime) | gfx1031 now has import libs copied from gfx1201 |
| D11 | Strata does NOT need MIOpen | Strata uses rocBLAS for GEMM only, no cuDNN/hipDNN dependency |
| D12 | Use --use_fast_math flag | Matches existing Strata comment // Compile with --use_fast_math |
| D13 | Use __builtin_amdgcn_sdot4 not md_mixed_dot | More direct, matches llama.cpp's approach |
| D14 | Arch detection via gcnArchName[256] | Replaces cudaDeviceProp.major/minor CUDA capability check |
| D15 | Copy import libs from gfx1201 to gfx1031 | Import libs are arch-agnostic symbol stubs referencing DLL exports |
@@
# Blockers

| ID | Description | Resolution |
|----|-------------|------------|
| B1 | gfx1031 missing rocBLAS/hipBLAS/MIOpen import libs | **FIXED**: Copied from gfx1201 (import libs are arch-agnostic) |
| B2 | __builtin_amdgcn_sdot4 compilation on gfx1031 | Confirmed: V_DOT4_I32_I8 ISA is RDNA2 native (llama_findings_RDNA.md) |
| B3 | __shfl_*_sync width=32 on 64-thread wavefronts | Verified: offsets 1,2,4,8,16 stay within 32-thread partition, safe |
| B4 | hipify-clang validation | **N/A**: the RDNA4 kernels are hand-written HIP (the `kernels/cuda` tree), not bulk-converted, so there is no hipify pass to validate |
| B5 | Remaining Strata file reads | **DONE**: `iq_kernels.cu`, `moe_mmq.cu`, the parity tests and the engine (`generate.cpp`/`session.cpp`/`verify.cpp`) are ported and build for gfx1201 and gfx1031 |

## 11. Port Work Completed (runtime-verified on gfx1201 / RX 9070 XT)

### 11.1 Model: Qwen3.8-Flash-Next reap-288
- ggml format family is `qwen4exp`; **expert count is runtime, not hardcoded 512**.
- Per-expert blob: `gate/up = Q4_K (type 12)`, `down = Q5_0 (type 6)`.
  - IMPORTANT: the down projection is Q5_0, NOT Q8_0. `iq_row_bytes(6, n)` returns `(n/32)*22` (block_q5_0 = d + qh[4] + qs[16] = 22 B). Verified in `struct NativeExpertLayout` and at runtime (`native_expert_grouped` now accepts gu=12 / down=6).
  - The earlier Q8_0 (type 8) dot is still wired in (Fmt<8>/native_down_kernel<8>) for artifacts that use it.
- PLE (the token-embedding/rope auxiliary table) is Q5_0; supported in the ngram reader alongside IQ4_NL.
- Head: native Q5_K (`--native-head-gguf`, 521,472,000 bytes).
- 48 layers, hidden 2560, attn 24 heads, 2 kv, n_ff 1536/expert, top-10, 288 experts/layer.

### 11.2 GPU grouped-expert kernels (`src/kernels/cuda/iq_kernels.cu`)
- `native_expert_grouped` dispatches on `Fmt<TY>` (`row_dot<TY>`) for the gate/up (`native_gu_kernel<TY>`) and down (`native_down_kernel<TY>`).
- Added: `Fmt<6>` (qk=32, ipb=1, step=1), `Fmt<8>`, `Fmt<12>` (existed); `native_gu_kernel<12>`, `native_down_kernel<8>`, `native_down_kernel<6>` switch cases.
- `vec_dot_q5_0_q8_1` and `vec_dot_q8_0_q8_1` added. They follow the file's **scalar** dp4a-free inner-loop idiom used by `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1`, not the dp4a `char4` packing — which avoids the misaligned-int-alias bugs the 3-agent debate converged onto. q5_0 5th-bit extraction: qh bits 0..15 -> values 0..15; bits 16..31 -> values 16..31 (`(qh >> (j+16)) & 1`); -16 correction via `block_q8_1.ds.y = d8 * sum(q8)`.
- Coverage: n_ff=1536, qk=32 → nb = 48, total calls = 48*1 = 48, elements = 48*32 = 1536 (exact).

### 11.3 Correctness fixes landed
- `fused_gr_read_multi` `hipErrorInvalidValue`: `kFusedGrMaxT=8` asked for 80 KB LDS, but RDNA4 opt-in LDS ceiling is 64 KB, so the `hipFuncSetAttribute` silently failed and the 60 KB launch hit the default limit. Fixed by clamping `hipFuncAttributeMaxDynamicSharedMemorySize` to `hipDeviceAttributeMaxSharedMemoryPerBlock` and guarding the launch size.
- `layer.cpp` router: the 512-only `native_router_top10` was hard-erroring on 288 experts. Fixed to fall back to the generic `router_top10` when `g.n_expert != 512 || k != 10` (same pattern as `prefill.cpp` and `mtp.cpp`).
- `native_expert_layout` / `iq_row_bytes` now accept type 6 and type 8.
- `verify.cpp` **arena-ordering fix (this was the speculation defeat)**: `hipMemset(arena_, 0, count.used)` ran on the legacy default stream, which is NOT ordered against the verifier's `hipStreamNonBlocking` streams; on ROCm 10.1 it executed *after the first verify window*, wiping that window's residual (`R_`) and the GDN/QSA snapshots (`qkv_L_`, `h_L_`, `z_`, `gate_L_`, `beta_L_`) that the first `commit` graph then reads, so the first committed recurrent state was built from zeros. Every later window inherited the corrupted state. The clear now runs on `cs_` with an explicit `hipStreamSynchronize`; the same ordering was applied to `d_parts` and the penalty history `d_hist` in `generate.cpp`.
- `iq_kernels.cu` `vec_dot_q8_0_q8_1`: the activation was indexed by both `kbx` and `iqs`, double-counting the offset and reading past the row on the last block. The weight block now advances by `kbx`, the activation by `iqs`.
- `ggml_cuda_host.cu` **MMQ arch id (this is why any i-quant model failed to prefill)**: the host glue set `device.cc = 100*major + 10*minor` (the CUDA formula) instead of ggml's AMD id `GGML_CUDA_CC_OFFSET_AMD + 0x1201`. With no AMD offset, `GGML_CUDA_CC_IS_AMD`/`GGML_CUDA_CC_IS_RDNA4` are false for a Radeon, so `ggml_cuda_mmq_get_config` returned a *CUDA* config on the host while the device code was compiled as RDNA4 (`-DRDNA4`), and every MMQ launch died as `mul_mat_q has no device code compatible with HIP arch` (`NO_DEVICE_CODE`). The glue now parses `prop.gcnArchName` exactly as `ggml_cuda_parse_id` does. reap-288 hid this because its canonical Q2_0 pack prefill goes through the port's own `gather_strata_q2`/Q2_0 path, never the failing `IQ2_S` instance. **Verified with Swift IQ2_XS** (same architecture, 512 experts): prefill 19.9 tok/s, decode 15.0 tok/s, spec 2 and spec 4 give the identical stream, self-oracle 8/8.
- `strata_mmq` coverage widened from the i-quants + `Q2_0` to also `Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S` (the RDNA4 MMQ config already carried them; only the instance list, `mmq::supported` and the dispatch switch needed them), so GGUF-native packs whose experts are K-quants can prefill too.
- `iq_kernels.cu` **q8_1 bias term**: `quantize_q8_1_kernel` stored `block_q8_1.s` as the sum of the raw inputs; ggml defines it as `d * sum(quantized q)`. The Q4_0/Q5_0/K-quant dots subtract a closed-form bias times this term, so the raw-input sum left the bias partly uncancelled. The Q5_0 expert-down projection (reap-288's `down = Q5_0`) drifted ~3.4e-2 from the float reference and failed `native_expert_parity` on the layers whose down projection is Q5_0 (6, 7, 12, 21, 24, 28, 30). The kernel now reduces the quantized `q` and stores `d * sum`; **48/48 layers pass** (gpu rel ~3.4e-2 -> ~1.16e-2).

### 11.4 Measured end-to-end (gfx1201, spec 4, 280-slot VRAM cache, bootstrap profile)
- decode 8.20 tok/s (16 tokens / 1950 ms)
- prefill 4.22 tok/s (6 tokens / 1420 ms; TTFT 1614 ms)
- CPU expert pool dominates: ~48.8 ms/round
- expert arena is currently hiphostregister-pinned fully (42.19 GiB, 0.74 GiB/s at startup)
- speculation: 0 / 45 drafts accepted at this snapshot (engine 0.1.15). The cause was the verify-window arena wipe above; post-fix numbers are in 11.5.
- output decodes to chat structure (`<|im_start|>assistant ...`), but prompt must be tokenized from `tokenizer.json` to judge coherence (see 11.7).

### 11.5 Open work / hot items (TODO)
- B1 FIXED, B2 FIXED, B3 FIXED (shuffle width=32 safe).
- **RAM/VR low**: replace the full 42 GiB arena with file-backed mmap + a bounded pinned staging ring (demand-load only hot experts).
- **GPU utilization**: the miss path also runs on the GPU. `expert_pool_dispatch_multi` builds a `GpuPlanSink` for the resident experts plus a `pcie_num/256` share of the missed ones (VRAM groups + PCIe groups), and `native_expert_grouped` computes them; the missed blobs are read over REBAR with no staging copy (`--pcie-mode direct`, `device_alias`) or staged (`dma`/`kernel`). `--pcie-frac` selects the share (auto: 0.2 direct for the Q2_0 pack, 0.55 DMA for native packs). Remaining: tune the fraction against the real profile (the 48.8 ms/round CPU figure predates this path).
- **Real hot-expert profile**: build from `route_reap288.bin` (768 records), not the round-robin bootstrap profile (hit rate ~13/1579).
- **Speculation**: FIXED. The verify-window arena wipe in 11.3 caused the 0/N acceptance. Post-fix validation: `--spec 2`, `--spec 4` and `--spec 6` produce the *identical* greedy stream (they diverged before); feeding the engine's own continuation back as `--spec-oracle` accepts 12/12 at `--spec 4` and 8/8 at `--spec 2`; repeated runs are bit-identical. Decode 23.5 tok/s at `--spec 4` with accepted drafts (14.2 tok/s at `--spec 2` with all drafts rejected), measured on a 64-token run with `--expert-cache 2600`.
- **A second model (Swift) runs**: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (512 experts, same 48L/2560 geometry, `IQ2_S` gate/up + `Q2_0` down) packs with `tools/iq_pack.py` and runs end-to-end after the MMQ arch-id fix (11.3): prefill 19.9 tok/s, decode 15.0 tok/s, spec-2/4 identical, self-oracle 8/8. A self-contained pack (with `--experts-bin`, 33.02 GiB of expert blobs) is built at `H:\OLLAMA-Models\strata-pack-swift`. Still unsupported: `Whittle-Qwen-3.8-35B-A3B` (40 layers / 2048 embd / 180 experts / `expert_used_count=8`) - `Verifier::init` requires `ss.k == 10` and `prefill.cpp` hard-codes the reap-288 geometry (`N=2560, K=10`), so a different geometry needs both relaxed first.
- **`hipModuleUnload ... while stream is capturing`**: NOT called by Strata; comes from the ROCm runtime tearing down a module (likely the lazily-loaded Q5_K head module / graph-captured module) mid-capture. Fix: defer that module's teardown past `hipStreamEndCapture`, or load the head module eagerly before capture begins. TODO in `src/core/native_head.cpp`.
- **`iq_parity` fixture**: `src/kernels/iq_parity.cpp` reads `<TYPE>.bin`/`.f32` fixtures that were never committed, and its usage line names a `tools/iq_fixture.py` that never existed. gguf-py cannot replace it: it ships the i-quant *de*quantizers only and raises `NotImplementedError` from `quantize_blocks`, so there is no Python quantizer to build the blocks. Closing this needs a generator linked against ggml's C quantizers (`ggml_quantize_chunk`); it is the one parity test that cannot run. The others (`native_expert_parity`, `s_gemv*`, `quantize_act`, `shared_expert`, `s2_gemv*`) build and pass.

### 11.6 Verification commands
- gfx1201 build: `cmake -DSTRATA_ENABLE_HIP=ON -DROCM_INSTALL_DIR=G:\ROCM10RT-gfx1201 -DCMAKE_HIP_ARCHITECTURES=gfx1201 .. && cmake --build . --target strata`
- run (from build_gfx1201): `strata.exe --pack ... --native <GGUF> --ple-gguf <GGUF> --spec 4 --prefill 512 --expert-cache 280 --expert-profile data/reap288-profile.bin --tokens "..." --max-new 16 --max-context 4096`
- compile flags applied to the 11 rn-intrinsic files: `-ffp-contract=off` (prevents contraction of `__builtin_amdgcn_sdot4`/fmul patterns); `--use_fast_math` per D12.

### 11.7 Coherence test recipe (engine takes `--tokens` IDs only)
- Tokenize a real chat prompt with the pack's `tokenizer/tokenizer.json` via HF `tokenizers`:
  `<|im_start|>user\nQwen, write a Python function that adds two numbers.<|im_end|>\n<|im_start|>assistant\n`
  (`248045` = `<|im_start|>`, `248046` = `<|im_end|>` on this vocab).
- Run, decode the generated IDs via the inverted vocab.
- Compare per-token top-1 logits against a llama.cpp/gguf run on the SAME GGUF (`llama-cli --model <GGUF> --temp 0`). The single decisive metric separating "coherent but slow" from "broken logits": **top-1 token agreement** — the target engine's greedy next token must equal llama.cpp's at each step. Mismatches = a correctness bug; all-pass = coherent.

### 11.8 Prompt-path perf: rocBLAS warm-up + cache-preferring reads (commit a6949a0)

`STRATA_PREFILL_TIMING=1` now reports, per phase, GPU-elapsed **and host-elapsed** (`gpu/host ms`),
per-call host stopwatches (`need`, `proj.bf16`, `proj.native`, `grp.*`) and the expert PCIe volume.
That instrumentation pinned the following on the 39-token Swift prompt (`--prefill 128`,
`--expert-cache 2600 --expert-profile data/expert-profile.bin`):

- **A one-time rocBLAS BF16/FP16 init was landing on the first token.** Of 432 `proj.bf16` calls exactly
  one exceeded 5 ms and it was **346 ms** (likewise one 281 ms `proj.native`); every other call was
  ~0.04 ms. `Gemm::warmup` now runs a throwaway BF16 and FP16 GEMM in `Prefill::init`, so the init is
  paid at load: **prefill 1153 -> 566 ms**.
- **Resident experts were read over PCIe even when already in the VRAM cache.** The direct path checked
  the arena's REBAR `device_alias` first; it now prefers the cache slot when `host_res >= 0`:
  **566 -> 485 ms**.
- Net: **1153 -> 485 ms, 33 -> 80 tok/s (~2.4x)** on the short prompt.

**The bottleneck is the PCIe link, and it is saturated.** The profile shows the expert gather (`dequant`
phase) at ~280 ms carrying 8.0 GiB — i.e. **30.4 GB/s**. This box is **X570 + Ryzen 9 5900XT = PCIe 4.0
x16** (~32 GB/s theoretical, ~28-30 usable), and ReBAR is active (`local heap 15.922 GB, invisible heap
0 B`). A micro confirmed a single in-kernel BAR read runs at 26 GB/s, and that `hipMemcpy` from the
mapped arena is **elided (no-op)** — there is no copy-engine path; the kernel BAR read is the only
mechanism. So:

- **Concurrent/parallel reads cannot help** — the link is the limit.
- **Overlapping the gather with the MMQ does not help either**: a double-buffered version (gathers on
  the copy stream, `gather_ev`/`mmq_ev` handoff, 2x group slots) measured **524 ms vs 485 ms — a
  regression** (the two streams contend for the same saturated link and SMs) and was reverted.
- Only **reading fewer bytes** helps — the tiered cache (`--expert-cache`/`host_res` in VRAM, the pinned
  arena in RAM, the pack on disk) — and only on **reuse** (multi-chunk / multi-turn / decode). A cold
  single chunk genuinely needs its ~8 GiB, so ~458 ms is near its floor.

**External GPU profilers are unavailable on this Windows + RDNA4 stack** (recorded so it is not
re-litigated):

- **AMDuProf GPU profiling is Windows-unsupported** — AMD's own feature matrix lists GPU
  Profiling/Tracing as Linux-only; `collect --config gpu_sol` errors *"GPU Profiling is not supported on
  this System"*. (µProf CPU profiling does work; its help ships as text files under
  `AMDuProf\bin\Help\text\`, so never call `-h`, which pages the console.)
- `rocprofiler-sdk` / `rocprofv3` and `rocgdb` are Linux-only; TheRock (`G:\rocm-10\CMakeLists.txt:215`)
  forces `THEROCK_FLAG_INCLUDE_PROFILER` **OFF on WIN32**; AMD's component table lists the Windows
  profiler as **RGP**. `rocgdb` (`C:\ROCm72\bin\rocgdb.exe`) is `GNU gdb (ROCm)` — a debugger, no
  profile commands.
- **RGP** does support HIP on RX 9000 / Windows 11, but its Radeon Developer Panel would not connect to
  this ROCm-HIP binary (tried hosted and elevated, and a trivial HIP micro; the RDP service/router comes
  up but no app connects). Live per-kernel GPU timing therefore comes from the engine's own
  `STRATA_PREFILL_TIMING`.

**Known issue**: `--expert-cache-per-layer` aborts with
`ExpertCache::verify_slot: slot 0 differs from the arena at byte 0` (the profile has 24576 ranked pairs
vs 5243 per-layer slots), so per-layer admission is currently unusable.

### 11.9 Toward the RDNA4 target: prefill 800 tok/s, TG 40+ (converged plan)

Grounded in the §11.8 measurements: PCIe 4.0 x16 (~30 GB/s) is the hard link, 16 GB VRAM, ~33 GiB of
experts, ~1.5 GiB dense, 2-4 GB KV. This section is the converged result of a delegated 3-agent debate
(Poolside), corrected against the measured facts.

**Prefill 800 tok/s is a LARGE-CHUNK target and is already met.** The per-chunk expert read is
paid once per layer and then reused across the chunk, so throughput scales with chunk length:
a 39-token chunk reads ~8 GiB for 39 tokens (PCIe-bound, 80 tok/s), while a 2047-token chunk amortizes
those same classes of reads over ~52x the tokens — **measured 943.4 tok/s generate (871 tok/s
end-to-end) for 2047 tokens after the §11.8 warm-up**. The 80 tok/s figure is a short-chunk artifact,
not the model's ceiling. Amortization: unique-experts/token falls from ~5.4 (39 tok) to ~0.14 (2048 tok,
all 288 experts/layer). The open target is now **decode, measured 12.6 tok/s** (see action 4).

**Ranked actions (highest ceiling first):**

1. **(DONE, §11.8)** rocBLAS warm-up at load — removes the ~630 ms one-time init from the first token.
2. **Prefill with large chunks** (>=1024, ideally 2048). The dominant control; most of the 80 -> 800.
3. **Tiered expert residency** — the only lever that beats the link (a hit reads 0 B over PCIe):
   VRAM hot set from the routing-frequency profile (`--expert-profile`), per-layer quota sized to ~6-8 GB
   (16 - 1.5 dense - 2-4 KV - scratch); pinned arena = RAM warm tier; pack = cold. **Fix the
   `--expert-cache-per-layer` `verify_slot` abort first** — per-layer admission is currently unusable.
4. **Decode/TG 40+.** Speculative decoding is already wired (§11.5: 23.5 tok/s at `--spec 4`, identical
   greedy stream). Push it with warm experts for the verify pass, a larger draft depth, and a tuned
   CPU-pool-vs-GPU split. Target 40+.
5. **KV cache in VRAM** (block layout) so the 2-4 GB KV does not displace the hot expert set.
6. **Later micro-opts (low ceiling):** resident-set quantization; expert weight layout for rocBLAS.

**Physically impossible on this link (do not chase):** zero PCIe transfers (cold experts must cross it);
>30 GB/s (saturated); a fast 39-token chunk (it needs a fixed ~8 GiB); any all-experts-resident scheme
(33 GiB >> 16 GB VRAM).

**Corrections to the delegated plans.** Both agents proposed "compute/transfer overlap" as a win, but
§11.8 measured a naive per-group overlap as a **regression** (524 vs 485 ms) — the read and the compute
contend for the same saturated link and SMs; overlap only helps if the overlapped work is *independent*
of the read (e.g. the next layer's dense/attention), which is unproven here. Their VRAM budgets
(>12 GB for experts) are also infeasible on 16 GB.

### 11.10 Literature scan: borrowable techniques

Evaluated: ISTA-DASLab GSQ-RCO model card (the Swift GGUF family), the Qwen3.8-Flash-Next architecture
deep-dive, and arXiv 2609.19969 (DeepSeek-V4.1-Flash KV compression), 2609.26368 (HySparse2),
2607.20981 (edge MoE efficiency survey), 2605.23893 (Complete-muE), 2604.07035 (dense/MoE
accuracy-efficiency benchmark), 2505.04081 (QStore).

**Directly applicable, ranked:**

1. **Quant format is a first-order prefill lever — `Q2_0` over `IQ2_XS`.** The GSQ-RCO card measures, on
   the same 512-expert / 48-layer model, **367.49 vs 108.19 prompt tok/s** and 93.79 vs 70.30 decode tok/s
   (`Q2_0` vs `IQ2_XS`): "decoding them costs real time, and on this model that cost dominates
   inference." `Q2_0` is block-64 with no per-format lookup tables. Strata's Swift run uses `IQ2_XS`
   while the reap-288 pack uses `Q2_0`, so a `Q2_0` Swift build is the cheapest big prefill win.
2. **Speculative decoding for TG 40+.** Qwen ships a 4B MTP head (1 layer + LM head, QSA attention) for
   spec decode; DSpark (2609.19969) adds confidence-scheduled verification length. Strata already has
   `--spec`; for a MoE, verifying k tokens per expert-weight fetch also amortizes the PCIe transfer.
3. **Co-activation-ordered expert layout** (ZipMoE, via 2607.20981): pack frequently co-routed top-10
   experts contiguously so PCIe reads are clean bursts (the survey cites up to 72.77% latency reduction).
4. **Routing-predicted prefetch + hot-cold hierarchy** (2607.20981; DeepSeek-V4.1 Engram, 2609.19969):
   prefetch the next layer's predicted experts while computing; pin non-expert weights + the hottest
   experts in VRAM. `--expert-profile` already predicts; it needs to drive a prefetch, not just admission.
5. **Per-expert mixed precision + routing-preserving quantization** (2607.20981): more bits on
   rare-critical experts, fewer on routine ones; keep the router logits precise so the Top-10 *set* is
   stable when experts are re-quantized.
6. **Fuse gate/up/down into one expert kernel** (Mega-MoE, 2609.19969) — Strata runs MMQ gu, swiglu and
   MMQ down as separate launches today.
7. **FP4/FP8 expert weights** (2609.19969) cut PCIe bytes 2-4x *only above ~4 bpw*; at Strata's 2.5 bpw
   the expert bytes are already near-minimal, so this matters only through item 1 (a cheaper-to-decode
   format may be slightly larger yet faster).

**Not applicable** (recorded so it is not re-checked): HySparse2 (2609.26368) — attention/KV architecture,
needs retraining, nothing on expert offload; Complete-muE (2605.23893) — training hyperparameters;
2604.07035 — an accuracy-efficiency benchmark with no systems technique; QStore (2505.04081) — lossless
joint high/low-precision storage, not a speed lever.

## 12. Optimization backlog (TODO)

Measured baseline on the 39-token and large-chunk Swift prompts (gfx1201, RDNA4): short-chunk prefill
80 tok/s, **large-chunk prefill 944 tok/s (2047 tokens, generate) / 871 e2e**, decode **12.6 tok/s**.
Targets: prefill 800 tok/s (**met on large chunks**), decode 40+ (**open**).

| # | item | why | status |
|---|------|-----|--------|
| 1 | **Adaptive chunk sizing** so short prompts stop being PCIe-dominated | a 39-tok chunk reads ~8 GiB for 39 tok; a >=512-tok chunk amortizes | in progress |
| 2 | **Decode to 40+ tok/s**: spec-decode (MTP / `--spec`), resident hot experts for the verify pass, CPU-pool-vs-GPU split | decode is now the only category below target | open |
| 3 | **[LIT-1] Q2_0 Swift quant, A/B vs IQ2_XS** | GSQ-RCO card: 367 vs 108 prompt tok/s on the same model — lookup-table decode dominates | open |
| 4 | **Fix `--expert-cache-per-layer` `verify_slot` abort** | unblocks per-layer residency | open |
| 5 | **[LIT-4] Tiered residency + routing-predicted prefetch** (hot VRAM / warm arena / cold disk) | only lever that beats the PCIe link | open |
| 6 | **[LIT-3] Co-activation-ordered expert packing** (ZipMoE) | clean PCIe bursts for co-routed top-10 | open |
| 7 | **[LIT-5] Per-expert mixed precision + routing-preserving quant** | keep the Top-10 *set* stable under re-quant | open |
| 8 | **[LIT-6] Fuse gate/up/down into one expert kernel** (Mega-MoE) | fewer launches, less traffic | open |
| 9 | **[LIT-7] FP4/FP8 expert weights** | cuts PCIe bytes only above ~4 bpw; subordinate to #3 | open |
| 10 | **KV cache in VRAM** (block layout) | keep KV from displacing the expert hot set | open |

### 12.1 Multi-architecture backlog (see §14 for the DeepSeek plan)

| # | item | status |
|---|------|--------|
| 11 | **`--model-info`**: recognise every architecture + read its geometry, report support | **done** |
| 12 | **Parity reference** (llama.cpp on the same GGUF) — the gate every arch stage must pass | open |
| 13 | **DeepSeek-V4 pack tooling** (MLA tensors, Engram tables, sigmoid MoE 256/6/1 ff2048, MTP) + per-tensor xcheck | open |
| 14 | **DeepSeek-V4 layer graph** (RMSNorm → MLA → sparse select → output → hyper-connection → sigmoid MoE → Engram) | open |
| 15 | **MLA attention kernel** (naive-correct → partial RoPE 64d/yarn → compressed-KV → indexer) | open |
| 16 | **Compressed sparse attention** (compress_ratios 4/128, sliding 128, indexer top_k 512) | open |
| 17 | **Engram hash layers**; **MTP head** (`nextn_predict_layers 1`) | open |
| 18 | **`qwen35moe`** path (GDN + gated attention lineage) — smallest new model; warm-up before MLA | open |


## 13. Measured scores (gfx1201, RX 9070 XT, PCIe 4.0 x16)

All runs: Qwen3.8-Flash-Next Swift (IQ2_XS, 512 experts) via the native pack, `--expert-cache 2600
--expert-profile data/expert-profile.bin`.

| path | config | result | note |
|------|--------|--------|------|
| prefill, 39 tok | baseline (pre-fix) | 1153 ms / 33 tok/s | |
| prefill, 39 tok | + warm-up + VRAM-cache read (§11.8) | **485 ms / 80 tok/s** | 2.4x; small-chunk, PCIe-dominated (~8 GiB for 39 tok) |
| prefill, 2047 tok | default (`stream_all`) | 2168 ms / **943 tok/s** (871 e2e) | real prompt; experts mostly resident |
| prefill, 4096 tok | default (`stream_all`) | 579 tok/s | crosses `STREAM_ALL_MIN`, expert DMA streaming |
| prefill, 4096 tok | **resident** (`STRATA_STREAM_ALL_MIN` high) | **1191.8 tok/s** | **2.06x vs streaming; above the 800 tok/s target** |
| decode, 48 tok | `--spec 2` | **22.1 tok/s** | best; `--spec 4` 19.3, `--spec 6` 16.5 |
| coherence | real question via tokenizer | correct `<think>…</think>` + `def add(a, b): return a + b` | **model verified coherent** |

**Headline:** prefill target (800 tok/s) is **met and exceeded** (943–1192 tok/s depending on chunk);
the 80 tok/s figure was the 39-token-chunk artifact. Decode (22 tok/s) is the one open category vs the
40+ target. New tuning knob: `STRATA_STREAM_ALL_MIN` — the resident path is ~2x faster than the streamed
ring for IQ2_S at large chunks, so keep large chunks resident unless the streamed ring measured faster
for the format (it does for Q2_0).

**Known correctness bug:** under `--spec` the generator does not stop at `<|im_end|>` (248046) /
`<|endoftext|>` (248044) — it runs to `--max-new` and degenerates into `<|im_start|>` repetition after a
correct answer. The answer itself is coherent.

## 14. DeepSeek-V4 (`deepseek4`) support — staged plan

Scope read from the real metadata of `DeepSeek-V4-Flash-0731-K160-IQ2XXS-…-imatrix.gguf`:

```
arch deepseek4, block_count 43, embedding_length 4096, vocab 129280, ctx 1M
attention  MLA: head_count 64, head_count_kv 1, key_length 512, value_length 512,
           q_lora_rank 1024, output_lora_rank 1024, output_group_count 8,
           rope.dimension_count 64, rope.freq_base 10000, yarn factor 16 (orig 65536)
sparse     compress_ratios [0,0,4,128,4,128,...] (44 entries), indexer.head_count 64,
           indexer.key_length 128, indexer.top_k 512, sliding_window 128
engram     hash_layer_count 3
MoE        expert_count 256, expert_used_count 6, expert_shared_count 1,
           expert_feed_forward_length 2048, expert_gating_func 4 (sigmoid),
           expert_weights_norm true, expert_weights_scale 1.5
extra      hyper_connection.count 4 (≈ qwen4exp's hyper-connections), nextn_predict_layers 1 (MTP),
           swiglu_clamp_exp [10.0 * 43]
```

**Reused from qwen4exp:** the hyper-connection block (count 4), the generic MoE plumbing (routing →
grouped expert matmul → combine), the sampler, the batching/prefill engine. **New:** everything in the
attention tower.

**Reference implementation (this changes the plan from "write MLA from scratch" to "port"):**
[`antirez/ds4`](https://github.com/antirez/ds4) — DwarfStar 4 — is a model-specific DeepSeek-V4-Flash
runtime, and [`Anemll/ds4-ssd`](https://github.com/Anemll/ds4-ssd) is its SSD-streaming fork. This gives:

- **`ds4_cuda.cu`** (~480 KB) — the DS4 compute (matmul/dequant/embed/MLA/… kernels) **in CUDA**: port to
  HIP on gfx1201 the same way the `kernels/cuda` tree was ported for Strata (hipify + HANDLE/arch notes).
- **`tests/test-vectors/`** (`official.vec` + per-prompt `*.official.json`) — the **parity gate**: DS4's
  own correctness vectors, so a stage is verifiable without hand-building a reference.
- **`gguf-tools/deepseek4-quantize.c`** (+ `quants.[ch]`) — the DS4 quantizer (q8_0, q8_K, q4_K, q2_K,
  iq2_xxs); builds the GGUFs we already have in `G:\More-models`.
- **SSD sidecar / slot-bank** (`docs/SIDECAR.md`, `--moe-slot-bank`, `--ssd-cache`) — the *lazy-load*
  design (routed experts paged from disk through a bounded resident slot bank) asked for earlier.
- Upstream also carries **`gguf-tools/qwen4_exp_convert.py`** — a converter to the **`qwen4exp`** schema
  DS4 loads (MTP as `blk.<n>.nextn.*`), plus `qwen4_iq2.py` (IQ2_XXS+MXFP4+MTP) and the native n-gram
  packer. That is the tooling for the Q2/IQ2 "more Qwen" variants (LIT-1), independent of DeepSeek.

So stages 2-5 become a **CUDA→HIP port with a test-vector gate**, not new research. Stage 2 is now
"port `deepseek4-quantize` / build a DS4 pack (or read the existing GGUFs) and pin the test vectors",
not "invent a pack format".

Stages, each independently testable (parity against a reference before the next):

1. **Architecture selector + metadata schema.** Introduce an `Arch` enum (`qwen4exp`, `deepseek4`,
   `qwen35moe`) and route `ModelGeometry` population + tensor naming through it, replacing the single
   `check_architecture` gate. Deliverable: `deepseek4` models are *recognised* and their geometry is
   read into a `Ds4Geometry`; the engine still refuses to run until stage 4.
2. **Pack tooling for the DS4 tensors.** `tools/strata_pack.py` (or a `ds4_pack.py`) must map the MLA
   tensors (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` with 8 output groups),
   Engram hash tables, the 256-expert/6-used/1-shared MoE, and the MTP head into the pack layout, with a
   `canonical_xcheck.py`-style byte xcheck per tensor class.
3. **Layer graph.** A `Ds4Layer` that wires: RMSNorm → MLA proj (with the q/kv LoRA ranks) → sparse
   attention selection (compress_ratios 4/128 + sliding_window 128 + indexer top_k 512) → MLA output
   (8 groups, output_lora_rank 1024) → hyper-connection residual → sigmoid-gated MoE (256/6/1, ff 2048,
   weights_norm, scale 1.5, swiglu_clamp_exp 10) → Engram hash layers (first 3). Correctness first,
   speed second.
4. **MLA attention kernel.** Start with a *naive, correct* MLA (materialise the absorbed `q` and the
   latent KV, standard softmax) — no absorption, no flash. Verify per-layer logits against a reference.
   Then add: RoPE (partial, 64 dims, yarn), the compressed-KV cache for compress_ratio 128 layers, and
   the DSA indexer gate. This is the largest stage.
5. **Sparse / compressed attention.** Implement the compress-4 and compress-128 paths + the sliding
   window (128) + the indexer's top_k 512 selection. This is what makes 1M context tractable; get it
   correct before optimizing.
6. **Engram hash layers** (first 3 layers): the deterministic hash lookup (analogous to qwen4exp's PLE
   n-gram table) — host RAM / mmap, prefetched.
7. **MTP head** (`nextn_predict_layers 1`) for speculative decode — reuse the existing `--mtp` path.
8. **GPU tuning on gfx1201** — MMA shape selection, LDS budget (64 KB/block), the PCIe-4.0 budget, the
   hyper-connection FP8 state, eviction/residency — only after 1-6 pass parity.

**Effort:** stages 1-3 ≈ one session each; stage 4-5 are the project (MLA + sparse attention is a
kernel-suite); 6-8 follow. Each stage has a hard gate (byte/logit parity vs a reference before
proceeding). `qwen35moe` (§13 line item) is the smaller sibling — same GDN + gated-attention lineage as
qwen4exp — and is the recommended warm-up before the DeepSeek attention tower.







