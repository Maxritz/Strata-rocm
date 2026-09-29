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
| B4 | hipify-clang validation | TODO: run on sample kernel before bulk conversion |
| B5 | Remaining Strata file reads | TODO: iq_kernels.cu, native_mmoq.cu, parity tests, session.cpp, moe_mmoq.cu |

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
- **GPU utilization**: misses still go to the CPU pool (48.8 ms/round). Need the miss path to also run on the GPU via a tiled `native_expert_grouped` (HipKittens-style shared-memory tiling) so the GPU is the bottleneck, not idle.
- **Real hot-expert profile**: build from `route_reap288.bin` (768 records), not the round-robin bootstrap profile (hit rate ~13/1579).
- **Speculation**: FIXED. The verify-window arena wipe in 11.3 caused the 0/N acceptance. Post-fix validation: `--spec 2`, `--spec 4` and `--spec 6` produce the *identical* greedy stream (they diverged before); feeding the engine's own continuation back as `--spec-oracle` accepts 12/12 at `--spec 4` and 8/8 at `--spec 2`; repeated runs are bit-identical. Decode 23.5 tok/s at `--spec 4` with accepted drafts (14.2 tok/s at `--spec 2` with all drafts rejected), measured on a 64-token run with `--expert-cache 2600`.
- **`hipModuleUnload ... while stream is capturing`**: NOT called by Strata; comes from the ROCm runtime tearing down a module (likely the lazily-loaded Q5_K head module / graph-captured module) mid-capture. Fix: defer that module's teardown past `hipStreamEndCapture`, or load the head module eagerly before capture begins. TODO in `src/core/native_head.cpp`.

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

