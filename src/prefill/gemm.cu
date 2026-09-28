// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

void ck(hipblasStatus_t s, const char* what) {
    if (s != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

}  // namespace

Gemm::~Gemm() {
    if (handle_) hipblasDestroy((hipblasHandle_t) handle_);
    if (!external_) {
        if (scratch_) hipFree(scratch_);
        if (workspace_) hipFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    hipblasHandle_t h = nullptr;
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) { err = "prefill gemm: hipblasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    hipblasSetStream(h, (hipStream_t) stream);
    workspace_ = workspace;
    hipblasSetWorkspace(h, workspace_, ws_bytes);
    hipblasSetMathMode(h, HIPBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    hipblasSetWorkspace((hipblasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    hipblasHandle_t h = nullptr;
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) { err = "prefill gemm: hipblasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    hipblasSetStream(h, (hipStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (hipMalloc(&workspace_, ws) != hipSuccess) { err = "prefill gemm: workspace"; return false; }
    hipblasSetWorkspace(h, workspace_, ws);
    hipblasSetMathMode(h, HIPBLAS_DEFAULT_MATH);
    if (scratch_elems > 0 && hipMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != hipSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(hipblasGemmEx((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    HIP_R_16BF, (int) K, X, HIP_R_16BF, (int) K, &beta, Y, HIP_R_32F, (int) ldy,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
       "hipblasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(hipblasGemmEx((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    HIP_R_16F, (int) K, X, HIP_R_16F, (int) K, &beta, Y, HIP_R_32F, (int) ldy,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
       "hipblasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
