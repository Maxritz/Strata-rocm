#pragma once
// Strata AMDGPU device-intrinsic layer.
//
// CUDA intrinsics that have no HIP spelling are mapped either to the AMD
// instruction that implements them, or to SWAR when the ISA has no packed-byte
// ALU. Nothing here is a stub: every entry is the real op for the target ISA.
//
// Target ISAs: RDNA4 (gfx120x), RDNA3 (gfx11xx), RDNA2 (gfx103x), CDNA (gfx9xx).

#ifdef __HIP__

#include <hip/hip_runtime.h>
#include <cmath>

// ---- __dp4a: 4x int8 dot product + 32-bit accumulator ----------------------
// CUDA sm_61+ __dp4a. AMD equivalent is V_DOT4_I32_I8:
//   RDNA3 / RDNA4 : __builtin_amdgcn_sudot4  (signed x signed via two negate flags)
//   CDNA / RDNA2 / gfx906 : __builtin_amdgcn_sdot4
// Pre-dot ISAs fall back to the scalar form; there is no instruction to use.
__device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(RDNA4) || defined(RDNA3) || \
    defined(__gfx1200__) || defined(__gfx1201__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(CDNA) || defined(RDNA2) || defined(__gfx906__) || \
      defined(__gfx1030__) || defined(__gfx1031__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t * pa = reinterpret_cast<const int8_t *>(&a);
    const int8_t * pb = reinterpret_cast<const int8_t *>(&b);
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
#endif
}

// ---- __vsub4: per-byte wrapping subtract (no borrow across bytes) ----------
// SWAR: (a | 0x80) - (b & 0x7f) keeps each byte result in [0x7f, 0xfe] so no
// borrow escapes; the xor restores the bit-7 parity of the true difference.
__device__ __forceinline__ int __vsub4(const int a, const int b) {
    const unsigned H = 0x80808080u;
    const unsigned ua = static_cast<unsigned>(a);
    const unsigned ub = static_cast<unsigned>(b);
    return static_cast<int>((((ua | H) - (ub & ~H)) ^ ((ua ^ ~ub) & H)));
}

// ---- __vcmpne4: 0xFF in every byte where a_i != b_i -----------------------
// SWAR: flag bytes whose low 7 bits are nonzero, OR in the raw high bits (so a
// byte of exactly 0x80 is caught), then smear each 0x80 flag to 0xFF.
__device__ __forceinline__ int __vcmpne4(const int a, const int b) {
    const unsigned H = 0x80808080u;
    const unsigned x = static_cast<unsigned>(a) ^ static_cast<unsigned>(b);
    const unsigned u = (x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu;
    const unsigned nz = (u | x) & H;
    return static_cast<int>(nz | (nz - (nz >> 7)));
}

// ---- __vsubss4: per-byte signed saturating subtract -----------------------
// No packed saturating byte op exists on AMD; this is the operation itself.
__device__ __forceinline__ int __vsubss4(const int a, const int b) {
    unsigned result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int av = static_cast<int8_t>(static_cast<unsigned>(a) >> (8 * i));
        const int bv = static_cast<int8_t>(static_cast<unsigned>(b) >> (8 * i));
        int d = av - bv;
        d = d > 127 ? 127 : (d < -128 ? -128 : d);
        result |= static_cast<unsigned>(d & 0xFF) << (8 * i);
    }
    return static_cast<int>(result);
}

// ---- __fmaf_rn: fused multiply-add, round to nearest even -----------------
// Identical to v_fma_f32, which is what fmaf() lowers to.
#define __fmaf_rn(a, b, c) fmaf((a), (b), (c))

// __fmul_rn / __fadd_rn / __fmaf_rz / __int_as_float / __float_as_int /
// __byte_perm are provided natively by HIP's device headers.

// ---- __nanosleep: wave sleep ----------------------------------------------
// CUDA __nanosleep(ns). AMD s_sleep takes a compile-time immediate (cycle
// count), so the ns hint is scaled at the ~1 GHz shader clock and clamped to
// the 8-bit immediate. The argument must be constant, as CUDA's is here.
#define __nanosleep(ns) __builtin_amdgcn_s_sleep( \
    ((((ns) / 1000) + 1) > 0xFF) ? 0xFF : (((ns) / 1000) + 1))

// __shfl_sync / __shfl_up_sync / __shfl_down_sync / __shfl_xor_sync are
// provided by HIP's runtime with a default width; no local mapping is needed.
// __fmul_rn / __fadd_rn / __int_as_float / __float_as_int likewise.

#else
// CUDA path: the above are compiler builtins / device functions already.
#include <cuda_runtime.h>
#endif
