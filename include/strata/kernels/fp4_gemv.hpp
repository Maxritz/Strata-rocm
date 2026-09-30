// include/strata/kernels/fp4_gemv.hpp - the MXFP4 / NVFP4 GEMV, host-callable (P2.S2).
//
// FP4 on gfx1201 is software.  There is no FP4 tensor core on this host (or on gfx1031); the formats are
// microscaling FP4, gated in `amd_hip_fp4.h` behind `HIP_ENABLE_GFX950_OCP_BUILTINS` and falling back to a
// host conversion path on everything that isn't gfx950/gfx1250.  So the kernel decodes the 4-bit codes here,
// in software, against the SAME scalar decoders `dequant_fp4_test` proves bit-identical to ggml.
//
// The matmul itself: y[o] = sum_i kvalues_fp4[code(i)] * block_scale * fp16_to_fp32(x[i]).  This is the
// Path-A (fp16 activations, fp32 accumulation) kernel - correct and parity-tested against the validated
// scalar reference.  The U44 dot-product path (Path B, `v_dot8_u32_u4` / `v_dot8_i32_iu4`, 8 codes/instr) is
// an optimization of the same decode, not a different one - it must reproduce this result within rounding.
//
// Why fp16 activations, not int4: the engine's activation contract is fp16 (`docs/activation-contract.md`),
// and the activation is BROADCAST across all n_out rows - one input vector feeds every output row, the same
// shape `s_gemv` uses for the legacy formats.  Quantizing the activation to int4 to use V_DOT8 is a later,
// separately-parity-tested step; shipping it first would make the activation quantization the first thing
// that can be wrong about this kernel.
#pragma once

#include <cstdint>

namespace strata::kernels {

// y[o] = sum_i weight_fp4(o, i) * fp16_to_fp32(x[i]).
//
// `x`   fp16 activations, n_in elements, one vector broadcast over all n_out rows.
// `w`   n_out * n_blocks_per_row weight blocks, row-major.  `mxfp4` selects the block layout:
//       MXFP4: 17 bytes/block (1x E8M0 scale + 16 bytes = 32 values), n_in must be a multiple of 32.
//       NVFP4: 36 bytes/block (4x UE4M3 scales + 32 bytes = 64 values), n_in must be a multiple of 64.
// `y`   n_out floats.
//
// The scale is INSIDE each block (`w` is not a separate scale array - that is the microscaling format),
// so the signature is deliberately different from `s_gemv` to make a wrong call site compile-error instead
// of silently feeding the wrong bytes as a scale.
void fp4_gemv(const uint16_t* x, const uint8_t* w, float* y,
              int64_t n_in, int64_t n_out, bool mxfp4);

// The SAME matmul, vectorized decode (8 codes per load via the codebook table, fp16 activations two at a
// time via __half22float2).  Numerics are identical to `fp4_gemv` aside from summation order, so it is
// parity-tested against the same scalar reference - not against the naive kernel alone.
void fp4_gemv_fast(const uint16_t* x, const uint8_t* w, float* y,
                   int64_t n_in, int64_t n_out, bool mxfp4);

// Tiled GEMV: `x` is a single vector of length `n_in` reused across all `n_out` rows (the MoE pattern),
// loaded once per 32-row block into shared memory so the kernel is no longer activation-bandwidth-bound.
// Numerically identical to the others (same dot order j=0..n_in), parity-tested against the scalar
// reference.  `n_in` must fit in the fixed shared-memory tile (kXTile = 4096).
void fp4_gemv_tiled(const uint16_t* x, const uint8_t* w, float* y,
                    int64_t n_in, int64_t n_out, bool mxfp4);

}  // namespace strata::kernels
