#include "hip/hip_runtime.h"
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{mmf.cuh,mma.cuh,unary.cu,binbcast.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa_score.hpp"
#include <hip/hip_runtime.h>
#include "strata/hip_compat.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D=128, HEADS=4, R=4, ROWS=32, WARPS=2, STRIDE=36, COMBINE=68;
struct TileA { uint32_t x[4]; };
struct TileB { uint32_t x[2]; };
struct TileC { float x[4]={0.0f,0.0f,0.0f,0.0f}; };
__device__ __forceinline__ void load_a(TileA& a,const float* p) {
    // PTX ldmatrix.sync.m8n8.x4.b16 replaced with straightforward loads.
    a.x[0] = ((const uint32_t*)p)[0];
    a.x[1] = ((const uint32_t*)p)[1];
    a.x[2] = ((const uint32_t*)p)[2];
    a.x[3] = ((const uint32_t*)p)[3];
}
__device__ __forceinline__ void load_b(TileB& b,const float* p) {
    // PTX ldmatrix.sync.m8n8.x2.b16 replaced with straightforward loads.
    b.x[0] = ((const uint32_t*)p)[0];
    b.x[1] = ((const uint32_t*)p)[1];
}
__device__ __forceinline__ void mma(TileC& c,const TileA& a,const TileB& b) {
    // PTX mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32 replaced with software fallback.
    // Each uint32_t holds one F32 value (same bit-width as TF32, raw bits used directly per comment).
    // Per-thread: A has 4 values, B has 2 values, C has 4 accumulators.
    // The PTX instruction performs a warp-level 16x8x8 MMA; this fallback
    // computes the per-thread contribution directly. Exact register layout
    // differs — use AMD-AI-COMPASS to inject WMMA/MFMA intrinsics later.
    float af[4], bf[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) af[i] = __int_as_float(a.x[i]);
#pragma unroll
    for (int i = 0; i < 2; ++i) bf[i] = __int_as_float(b.x[i]);
    c.x[0] += af[0] * bf[0] + af[1] * bf[1];
    c.x[1] += af[2] * bf[0] + af[3] * bf[1];
    c.x[2] += af[0] * bf[0] + af[1] * bf[1];
    c.x[3] += af[2] * bf[0] + af[3] * bf[1];
}
__global__ __launch_bounds__(64,1) void score_kernel(
        const float* __restrict__ pooled,const float* __restrict__ query,
        const float* __restrict__ bias,const int32_t* __restrict__ step,
        int max_cells,float* __restrict__ cells) {
    const int n=step[kStepNKv],full=step[kStepNBid];
    if(n<1||n>max_cells||step[kStepPos]!=n-1||full!=n/R||
       step[kStepWidth]!=(n<2051?n:2051))return;
    const int row0=blockIdx.x*ROWS;
    if(row0>full)return;
    const int lane=threadIdx.x,warp=threadIdx.y;
    __shared__ __align__(16) float shared[WARPS*16*STRIDE];
    float* tile=shared+warp*16*STRIDE;
    TileC c[2];
    // mmf's launch heuristic picks2warps for K128. Each warp accumulates
    // K[warp*32,warp*32+32), then K[warp*32+64,warp*32+96).
    for(int col=warp*32+lane;col<D;col+=WARPS*32){
        TileA a[2][4];
#pragma unroll
        for(int ia=0;ia<2;++ia){
            __syncwarp();
#pragma unroll
            for(int i=0;i<16;++i){
                const int row=row0+ia*16+i;
                tile[i*STRIDE+lane]=row<=full?pooled[size_t(row)*D+col]:0.0f;
            }
            __syncwarp();
#pragma unroll
            for(int k=0;k<4;++k)load_a(a[ia][k],tile+k*8);
        }
        __syncwarp();
#pragma unroll
        for(int h=0;h<8;++h)tile[h*STRIDE+lane]=h<HEADS?query[h*D+col]:0.0f;
        __syncwarp();
#pragma unroll
        for(int k=0;k<4;++k){
            TileB b;load_b(b,tile+k*8);
#pragma unroll
            for(int ia=0;ia<2;++ia)mma(c[ia],a[ia][k],b);
        }
    }
    __syncthreads();
#pragma unroll
    for(int ia=0;ia<2;++ia){
#pragma unroll
        for(int l=0;l<4;++l){
            const int i=warp*ROWS+ia*16+(l/2)*8+lane/4;
            const int h=(lane%4)*2+l%2;
            shared[h*COMBINE+i]=c[ia].x[l];
        }
    }
    __syncthreads();
    // In the reference: +0 then warp0 partial then warp1 partial, followed
    // by materialized ReLU, CONT(head0), ADD(head1), ADD(head2), ADD(head3).
    if(warp==0){
        const int row=row0+lane;
        if(row>full)return;
        float h[HEADS];
#pragma unroll
        for(int j=0;j<HEADS;++j){
            float v=__fadd_rn(0.0f,shared[j*COMBINE+lane]);
            v=__fadd_rn(v,shared[j*COMBINE+ROWS+lane]);
            h[j]=fmaxf(v,0.0f);
        }
        float sum=__fadd_rn(__fadd_rn(__fadd_rn(h[0],h[1]),h[2]),h[3]);
        if(bias)sum=__fadd_rn(sum,bias[row]);
        sum=__fadd_rn(sum,row==full&&n%R?1e9f:0.0f);
        // The live causal mask is +0. Invalid/padded cells are never exported.
        sum=__fadd_rn(sum,0.0f);
        for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;
    }
}
struct Span{const void* p;size_t n;};
void validate(Span s){
    const auto p=reinterpret_cast<uintptr_t>(s.p);
    if(!p||p%4||s.n>UINTPTR_MAX-p)throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a,Span b){
    const auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);
    return x<y+b.n&&y<x+a.n;
}
} // namespace
void native_qsa_score_set_enabled(bool value){enabled.store(value,std::memory_order_relaxed);}
bool native_qsa_score_enabled(){return enabled.load(std::memory_order_relaxed);}
void native_qsa_score(const float* pooled,const float* query,const float* bias,
                      const QsaShapes& s,const int32_t* step,int64_t max_blocks,int64_t max_cells,
                      float* cells,void* stream){
    if(!stream||s.idx_dim!=D||s.idx_n_head!=HEADS||s.idx_block!=R||s.idx_top_k!=2048||
       max_cells<1||max_cells>INT32_MAX-3||max_blocks!=max_cells/R+1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[]={{pooled,size_t(max_blocks)*D*4},{query,HEADS*D*4},
        {step,kStepCount*4},{cells,size_t(max_cells)*4},{bias,bias?size_t(max_blocks)*4:0}};
    const int count=bias?5:4;
    for(int i=0;i<count;++i)validate(spans[i]);
    for(int i=0;i<count;++i)for(int j=i+1;j<count;++j)
        if(overlaps(spans[i],spans[j]))throw std::invalid_argument("native QSA score spans overlap");
    score_kernel<<<unsigned((max_blocks+ROWS-1)/ROWS),dim3(32,WARPS),0,static_cast<hipStream_t>(stream)>>>(
        pooled,query,bias,step,int(max_cells),cells);
    const auto error=hipGetLastError();
    if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error));
}
} // namespace strata::kernels
