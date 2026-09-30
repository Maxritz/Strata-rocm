// include/strata/core/ds2_prefill.hpp - the BATCHED deepseek2 prefill: a prompt chunk through the MLA + MoE
// together, instead of `ds2_token` once per token.  See docs/DEEPSEEK.md section 7 (the OPEN item) and
// docs/CONTRACT.md (blocker A).
//
// The dense projections are ONE batched quantized GEMM per weight (llama.cpp's MMQ over the chunk's T rows)
// rather than T per-token GEMVs, the attention is one causal kernel over the chunk (`ds2pf::attention`), and
// the routed experts are grouped so each DISTINCT expert is read and computed ONCE per chunk by the GPU
// grouped kernel (`native_expert_grouped`) reading the arena's device alias.  The per-token path in
// `ds2_token` is untouched.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

class WeightTable;
struct ModelGeometry;
struct SessionState;
class ExpertSource;

/// Device scratch for one prefill chunk of up to `max_T` tokens.  Allocated once (caller-owned arena) so the
/// token path pays no allocation.
struct Ds2PrefillScratch {
    int64_t max_T = 0, k = 0, n_layers = 0, n_ff_dense = 0;
    float *X = nullptr, *XN = nullptr, *QA = nullptr, *QAN = nullptr, *Q = nullptr, *QNOPE = nullptr;
    float *QPE = nullptr, *KVA = nullptr, *KLAT = nullptr, *KPE = nullptr, *QABS = nullptr, *ALAT = nullptr;
    float *ATTN = nullptr, *OUT = nullptr, *LOGITS = nullptr, *WEIGHTS = nullptr, *PARTS = nullptr, *SH = nullptr;
    float *DGATE = nullptr, *DUP = nullptr;   ///< intermediate buffers, sized max_T * n_ff_dense
    void* XBF16 = nullptr;
    void* XQ = nullptr;         ///< MMQ q8_1 scratch for the widest activation (reused per projection)
    void* EX_XQ = nullptr;      ///< block_q8_1 rows for `native_expert_grouped`
    void* EX_SCRATCH = nullptr; ///< `native_expert_scratch_bytes(max_T*k, n_ff)`
    int32_t *IDS = nullptr, *BOUNDS = nullptr, *IDS_ID = nullptr;
    int32_t *GRP_START = nullptr, *GRP_N = nullptr, *ENT_DST = nullptr, *ENT_TOK = nullptr;
    void* GRP_PTR = nullptr;    ///< unsigned long long[n_expert]
    void* HOST_IDS = nullptr;   ///< pinned int32[max_T*k]
    void* MMQ_CTX = nullptr;    ///< `mmq::Context*`
};

uint64_t ds2_prefill_scratch_bytes(const ModelGeometry& g, int64_t max_T, int64_t k, int64_t n_ff_dense);
bool ds2_prefill_scratch_init(const ModelGeometry& g, int64_t max_T, int64_t k, int64_t n_ff_dense, void* base,
                              Ds2PrefillScratch& s);

/// Run positions `[pos0, pos0+T)` of the prompt through every layer.  `sc.X` holds the T embeddings
/// (`n_embd` floats each, row-major) on entry and the residual on exit.  `src` supplies the expert blobs; its
/// device aliases are read over PCIe by the grouped kernel.
bool ds2_prefill(const WeightTable& tables, const ModelGeometry& g, int64_t pos0, int64_t T, SessionState& sess,
                 Ds2PrefillScratch& sc, ExpertSource& src, void* stream, std::string& err);

}  // namespace strata::core
