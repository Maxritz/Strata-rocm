#include "hip/hip_runtime.h"
// src/core/native_dense_3d_test.cpp - the deepseek2 native serving gate.
//
// The GLM pack marks every quantized non-expert tensor "native-only" (`dst_bytes == 0`), so the loader REFUSES the
// pack unless `NativeDense::served_names` names it and `NativeDense::load` attaches the raw GGUF block.  MLA's
// `attn_k_b`/`attn_v_b` are 3-D (`[nope, kv_lora, n_head]` / `[kv_lora, head_dim, n_head]`), which the 2-D-only
// code refused.  This test proves they are now served, with the right types and shapes, by loading the real pack.
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>

namespace {

const char* kPack = "H:/OLLAMA-Models/strata-pack-glm";
const char* kGguf = "H:/OLLAMA-Models/GGUF/GLM-4.7-Flash-APEX-I-Quality.gguf";

int g_bad = 0;
void require(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-56s %-6s %s\n", name.c_str(), ok ? "ok" : "*** BAD ***", detail.c_str());
    if (!ok) ++g_bad;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!std::filesystem::exists(std::string(kPack) + "/index.txt") || !std::filesystem::exists(kGguf)) {
        std::printf("native_dense_3d_test: SKIP (no GLM pack at %s)\n", kPack);
        return 0;
    }
    std::string err;
    std::set<std::string> skip;
    if (!strata::core::NativeDense::served_names({kGguf}, false, skip, err)) {
        std::printf("served_names: %s\n", err.c_str());
        return 1;
    }
    // The embedding and the tied head are served by other native components in the engine (the native head and
    // the embedding path); the pack marks them shape-only too, so a bare `WeightTable` must skip them.
    skip.insert("token_embd.weight");
    skip.insert("output.weight");
    require("attn_k_b is in the native-served set", skip.count("blk.1.attn_k_b.weight") == 1);
    require("attn_v_b is in the native-served set", skip.count("blk.1.attn_v_b.weight") == 1);
    require("attn_q_b is in the native-served set", skip.count("blk.1.attn_q_b.weight") == 1);
    require("dense ffn_down is in the native-served set", skip.count("blk.0.ffn_down.weight") == 1);

    uint64_t pool = 0;
    if (!strata::core::WeightTable::pool_bytes(kPack, pool, err, &skip)) {
        std::printf("pool_bytes: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (hipMalloc(&arena, pool) != hipSuccess) { std::printf("arena alloc failed\n"); return 1; }
    strata::core::WeightTable table;
    if (!table.load(kPack, arena, pool, err, &skip)) {
        std::printf("load: %s\n", err.c_str());
        return 1;
    }
    strata::core::NativeDense nd;
    if (!nd.load({kGguf}, table, err)) {
        std::printf("native dense load: %s\n", err.c_str());
        return 1;
    }
    const strata::core::WeightRef* kb = table.find("blk.1.attn_k_b.weight");
    require("attn_k_b has native data", kb && kb->native_data != nullptr);
    require("attn_k_b is Q8_0 (8)", kb && kb->native_type == 8, kb ? std::to_string(kb->native_type) : "null");
    require("attn_k_b shape is [192, 512]", kb && kb->ne0 == 192 && kb->ne1 == 512);
    const strata::core::WeightRef* vb = table.find("blk.1.attn_v_b.weight");
    require("attn_v_b has native data + Q6_K (14)", vb && vb->native_data != nullptr && vb->native_type == 14,
            vb ? std::to_string(vb->native_type) : "null");
    require("attn_v_b shape is [512, 256]", vb && vb->ne0 == 512 && vb->ne1 == 256);
    const strata::core::WeightRef* qb = table.find("blk.1.attn_q_b.weight");
    require("attn_q_b has native data + Q6_K (14)", qb && qb->native_data != nullptr && qb->native_type == 14);
    const strata::core::WeightRef* dn = table.find("blk.0.ffn_down.weight");
    require("dense ffn_down has native data", dn && dn->native_data != nullptr);

    hipFree(arena);
    std::printf("\nnative_dense_3d_test: %s (%d failing checks, %zu weights, %.1f MiB)\n",
                g_bad == 0 ? "PASS" : "FAIL", g_bad, nd.tensor_count(), (double) nd.weight_bytes() / 1048576.0);
    return g_bad == 0 ? 0 : 1;
}
