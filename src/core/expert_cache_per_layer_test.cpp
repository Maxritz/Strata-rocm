// src/core/expert_cache_per_layer_test.cpp - contract E7 / TODO P3 #29.
//
// `--expert-cache-per-layer` aborted at startup with
//   ExpertCache::verify_slot: slot 0 differs from the arena at byte 0
// because `open_sized` zeroed `layer_next_` after `open` had set each layer's start to its own range, so EVERY
// layer admitted into slot 0 and the first profile read-back compared two different experts.
//
// This checks the property that was false: under per-layer admission, each layer's admitted slots fall inside
// ITS range (`layer_slot_range`) and two layers never share a slot.
#include "strata/core/expert_cache.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const int64_t n_layers = 2, n_expert = 4;
    std::vector<int64_t> slot_bytes(8, 4096);   // >= 256-byte aligned inside open_sized
    strata::core::ExpertCache c;
    std::string err;
    if (!c.open_sized(slot_bytes, n_layers, n_expert, err)) {
        std::fprintf(stderr, "open_sized: %s\n", err.c_str());
        return 2;
    }
    c.set_per_layer_admission(true);

    int bad = 0;
    const int32_t a00 = c.admit(0, 0), a01 = c.admit(0, 1);
    const int32_t a10 = c.admit(1, 0), a11 = c.admit(1, 1);

    // q = 8 / 2 = 4: layer 0 owns [0,4), layer 1 owns [4,8).
    int64_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
    c.layer_slot_range(0, lo0, hi0);
    c.layer_slot_range(1, lo1, hi1);
    auto in_range = [](int32_t s, int64_t lo, int64_t hi) { return s >= lo && s < hi; };

    if (!in_range(a00, lo0, hi0) || !in_range(a01, lo0, hi0)) { std::fprintf(stderr, "layer 0 out of range\n"); ++bad; }
    if (!in_range(a10, lo1, hi1) || !in_range(a11, lo1, hi1)) { std::fprintf(stderr, "layer 1 out of range\n"); ++bad; }
    if (a00 == a10 || a01 == a11) { std::fprintf(stderr, "layers share a slot\n"); ++bad; }   // the bug
    if (c.slot_of(0, 0) != a00 || c.slot_of(1, 0) != a10) { std::fprintf(stderr, "slot_of mismatch\n"); ++bad; }

    c.close();
    std::printf("expert_cache per-layer: %s  layer0 slots [%lld,%lld) -> <%d,%d>, layer1 [%lld,%lld) -> <%d,%d>\n",
                bad ? "FAIL" : "PASS", (long long) lo0, (long long) hi0, a00, a01,
                (long long) lo1, (long long) hi1, a10, a11);
    return bad ? 1 : 0;
}
