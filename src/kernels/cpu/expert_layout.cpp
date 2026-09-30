// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

// The CPU-probe lives HERE, not in expert.cpp: that file is compiled with -mavx512*, and a TU built for AVX-512
// may use AVX-512 in ANY of its code - including the probe and a namespace-scope dynamic initializer that runs
// before main.  On a CPU without AVX-512 that is a STATUS_ILLEGAL_INSTRUCTION (0xC000001D) before the runtime
// guard can skip.  This file carries no ISA flags, so the probe and the skip path are safe on any x86-64.
const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    static char buf[160];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s", avx512f ? "" : "AVX512F ",
                  avx512bw ? "" : "AVX512BW ", avx512vl ? "" : "AVX512VL ",
                  avx512_vnni ? "" : "AVX512-VNNI ", avx512_vbmi ? "" : "AVX512-VBMI");
    return buf;
}

CpuFeatures cpu_features() {
    CpuFeatures f;
    int reg[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(reg, 0);
    if (reg[0] < 7) return f;
    __cpuidex(reg, 7, 0);
#else
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return f;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
    for (int i = 0; i < 4; ++i) reg[i] = (int) r[i];
#endif
    const unsigned ebx = (unsigned) reg[1], ecx = (unsigned) reg[2];
    f.avx512f = (ebx >> 16) & 1u;
    f.avx512bw = (ebx >> 30) & 1u;
    f.avx512vl = (ebx >> 31) & 1u;
    f.avx512_vnni = (ecx >> 11) & 1u;
    f.avx512_vbmi = (ecx >> 1) & 1u;
    return f;
}

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (f.usable()) return;
    std::fprintf(stderr,
                 "strata: this CPU cannot run the expert kernel: %s.\n"
                 "        The engine needs AVX512-VNNI and AVX512-VBMI (Intel Ice Lake / AMD Zen 4 or newer).\n"
                 "        The scalar fallback exists for tests only and is far too slow to decode with.\n",
                 f.reason());
    std::exit(1);
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

// The canonical Q2_0 pack records its own expert count in `manifest.json`
// ("n_experts_per_layer").  Read it so a 288-expert Flash-Next artifact is not
// forced through the engine's 512 default.  Returns 0 when absent/unreadable.
static int64_t manifest_expert_count(const std::string& pack_dir) {
    std::ifstream in(pack_dir + "/manifest.json");
    if (!in) return 0;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string key = "\"n_experts_per_layer\"";
    const size_t p = s.find(key);
    if (p == std::string::npos) return 0;
    const size_t c = s.find(':', p + key.size());
    if (c == std::string::npos) return 0;
    return (int64_t) std::strtoll(s.c_str() + c + 1, nullptr, 10);
}

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        const int64_t ne = manifest_expert_count(pack_dir);
        if (ne > 0) L.n_expert = ne;   // the artifact is authoritative over the engine default
        L.total = (uint64_t) n_layers * (uint64_t) L.n_expert * (uint64_t) BLOB;
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    // **THE MODEL'S EXPERT WIDTHS, NOT THE COMPILED 2560/640.**  A v4 header carries `n_embd`/`n_ff`; a v3 one
    // does not, and the compiled defaults (H, FF) are then correct for the 2560 artifact.
    int64_t n_embd = H, n_ff = FF;
    const auto header_int = [&](const std::string& ln, const char* tag, int64_t dflt) -> int64_t {
        const std::string t(tag);
        const size_t p = ln.find(t);
        if (p == std::string::npos) return dflt;
        return (int64_t) std::strtoll(ln.c_str() + p + t.size(), nullptr, 10);
    };
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            // The header records the artifact's own expert count: "(n_expert <N>, n_embd <E>, n_ff <F>, ...)".
            // The engine's 512 default is wrong for a 288-expert reap artifact, so the table is authoritative.
            const std::string tag = "(n_expert ";
            const size_t p = line.find(tag);
            if (p != std::string::npos) {
                const int64_t ne = (int64_t) std::strtoll(line.c_str() + p + tag.size(), nullptr, 10);
                if (ne > 0) { n_expert = ne; L.n_expert = ne; }
            }
            n_embd = header_int(line, "n_embd ", n_embd);
            n_ff = header_int(line, "n_ff ", n_ff);
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, n_embd, n_ff, f, err)) return false;
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            std::string file;             // v3: the shard that holds this layer (a file name beside --native)
            if (ss >> file) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) n_layers, std::string());
                L.gguf_file[(size_t) l] = file;
            }
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    // **THE TABLE'S OWN LAYER COUNT.**  `expert_layout_load` is called before the model geometry is read, so it
    // is handed the engine default (48).  A pack with fewer layers (Whittle has 40) must set its own - the last
    // layer seen in the table is authoritative, and a table with MORE layers than expected is still refused.
    int64_t max_layer = -1;
    for (int64_t l = 0; l < n_layers; ++l)
        if (L.offset[(size_t) l] != ~0ull) max_layer = l;
    if (max_layer >= 0 && max_layer + 1 < n_layers) { n_layers = max_layer + 1; L.n_layers = n_layers; }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) n_expert;
    }
    L.total = at;
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu
