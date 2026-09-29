// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
// ggml's AMD architecture id: `gfx1201` -> `GGML_CUDA_CC_OFFSET_AMD + 0x1201`.  ggml-cuda.cu computes this with
// `ggml_cuda_parse_id(prop.gcnArchName)`, but that function is static in a file this build does not compile, so the
// parse is repeated here.  It matters: `ggml_cuda_mmq_get_config` selects the MMQ config by `cc`, and with the CUDA
// formula (`100*major + 10*minor`, no AMD offset) `GGML_CUDA_CC_IS_AMD` is false for a Radeon, so the host picked a
// CUDA config while the device code was compiled as RDNA4 - every i-quant MMQ launch failed as `NO_DEVICE_CODE`.
int gcn_arch_id(const char* dev_name) {
    int major = 0, minor = 0;
    char name[64];
    size_t n = std::strlen(dev_name);
    if (n > 3) { std::memcpy(name, dev_name + 3, n - 3); n -= 3; } else { n = 0; }
    if (n >= sizeof name) n = sizeof name - 1;
    name[n] = '\0';
    n = std::strcspn(name, ":");            // trim a trailing ":xnack-" / ":sramecc-" status
    name[n] = '\0';
    if (n > 8 && std::strstr(name, "-generic")) {
        name[n - 8] = '\0';
        if (char* p = std::strtok(name, "-")) {
            major = (int) std::strtoul(p, nullptr, 16);
            if ((p = std::strtok(nullptr, "-"))) minor = 0x10 * (int) std::strtoul(p, nullptr, 16);
        }
    } else if (n >= 3) {
        minor = (int) std::strtoul(&name[n - 2], nullptr, 16);   // last two hex digits: minor*0x10 + stepping
        name[n - 2] = '\0';
        major = (int) std::strtoul(name, nullptr, 16);
    }
    return GGML_CUDA_CC_OFFSET_AMD + major * 0x100 + minor;
}
}  // namespace

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(hipGetDevice(&id));
    return id;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            hipDeviceProp_t prop;
            CUDA_CHECK(hipGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
            // The AMD arch id, not the CUDA formula: everything downstream (the MMQ config table, the MMA
            // availability checks) predicates on GGML_CUDA_CC_IS_AMD/RDNA*, which the CUDA form never matches.
            d.cc = gcn_arch_id(prop.gcnArchName);
            if ((d.cc & 0xff00) == 0 && prop.major > 0)   // a name the parse did not understand: the props fallback
                d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 + prop.minor * 0x10;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.integrated = prop.integrated != 0;
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.warp_size = prop.warpSize;
            d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(hipMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) hipFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
