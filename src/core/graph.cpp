// src/core/graph.cpp - P2.S5: the GraphRegistry implementation.
#include "strata/core/graph.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>

namespace strata::core {
namespace {

/// Fills `err` from the CUDA runtime, naming the call that failed.  A bare "invalid argument" with no call
/// site is the least useful error this API can produce and the easiest to avoid.
bool fail(std::string& err, const char* what, hipError_t e) {
    err = std::string(what) + ": " + hipGetErrorString(e);
    return false;
}

}  // namespace

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = o.graph_;
        exec_ = o.exec_;
        done_ = o.done_;
        nodes_ = o.nodes_;
        o.graph_ = nullptr;
        o.exec_ = nullptr;
        o.done_ = nullptr;
        o.nodes_ = 0;
    }
    return *this;
}

void CapturedGraph::reset() {
    // ORDER MATTERS: destroying a graph exec BLOCKS until the launch it is destroying has completed, so
    // destroying it before the event is harmless but destroying it after a spin that was supposed to observe
    // completion would silently become the thing that caused it.  `bench/micro/graph_capture.cu` was fooled by
    // exactly this once - it looked like captured graphs made a doorbell visible and direct launches did not.
    if (exec_) { hipGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { hipGraphDestroy(graph_); graph_ = nullptr; }
    if (done_) { hipEventDestroy(done_); done_ = nullptr; }
    nodes_ = 0;
}

bool CapturedGraph::begin(void* stream, std::string& err) {
    if (graph_ || exec_) { err = "begin: this CapturedGraph is already recorded"; return false; }
    const hipError_t e = hipStreamBeginCapture((hipStream_t) stream, hipStreamCaptureModeThreadLocal);
    if (e != hipSuccess) return fail(err, "hipStreamBeginCapture", e);
    return true;
}

bool CapturedGraph::end(void* stream, std::string& err) {
    hipError_t e = hipStreamEndCapture((hipStream_t) stream, &graph_);
    if (e != hipSuccess) { graph_ = nullptr; return fail(err, "hipStreamEndCapture", e); }

    nodes_ = 0;
    e = hipGraphGetNodes(graph_, nullptr, &nodes_);
    if (e != hipSuccess) return fail(err, "hipGraphGetNodes", e);
    // A capture that recorded NOTHING is a wiring mistake, and a graph that replays nothing produces no error
    // and no output - the silent kind of failure this project keeps paying for.
    if (nodes_ == 0) { err = "end: the capture recorded ZERO nodes - the body launched nothing"; reset(); return false; }

    e = hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (e != hipSuccess) return fail(err, "hipGraphInstantiate", e);

    // The completion event is recorded ONCE and reused: `launch` records it again after each replay, which is
    // what makes `wait_ms` a query rather than a sync.
    e = hipEventCreateWithFlags(&done_, hipEventDisableTiming);
    if (e != hipSuccess) return fail(err, "hipEventCreateWithFlags", e);
    return true;
}

bool CapturedGraph::launch(void* stream, std::string& err) const {
    if (!exec_) { err = "launch: not recorded"; return false; }
    hipError_t e = hipGraphLaunch(exec_, (hipStream_t) stream);
    if (e != hipSuccess) return fail(err, "hipGraphLaunch", e);
    e = hipEventRecord(done_, (hipStream_t) stream);
    if (e != hipSuccess) return fail(err, "hipEventRecord", e);
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    // A BOUNDED wait, and the bound is the point: an unbounded spin turns a protocol bug into a hung run, and
    // a hung run says nothing about which side is stuck.  `cudaEventQuery` is a QUERY - it does not block and
    // it does not synchronise - so this satisfies P2.X3 while still giving the driver the call it needs to
    // flush the submission.  See NOTE 1 in the header: without a driver call here the work never starts.
    //
    // The deadline is a REAL CLOCK, not a count of pause instructions: `_mm_pause` is a few cycles, so
    // counting pauses as microseconds would make the timeout tens of times longer than the caller asked for -
    // a timeout that does not time out is worse than none, because it reports a hang as a pass.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const hipError_t q = hipEventQuery(done_);
        if (q == hipSuccess) return true;
        if (q != hipErrorNotReady) return false;   // a real error, not "not finished"
        if (std::chrono::steady_clock::now() >= deadline) return false;
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;              // already recorded: the point of a registry

    CapturedGraph g;
    if (!g.begin(stream_, err)) return false;
    body();                                        // the caller launches into the captured stream
    // The body must not have failed silently.  An error state left on the stream would make EndCapture
    // succeed with a broken graph, so it is checked and cleared first.
    const hipError_t body_err = hipGetLastError();
    if (body_err != hipSuccess) {
        // Abandon the capture without instantiating anything.
        hipGraph_t junk = nullptr;
        hipStreamEndCapture((hipStream_t) stream_, &junk);
        if (junk) hipGraphDestroy(junk);
        return fail(err, "the capture body left a CUDA error", body_err);
    }
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    if (!g->wait_ms(timeout_ms)) { err = "launch: timed out waiting for the graph to complete"; return false; }
    return true;
}

}  // namespace strata::core
