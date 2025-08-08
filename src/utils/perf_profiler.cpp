#include "utils/perf_profiler.hpp"

namespace stereo_vision { namespace perf {

ProfilerRegistry& ProfilerRegistry::instance() {
    static ProfilerRegistry inst;
    return inst;
}

ProfilerRegistry::ProfilerRegistry() {
    for (auto &b : buckets_) {
        b.count.store(0);
        b.sum_ms.store(0.0);
        b.min_ms.store(1e9);
        b.max_ms.store(0.0);
        b.ema_ms.store(0.0);
    }
}

void ProfilerRegistry::record(Stage stage, double ms) {
    if (!enabled()) return;
    auto &b = buckets_[static_cast<size_t>(stage)];
    b.count.fetch_add(1, std::memory_order_relaxed);
    double prev_sum = b.sum_ms.load(std::memory_order_relaxed);
    b.sum_ms.store(prev_sum + ms, std::memory_order_relaxed);
    // Min
    double cur_min = b.min_ms.load(std::memory_order_relaxed);
    if (ms < cur_min) b.min_ms.store(ms, std::memory_order_relaxed);
    // Max
    double cur_max = b.max_ms.load(std::memory_order_relaxed);
    if (ms > cur_max) b.max_ms.store(ms, std::memory_order_relaxed);
    // EMA
    double alpha = 0.1;
    double prev_ema = b.ema_ms.load(std::memory_order_relaxed);
    double new_ema = (prev_ema == 0.0) ? ms : (alpha * ms + (1 - alpha) * prev_ema);
    b.ema_ms.store(new_ema, std::memory_order_relaxed);
}

Snapshot ProfilerRegistry::get(Stage stage) const {
    const auto &b = buckets_[static_cast<size_t>(stage)];
    Snapshot s;
    s.count = b.count.load();
    s.sum_ms = b.sum_ms.load();
    s.min_ms = b.min_ms.load();
    s.max_ms = b.max_ms.load();
    s.ema_ms = b.ema_ms.load();
    return s;
}

void ProfilerRegistry::reset() {
    for (auto &b : buckets_) {
        b.count.store(0);
        b.sum_ms.store(0.0);
        b.min_ms.store(1e9);
        b.max_ms.store(0.0);
        b.ema_ms.store(0.0);
    }
}

std::unordered_map<std::string, double> ProfilerRegistry::snapshotAverages() const {
    std::unordered_map<std::string, double> out;
    for (size_t i = 0; i < static_cast<size_t>(Stage::COUNT); ++i) {
        auto &b = buckets_[i];
        uint64_t c = b.count.load();
        if (c > 0) {
            double avg = b.sum_ms.load() / static_cast<double>(c);
            out[stageName(static_cast<Stage>(i))] = avg;
        }
    }
    return out;
}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    ProfilerRegistry::instance().record(stage_, ms);
}

const char* stageName(Stage s) {
    switch (s) {
        case Stage::Capture: return "Capture";
        case Stage::Rectification: return "Rectification";
        case Stage::Disparity: return "Disparity";
        case Stage::PointCloud: return "PointCloud";
        case Stage::ONNXInference: return "ONNXInference";
        case Stage::UIUpdate: return "UIUpdate";
        default: return "Unknown";
    }
}

}} // namespace
