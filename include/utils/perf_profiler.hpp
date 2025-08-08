#pragma once
#include <chrono>
#include <atomic>
#include <string>
#include <vector>
#include <mutex>
#include <array>
#include <unordered_map>

namespace stereo_vision {
namespace perf {

enum class Stage {
    Capture = 0,
    Rectification,
    Disparity,
    PointCloud,
    ONNXInference,
    UIUpdate,
    COUNT
};

struct StatBucket {
    std::atomic<uint64_t> count{0};
    std::atomic<double> sum_ms{0.0};
    std::atomic<double> min_ms{1e9};
    std::atomic<double> max_ms{0.0};
    std::atomic<double> ema_ms{0.0};
};

struct Snapshot {
    uint64_t count{0};
    double sum_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
    double ema_ms{0.0};
};

class ProfilerRegistry {
public:
    static ProfilerRegistry& instance();
    void enable(bool on) { enabled_.store(on); }
    bool enabled() const { return enabled_.load(); }

    void record(Stage stage, double ms);
    Snapshot get(Stage stage) const;
    void reset();
    std::unordered_map<std::string, double> snapshotAverages() const;

private:
    ProfilerRegistry();
    std::array<StatBucket, static_cast<size_t>(Stage::COUNT)> buckets_;
    std::atomic<bool> enabled_{false};
};

class ScopedTimer {
public:
    explicit ScopedTimer(Stage stage) : stage_(stage), start_(Clock::now()) {}
    ~ScopedTimer();
private:
    using Clock = std::chrono::high_resolution_clock;
    Stage stage_;
    Clock::time_point start_;
};

const char* stageName(Stage s);

} // namespace perf
} // namespace stereo_vision
