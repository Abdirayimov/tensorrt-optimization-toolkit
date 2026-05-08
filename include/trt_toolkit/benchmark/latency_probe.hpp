#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <vector>

namespace trt_toolkit::benchmark {

struct LatencyStats {
    std::size_t iterations = 0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double stddev_ms = 0.0;

    void format(std::ostream& os) const;
};

struct LatencyProbeOptions {
    std::size_t warmup = 25;
    std::size_t iterations = 200;
    /// If non-zero, abort once the wall clock exceeds this many ms even
    /// if `iterations` has not been reached. Useful in CI loops.
    std::size_t max_wall_ms = 0;
};

/// Time a callable using CUDA events.
///
/// The callable is expected to enqueue work on `stream` (typically a
/// `TrtRunner::infer()` call) and *not* synchronize - the probe inserts
/// start / stop events around each invocation and synchronises once at
/// the end.
LatencyStats measure_latency(
    const std::function<void()>& enqueue,
    const LatencyProbeOptions& options = {});

}  // namespace trt_toolkit::benchmark
