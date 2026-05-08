#pragma once

#include <cstddef>
#include <functional>
#include <ostream>

namespace trt_toolkit::benchmark {

struct ThroughputStats {
    std::size_t iterations = 0;
    std::size_t batch_size = 1;
    double wall_seconds = 0.0;
    double samples_per_second = 0.0;
    double batches_per_second = 0.0;

    void format(std::ostream& os) const;
};

struct ThroughputProbeOptions {
    std::size_t warmup = 25;
    std::size_t iterations = 200;
    std::size_t batch_size = 1;
};

/// Throughput is intentionally measured back-to-back without inter-call
/// host synchronization between iterations - we synchronise once at the
/// end and divide. This matches how a saturated production loop would
/// look.
ThroughputStats measure_throughput(
    const std::function<void()>& enqueue,
    const std::function<void()>& sync,
    const ThroughputProbeOptions& options = {});

}  // namespace trt_toolkit::benchmark
