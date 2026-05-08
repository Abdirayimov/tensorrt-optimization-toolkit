#include "trt_toolkit/benchmark/throughput_probe.hpp"

#include <chrono>
#include <iomanip>
#include <stdexcept>

namespace trt_toolkit::benchmark {

void ThroughputStats::format(std::ostream& os) const {
    os << std::fixed << std::setprecision(2) << "iters=" << iterations
       << "  batch=" << batch_size << "  wall=" << wall_seconds << "s\n"
       << "  " << batches_per_second << " batches/s, " << samples_per_second << " samples/s";
}

ThroughputStats measure_throughput(const std::function<void()>& enqueue,
                                   const std::function<void()>& sync,
                                   const ThroughputProbeOptions& options) {
    if (!enqueue || !sync) {
        throw std::invalid_argument("measure_throughput requires enqueue + sync callbacks");
    }

    for (std::size_t i = 0; i < options.warmup; ++i) {
        enqueue();
    }
    sync();

    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < options.iterations; ++i) {
        enqueue();
    }
    sync();
    const auto t1 = std::chrono::steady_clock::now();

    ThroughputStats stats;
    stats.iterations = options.iterations;
    stats.batch_size = options.batch_size;
    stats.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
    if (stats.wall_seconds > 0.0) {
        stats.batches_per_second =
            static_cast<double>(options.iterations) / stats.wall_seconds;
        stats.samples_per_second =
            stats.batches_per_second * static_cast<double>(options.batch_size);
    }
    return stats;
}

}  // namespace trt_toolkit::benchmark
