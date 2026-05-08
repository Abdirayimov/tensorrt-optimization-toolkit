#include "trt_toolkit/benchmark/latency_probe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <stdexcept>

#include "trt_toolkit/utils/cuda_helpers.hpp"

namespace trt_toolkit::benchmark {

namespace {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= v.size()) return v[lo];
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

class CudaEvent {
public:
    CudaEvent() {
        utils::cuda_check(cudaEventCreate(&e_), __FILE__, __LINE__);
    }
    ~CudaEvent() {
        if (e_ != nullptr) cudaEventDestroy(e_);
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept : e_(other.e_) { other.e_ = nullptr; }

    cudaEvent_t get() const noexcept { return e_; }

private:
    cudaEvent_t e_ = nullptr;
};

}  // namespace

void LatencyStats::format(std::ostream& os) const {
    os << std::fixed << std::setprecision(3) << "iters=" << iterations
       << "  mean=" << mean_ms << "ms  stddev=" << stddev_ms << "ms\n"
       << "  p50=" << p50_ms << "ms  p90=" << p90_ms << "ms  p95=" << p95_ms
       << "ms  p99=" << p99_ms << "ms\n"
       << "  min=" << min_ms << "ms  max=" << max_ms << "ms";
}

LatencyStats measure_latency(const std::function<void()>& enqueue,
                             const LatencyProbeOptions& options) {
    if (!enqueue) {
        throw std::invalid_argument("measure_latency: enqueue must be non-null");
    }

    // Warm-up: kernels need their selectable tactics chosen, runtime
    // dispatcher caches need to populate, etc. We discard these.
    for (std::size_t i = 0; i < options.warmup; ++i) {
        enqueue();
    }
    utils::cuda_check(cudaDeviceSynchronize(), __FILE__, __LINE__);

    std::vector<CudaEvent> starts;
    starts.reserve(options.iterations);
    std::vector<CudaEvent> stops;
    stops.reserve(options.iterations);
    for (std::size_t i = 0; i < options.iterations; ++i) {
        starts.emplace_back();
        stops.emplace_back();
    }

    const auto deadline = options.max_wall_ms == 0
                              ? std::chrono::steady_clock::time_point::max()
                              : std::chrono::steady_clock::now() +
                                    std::chrono::milliseconds(options.max_wall_ms);

    std::size_t actual = 0;
    for (std::size_t i = 0; i < options.iterations; ++i) {
        if (std::chrono::steady_clock::now() > deadline) break;
        utils::cuda_check(cudaEventRecord(starts[i].get(), nullptr), __FILE__, __LINE__);
        enqueue();
        utils::cuda_check(cudaEventRecord(stops[i].get(), nullptr), __FILE__, __LINE__);
        ++actual;
    }
    utils::cuda_check(cudaDeviceSynchronize(), __FILE__, __LINE__);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(actual);
    for (std::size_t i = 0; i < actual; ++i) {
        float ms = 0.0f;
        utils::cuda_check(cudaEventElapsedTime(&ms, starts[i].get(), stops[i].get()),
                          __FILE__, __LINE__);
        latencies_ms.push_back(static_cast<double>(ms));
    }

    LatencyStats stats;
    stats.iterations = actual;
    if (actual == 0) return stats;

    double sum = 0.0;
    stats.min_ms = latencies_ms.front();
    stats.max_ms = latencies_ms.front();
    for (double v : latencies_ms) {
        sum += v;
        stats.min_ms = std::min(stats.min_ms, v);
        stats.max_ms = std::max(stats.max_ms, v);
    }
    stats.mean_ms = sum / static_cast<double>(actual);

    double sq_sum = 0.0;
    for (double v : latencies_ms) {
        const double d = v - stats.mean_ms;
        sq_sum += d * d;
    }
    stats.stddev_ms = std::sqrt(sq_sum / static_cast<double>(actual));

    stats.p50_ms = percentile(latencies_ms, 0.5);
    stats.p90_ms = percentile(latencies_ms, 0.9);
    stats.p95_ms = percentile(latencies_ms, 0.95);
    stats.p99_ms = percentile(latencies_ms, 0.99);
    return stats;
}

}  // namespace trt_toolkit::benchmark
