#include "trt_toolkit/benchmark/memory_probe.hpp"

#include <cuda_runtime.h>

#include <iomanip>

#include "trt_toolkit/utils/cuda_helpers.hpp"

namespace trt_toolkit::benchmark {

namespace {

double mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

}  // namespace

void MemoryStats::format(std::ostream& os) const {
    os << std::fixed << std::setprecision(1)
       << "free_before=" << mib(free_before_bytes) << " MiB,  free_after="
       << mib(free_after_bytes) << " MiB\n"
       << "  used_delta=" << mib(used_delta_bytes()) << " MiB,  total=" << mib(total_bytes)
       << " MiB";
}

void MemoryProbe::before() {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    utils::cuda_check(cudaMemGetInfo(&free_b, &total_b), __FILE__, __LINE__);
    stats_.free_before_bytes = free_b;
    stats_.total_bytes = total_b;
}

MemoryStats MemoryProbe::after() {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    utils::cuda_check(cudaMemGetInfo(&free_b, &total_b), __FILE__, __LINE__);
    stats_.free_after_bytes = free_b;
    stats_.total_bytes = total_b;
    return stats_;
}

}  // namespace trt_toolkit::benchmark
