#pragma once

#include <cstddef>
#include <ostream>

namespace trt_toolkit::benchmark {

struct MemoryStats {
    std::size_t free_before_bytes = 0;
    std::size_t free_after_bytes = 0;
    std::size_t total_bytes = 0;

    /// Resident GPU memory growth attributable to the work that ran
    /// between `before()` and `after()`. Note: this is approximate
    /// because other CUDA contexts on the same device can also move
    /// the free counter.
    std::size_t used_delta_bytes() const noexcept {
        return (free_before_bytes > free_after_bytes)
                   ? free_before_bytes - free_after_bytes
                   : 0;
    }

    void format(std::ostream& os) const;
};

class MemoryProbe {
public:
    MemoryProbe() = default;
    void before();
    MemoryStats after();

private:
    MemoryStats stats_;
};

}  // namespace trt_toolkit::benchmark
