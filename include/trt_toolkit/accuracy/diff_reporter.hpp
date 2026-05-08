#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace trt_toolkit::accuracy {

/// Element-wise comparison between two float tensors of equal length.
struct DiffReport {
    std::size_t element_count = 0;
    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    /// Mean over (|a - b| / max(|a|, |b|, eps)).
    double mean_rel_diff = 0.0;
    double max_rel_diff = 0.0;
    /// Cosine similarity, 1.0 = identical direction.
    double cosine_similarity = 0.0;

    bool within_tolerance(double max_abs_thresh, double max_rel_thresh) const noexcept {
        return max_abs_diff <= max_abs_thresh && max_rel_diff <= max_rel_thresh;
    }

    void format(std::ostream& os) const;
};

/// Compare a TRT-engine output (`actual`) against a reference array
/// (`expected`); both are flat float32 vectors of the same length.
DiffReport compare_arrays(const float* expected, const float* actual, std::size_t n);

/// Convenience overload.
DiffReport compare_arrays(const std::vector<float>& expected,
                          const std::vector<float>& actual);

}  // namespace trt_toolkit::accuracy
