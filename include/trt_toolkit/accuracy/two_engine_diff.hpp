#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "trt_toolkit/accuracy/diff_reporter.hpp"

namespace trt_toolkit::accuracy {

/// Run the same input through two engines (typically an FP32 reference
/// and an INT8 candidate) and compare every output binding.
///
/// Both engines must have matching binding names and shapes. The input
/// data is whatever the user provides via `input_blobs`, in the order
/// the engines list their input bindings. Random input is the most
/// useful for tactic-induced numerical regressions; for representative
/// numerical fidelity, feed real preprocessed validation data.
struct EngineDiffOptions {
    std::vector<std::vector<float>> input_blobs;  ///< One per engine input, length = volume.
};

struct EngineDiffReport {
    /// One DiffReport per output binding, in declaration order. The
    /// `binding_name` field on the parent envelope mirrors the order.
    std::vector<DiffReport> per_output;
    std::vector<std::string> binding_names;
};

EngineDiffReport diff_engines(const std::filesystem::path& reference_engine,
                              const std::filesystem::path& candidate_engine,
                              const EngineDiffOptions& options);

}  // namespace trt_toolkit::accuracy
