#pragma once

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace trt_toolkit::debug {

/// Per-binding metadata extracted from a serialized engine.
struct BindingDescriptor {
    std::string name;
    bool is_input = false;
    std::vector<std::int64_t> shape;
    int dtype_id = 0;       ///< nvinfer1::DataType cast to int
    std::string dtype_name; ///< Human-readable: "FP32", "FP16", ...
    std::size_t element_size = 0;
    std::size_t volume = 0;
    std::size_t bytes = 0;
};

/// Lightweight summary of an engine: name, version, layer count, total
/// device memory, list of bindings.
struct EngineSummary {
    std::string source_path;
    std::int32_t num_layers = 0;
    std::size_t device_memory_bytes = 0;
    std::int32_t num_optimization_profiles = 0;
    std::vector<BindingDescriptor> bindings;

    void format(std::ostream& os) const;
};

/// Load an engine (without running inference) and return a summary.
/// Useful for `trt-toolkit inspect ENGINE` style debugging.
EngineSummary inspect(const std::filesystem::path& engine_path);

}  // namespace trt_toolkit::debug
