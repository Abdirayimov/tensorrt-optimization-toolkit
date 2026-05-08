#pragma once

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace trt_toolkit::debug {

struct OnnxIO {
    std::string name;
    std::vector<std::int64_t> shape;  ///< -1 for dynamic axes.
    std::string dtype;                ///< "float32", "float16", ...
};

struct OnnxOpSummary {
    std::string op_type;
    std::int32_t count = 0;
};

struct OnnxSummary {
    std::string source_path;
    std::int64_t ir_version = 0;
    std::int64_t opset_version = 0;
    std::string producer;
    std::vector<OnnxIO> inputs;
    std::vector<OnnxIO> outputs;
    std::vector<OnnxOpSummary> ops;

    void format(std::ostream& os) const;
};

/// Parse an ONNX file and emit a summary similar to `polygraphy
/// inspect model`. Used by the CLI's future `inspect onnx` subcommand
/// to spot missing dynamic axes, unsupported ops, and producer
/// mismatches before invoking TRT.
///
/// Implementation note: this version uses TRT's own ONNX parser to
/// load the model, then walks the network. A pure protobuf-based
/// reader would not require TRT but would duplicate the upstream
/// parser's quirks; the trade-off is intentional.
OnnxSummary inspect_onnx(const std::filesystem::path& onnx_path);

}  // namespace trt_toolkit::debug
