#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "trt_toolkit/builder/dynamic_shapes.hpp"

namespace nvinfer1 {
class IInt8Calibrator;
}

namespace trt_toolkit::builder {

enum class Precision {
    FP32,
    FP16,
    INT8,
};

struct BuildOptions {
    Precision precision = Precision::FP16;

    /// Workspace size in MiB. Modern engines benefit from generous
    /// workspaces for tactic search; 4 GiB is a safe default on a
    /// 24 GB GPU.
    std::size_t workspace_mib = 4096;

    /// Optional dynamic-shape profile. Empty means "use whatever the
    /// ONNX file declares".
    ShapeProfile profile;

    /// INT8 calibrator. Required when precision == INT8 unless a cache
    /// is already available at the configured path.
    nvinfer1::IInt8Calibrator* calibrator = nullptr;

    /// Disable specific optimizations to reproduce or debug numerical
    /// issues. Maps to BuilderFlag::kPREFER_PRECISION_CONSTRAINTS.
    bool strict_types = false;

    /// Set the target SM (e.g. "8.6" for Ampere). Empty -> let TRT
    /// pick based on the runtime device.
    std::string target_sm;

    /// Logger verbosity for `trtexec`-style "what tactic was chosen"
    /// output.
    bool verbose = false;
};

class TrtBuilder {
public:
    TrtBuilder();
    ~TrtBuilder();

    TrtBuilder(const TrtBuilder&) = delete;
    TrtBuilder& operator=(const TrtBuilder&) = delete;

    /// Compile an ONNX file into a serialized engine.
    ///
    /// @param onnx_path        Source ONNX (must already be simplified
    ///                         if needed; this builder does not invoke
    ///                         onnx-simplifier).
    /// @param output_path      Destination .engine file. Parent
    ///                         directory must exist.
    /// @param options          Precision, profile, calibrator, etc.
    void build(const std::filesystem::path& onnx_path,
               const std::filesystem::path& output_path,
               const BuildOptions& options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace trt_toolkit::builder
