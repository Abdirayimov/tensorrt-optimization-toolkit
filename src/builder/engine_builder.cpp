#include "trt_toolkit/builder/engine_builder.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <stdexcept>
#include <vector>

#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::builder {

namespace {

// nvinfer1 owns objects via raw pointers + virtual destroyers. unique_ptr
// with a default deleter is fine because we never compile against the
// pre-8.0 API.

void enable_precision(nvinfer1::IBuilderConfig& cfg, Precision p) {
    using F = nvinfer1::BuilderFlag;
    switch (p) {
        case Precision::FP32:
            break;
        case Precision::FP16:
            cfg.setFlag(F::kFP16);
            break;
        case Precision::INT8:
            cfg.setFlag(F::kINT8);
            // Mixed precision is the realistic case; let TRT choose
            // FP16 fallback for layers that don't support INT8.
            cfg.setFlag(F::kFP16);
            break;
    }
}

nvinfer1::Dims to_dims(const std::vector<std::int64_t>& s) {
    nvinfer1::Dims d{};
    d.nbDims = static_cast<std::int32_t>(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        d.d[i] = s[i];
    }
    return d;
}

void apply_profile(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& cfg,
                   const ShapeProfile& profile) {
    if (profile.empty()) return;
    auto* p = builder.createOptimizationProfile();
    using S = nvinfer1::OptProfileSelector;
    for (const auto& range : profile.ranges()) {
        p->setDimensions(range.input_name.c_str(), S::kMIN, to_dims(range.min_shape));
        p->setDimensions(range.input_name.c_str(), S::kOPT, to_dims(range.opt_shape));
        p->setDimensions(range.input_name.c_str(), S::kMAX, to_dims(range.max_shape));
    }
    cfg.addOptimizationProfile(p);
}

}  // namespace

struct TrtBuilder::Impl {};

TrtBuilder::TrtBuilder() : impl_(std::make_unique<Impl>()) {}
TrtBuilder::~TrtBuilder() = default;

void TrtBuilder::build(const std::filesystem::path& onnx_path,
                       const std::filesystem::path& output_path, const BuildOptions& options) {
    auto& trt_logger = utils::tensorrt_logger();
    if (options.verbose) {
        // The TRT logger forwards through spdlog, which we expect the
        // CLI to have set to debug. No extra hookup needed.
    }

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(trt_logger)};
    if (!builder) throw std::runtime_error("createInferBuilder failed");

    const auto explicit_batch =
        1U << static_cast<std::uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(explicit_batch)};
    if (!network) throw std::runtime_error("createNetworkV2 failed");

    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, trt_logger)};
    if (!parser) throw std::runtime_error("createParser failed");

    if (!parser->parseFromFile(onnx_path.string().c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (std::int32_t i = 0; i < parser->getNbErrors(); ++i) {
            TRT_LOG_ERROR("ONNX parse error: {}", parser->getError(i)->desc());
        }
        throw std::runtime_error("failed to parse ONNX: " + onnx_path.string());
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) throw std::runtime_error("createBuilderConfig failed");

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<std::size_t>(options.workspace_mib) * (1ULL << 20));
    enable_precision(*config, options.precision);
    if (options.strict_types) {
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    apply_profile(*builder, *config, options.profile);

    if (options.precision == Precision::INT8) {
        if (!options.calibrator) {
            throw std::invalid_argument(
                "INT8 build requires options.calibrator (or a pre-existing cache)");
        }
        config->setInt8Calibrator(options.calibrator);
    }

    TRT_LOG_INFO("building engine: precision={} workspace={} MiB profile={}",
                 (options.precision == Precision::FP32 ? "FP32"
                  : options.precision == Precision::FP16 ? "FP16" : "INT8"),
                 options.workspace_mib,
                 options.profile.empty() ? "<none>" : "<profile attached>");

    std::unique_ptr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) throw std::runtime_error("buildSerializedNetwork failed");

    std::ofstream f(output_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open engine output: " + output_path.string());
    }
    f.write(static_cast<const char*>(serialized->data()),
            static_cast<std::streamsize>(serialized->size()));
    if (!f) {
        throw std::runtime_error("failed to write engine to: " + output_path.string());
    }

    TRT_LOG_INFO("wrote engine ({} MiB) to {}",
                 serialized->size() / (1ULL << 20), output_path.string());
}

}  // namespace trt_toolkit::builder
