#include "trt_toolkit/debug/onnx_inspector.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>

#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::debug {

namespace {

std::string dtype_to_string(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF:  return "float16";
        case nvinfer1::DataType::kINT8:  return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL:  return "bool";
        case nvinfer1::DataType::kUINT8: return "uint8";
        case nvinfer1::DataType::kFP8:   return "fp8";
        default: return "?";
    }
}

void collect_io(nvinfer1::ITensor* t, OnnxIO& out) {
    out.name = t->getName();
    out.dtype = dtype_to_string(t->getType());
    const auto dims = t->getDimensions();
    out.shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (std::int32_t i = 0; i < dims.nbDims; ++i) {
        out.shape.push_back(dims.d[i]);
    }
}

}  // namespace

void OnnxSummary::format(std::ostream& os) const {
    os << "onnx: " << source_path << "\n"
       << "  ir_version:     " << ir_version << "\n"
       << "  opset_version:  " << opset_version << "\n"
       << "  producer:       " << producer << "\n"
       << "  inputs (" << inputs.size() << "):\n";
    for (const auto& io : inputs) {
        os << "    " << io.name << "  " << io.dtype << "  shape=";
        for (std::size_t i = 0; i < io.shape.size(); ++i) {
            if (i > 0) os << "x";
            os << (io.shape[i] < 0 ? std::string("dyn") : std::to_string(io.shape[i]));
        }
        os << "\n";
    }
    os << "  outputs (" << outputs.size() << "):\n";
    for (const auto& io : outputs) {
        os << "    " << io.name << "  " << io.dtype << "  shape=";
        for (std::size_t i = 0; i < io.shape.size(); ++i) {
            if (i > 0) os << "x";
            os << (io.shape[i] < 0 ? std::string("dyn") : std::to_string(io.shape[i]));
        }
        os << "\n";
    }
    os << "  ops (top by count):\n";
    for (const auto& op : ops) {
        os << "    " << op.op_type << ":  " << op.count << "\n";
    }
}

OnnxSummary inspect_onnx(const std::filesystem::path& onnx_path) {
    auto& trt_logger = utils::tensorrt_logger();

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
                               static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
        for (std::int32_t i = 0; i < parser->getNbErrors(); ++i) {
            TRT_LOG_ERROR("ONNX parse error: {}", parser->getError(i)->desc());
        }
        throw std::runtime_error("failed to parse ONNX: " + onnx_path.string());
    }

    OnnxSummary out;
    out.source_path = onnx_path.string();

    const std::int32_t ni = network->getNbInputs();
    out.inputs.resize(static_cast<std::size_t>(ni));
    for (std::int32_t i = 0; i < ni; ++i) {
        collect_io(network->getInput(i), out.inputs[static_cast<std::size_t>(i)]);
    }
    const std::int32_t no = network->getNbOutputs();
    out.outputs.resize(static_cast<std::size_t>(no));
    for (std::int32_t i = 0; i < no; ++i) {
        collect_io(network->getOutput(i), out.outputs[static_cast<std::size_t>(i)]);
    }

    std::map<std::string, std::int32_t> op_counts;
    const std::int32_t nl = network->getNbLayers();
    for (std::int32_t i = 0; i < nl; ++i) {
        const auto* layer = network->getLayer(i);
        op_counts[std::string("Layer/") + std::to_string(static_cast<int>(layer->getType()))] += 1;
    }
    out.ops.reserve(op_counts.size());
    for (const auto& [k, v] : op_counts) {
        out.ops.push_back({k, v});
    }
    std::sort(out.ops.begin(), out.ops.end(),
              [](const OnnxOpSummary& a, const OnnxOpSummary& b) { return a.count > b.count; });

    return out;
}

}  // namespace trt_toolkit::debug
