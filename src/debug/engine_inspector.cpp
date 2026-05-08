#include "trt_toolkit/debug/engine_inspector.hpp"

#include <NvInfer.h>

#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::debug {

namespace {

std::string dtype_name(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kHALF:  return "FP16";
        case nvinfer1::DataType::kINT8:  return "INT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kBOOL:  return "BOOL";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kFP8:   return "FP8";
        default: return "?";
    }
}

}  // namespace

void EngineSummary::format(std::ostream& os) const {
    os << "engine: " << source_path << "\n"
       << "  layers:                     " << num_layers << "\n"
       << "  device memory:              " << (device_memory_bytes / (1024 * 1024))
       << " MiB\n"
       << "  optimization profiles:      " << num_optimization_profiles << "\n"
       << "  bindings (" << bindings.size() << "):\n";
    for (const auto& b : bindings) {
        os << "    " << (b.is_input ? "[in ] " : "[out] ") << b.name
           << "  " << b.dtype_name << "  shape=";
        for (std::size_t i = 0; i < b.shape.size(); ++i) {
            if (i > 0) os << "x";
            os << b.shape[i];
        }
        os << "  vol=" << b.volume << "  bytes=" << b.bytes << "\n";
    }
}

EngineSummary inspect(const std::filesystem::path& engine_path) {
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open engine: " + engine_path.string());
    }
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<std::size_t>(sz));
    f.read(blob.data(), sz);

    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(utils::tensorrt_logger())};
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(blob.data(), blob.size())};
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed");

    EngineSummary out;
    out.source_path = engine_path.string();
    out.num_layers = engine->getNbLayers();
    out.device_memory_bytes = engine->getDeviceMemorySizeV2();
    out.num_optimization_profiles = engine->getNbOptimizationProfiles();

    const std::int32_t n = engine->getNbIOTensors();
    out.bindings.reserve(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(name);
        const auto dtype = engine->getTensorDataType(name);
        const auto io = engine->getTensorIOMode(name);

        BindingDescriptor d;
        d.name = name;
        d.is_input = (io == nvinfer1::TensorIOMode::kINPUT);
        d.shape.reserve(static_cast<std::size_t>(dims.nbDims));
        std::size_t volume = 1;
        for (std::int32_t k = 0; k < dims.nbDims; ++k) {
            d.shape.push_back(dims.d[k]);
            if (dims.d[k] > 0) volume *= static_cast<std::size_t>(dims.d[k]);
        }
        d.dtype_id = static_cast<int>(dtype);
        d.dtype_name = dtype_name(dtype);
        d.element_size = utils::element_size_for(d.dtype_id);
        d.volume = volume;
        d.bytes = volume * d.element_size;
        out.bindings.push_back(std::move(d));
    }
    return out;
}

}  // namespace trt_toolkit::debug
