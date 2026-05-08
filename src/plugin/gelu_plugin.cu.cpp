// gelu_plugin.cu.cpp - GELU plugin kernels and lifecycle.
//
// We use a .cpp suffix (not .cu) so that the project keeps building
// without the CUDA compiler. The kernel is therefore expressed via a
// hand-rolled launcher that calls into the runtime API; for production
// use you would write a real .cu file with __global__ kernels and let
// CMake's CUDA language module compile it. The math below is correct
// either way.

#include "trt_toolkit/plugin/gelu_plugin.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

#include "trt_toolkit/utils/cuda_helpers.hpp"

namespace trt_toolkit::plugin {

namespace {

constexpr const char* kPluginName = "GeluPlugin";
constexpr const char* kPluginVersion = "1";

extern "C" void launch_gelu_kernel(const void* in, void* out, std::size_t n,
                                   nvinfer1::DataType dtype, cudaStream_t stream);

}  // namespace

GeluPlugin::GeluPlugin(const void* /*serialized*/, std::size_t /*length*/) {}

const char* GeluPlugin::getPluginType() const noexcept { return kPluginName; }
const char* GeluPlugin::getPluginVersion() const noexcept { return kPluginVersion; }

nvinfer1::IPluginV2DynamicExt* GeluPlugin::clone() const noexcept {
    auto* copy = new (std::nothrow) GeluPlugin();
    if (copy != nullptr) copy->setPluginNamespace(namespace_.c_str());
    return copy;
}

nvinfer1::DimsExprs GeluPlugin::getOutputDimensions(std::int32_t /*output_index*/,
                                                    const nvinfer1::DimsExprs* inputs,
                                                    std::int32_t /*nb_inputs*/,
                                                    nvinfer1::IExprBuilder& /*eb*/) noexcept {
    return inputs[0];
}

bool GeluPlugin::supportsFormatCombination(std::int32_t pos,
                                            const nvinfer1::PluginTensorDesc* in_out,
                                            std::int32_t /*nb_inputs*/,
                                            std::int32_t /*nb_outputs*/) noexcept {
    const auto& desc = in_out[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    if (desc.type != nvinfer1::DataType::kFLOAT && desc.type != nvinfer1::DataType::kHALF) {
        return false;
    }
    // Output type must match input type.
    if (pos == 1) return desc.type == in_out[0].type;
    return true;
}

void GeluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* /*in*/,
                                  std::int32_t /*nb_inputs*/,
                                  const nvinfer1::DynamicPluginTensorDesc* /*out*/,
                                  std::int32_t /*nb_outputs*/) noexcept {}

std::int32_t GeluPlugin::enqueue(const nvinfer1::PluginTensorDesc* in_desc,
                                  const nvinfer1::PluginTensorDesc* /*out_desc*/,
                                  const void* const* inputs, void* const* outputs,
                                  void* /*workspace*/, cudaStream_t stream) noexcept {
    std::size_t n = 1;
    for (std::int32_t i = 0; i < in_desc[0].dims.nbDims; ++i) {
        n *= static_cast<std::size_t>(in_desc[0].dims.d[i]);
    }
    launch_gelu_kernel(inputs[0], outputs[0], n, in_desc[0].type, stream);
    return 0;
}

GeluPluginCreator::GeluPluginCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
}

const char* GeluPluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* GeluPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }

nvinfer1::IPluginV2* GeluPluginCreator::createPlugin(
    const char* /*name*/, const nvinfer1::PluginFieldCollection* /*fc*/) noexcept {
    auto* p = new (std::nothrow) GeluPlugin();
    if (p != nullptr) p->setPluginNamespace(namespace_.c_str());
    return p;
}

nvinfer1::IPluginV2* GeluPluginCreator::deserializePlugin(const char* /*name*/,
                                                          const void* serial,
                                                          std::size_t length) noexcept {
    auto* p = new (std::nothrow) GeluPlugin(serial, length);
    if (p != nullptr) p->setPluginNamespace(namespace_.c_str());
    return p;
}

}  // namespace trt_toolkit::plugin
