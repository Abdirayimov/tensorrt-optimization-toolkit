#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_toolkit::plugin {

/// Reference GELU activation plugin.
///
/// Implements the GELU approximation `0.5 * x * (1 + tanh(sqrt(2/pi) *
/// (x + 0.044715 * x^3)))` over a single float input tensor.
/// Implementation uses `IPluginV2DynamicExt` rather than the newer
/// `IPluginV3` because V2DynamicExt is currently the most stable
/// cross-version API; migrate to V3 when targeting TRT 11+.
///
/// This is *not* a faster GELU than the built-in TRT op; the value
/// here is in the boilerplate showing how a custom plugin is wired.
class GeluPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    GeluPlugin() = default;
    explicit GeluPlugin(const void* serialized, std::size_t length);

    // ----- IPluginV2 -----
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    std::int32_t getNbOutputs() const noexcept override { return 1; }
    std::int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    std::size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    // ----- IPluginV2Ext -----
    nvinfer1::DataType getOutputDataType(std::int32_t /*index*/,
                                         const nvinfer1::DataType* input_types,
                                         std::int32_t /*nb_inputs*/) const noexcept override {
        return input_types[0];
    }

    // ----- IPluginV2DynamicExt -----
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        std::int32_t output_index, const nvinfer1::DimsExprs* inputs, std::int32_t nb_inputs,
        nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(std::int32_t pos,
                                   const nvinfer1::PluginTensorDesc* in_out,
                                   std::int32_t nb_inputs,
                                   std::int32_t nb_outputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         std::int32_t nb_inputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         std::int32_t nb_outputs) noexcept override;
    std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*in*/,
                                 std::int32_t /*nb_inputs*/,
                                 const nvinfer1::PluginTensorDesc* /*out*/,
                                 std::int32_t /*nb_outputs*/) const noexcept override {
        return 0;
    }
    std::int32_t enqueue(const nvinfer1::PluginTensorDesc* in_desc,
                         const nvinfer1::PluginTensorDesc* out_desc, const void* const* inputs,
                         void* const* outputs, void* workspace,
                         cudaStream_t stream) noexcept override;

private:
    std::string namespace_;
};

class GeluPluginCreator : public nvinfer1::IPluginCreator {
public:
    GeluPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }
    nvinfer1::IPluginV2* createPlugin(const char* name,
                                      const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial,
                                           std::size_t length) noexcept override;
    void setPluginNamespace(const char* ns) noexcept override { namespace_ = ns; }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    nvinfer1::PluginFieldCollection fc_{};
    std::vector<nvinfer1::PluginField> fields_;
    std::string namespace_;
};

}  // namespace trt_toolkit::plugin
