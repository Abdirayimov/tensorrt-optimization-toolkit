#include "trt_toolkit/plugin/plugin_registry.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <atomic>

#include "trt_toolkit/plugin/gelu_plugin.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::plugin {

namespace {

GeluPluginCreator& gelu_creator() {
    static GeluPluginCreator inst;
    return inst;
}

std::atomic<bool> g_registered{false};

}  // namespace

void register_builtin_plugins() {
    bool expected = false;
    if (!g_registered.compare_exchange_strong(expected, true)) return;

    // Standard NVIDIA plugin library (registers things like NMS, ROIAlign).
    initLibNvInferPlugins(&utils::tensorrt_logger(), "");

    auto* registry = nvinfer1::getPluginRegistry();
    if (registry == nullptr) {
        TRT_LOG_WARN("getPluginRegistry returned null; toolkit plugins will not be visible");
        return;
    }
    registry->registerCreator(gelu_creator(), "");
    TRT_LOG_INFO("registered toolkit plugins: GeluPlugin v1");
}

}  // namespace trt_toolkit::plugin
