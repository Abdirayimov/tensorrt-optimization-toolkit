#pragma once

#include <NvInferRuntime.h>
#include <spdlog/spdlog.h>

#include <string>

namespace trt_toolkit::utils {

/// Initialise the global spdlog logger. Idempotent.
void init_logger(const std::string& level = "info", bool json = false);

/// nvinfer1::ILogger implementation that forwards every TensorRT log
/// message to spdlog at the matching severity. Use the singleton
/// `tensorrt_logger()` rather than constructing your own.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

/// Process-wide TrtLogger. Pass to `nvinfer1::createInferRuntime` and
/// `nvinfer1::createInferBuilder`.
TrtLogger& tensorrt_logger();

}  // namespace trt_toolkit::utils

#define TRT_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define TRT_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define TRT_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define TRT_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define TRT_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define TRT_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
