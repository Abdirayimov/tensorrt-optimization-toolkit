#pragma once

namespace trt_toolkit::plugin {

/// Register every plugin shipped with this library against the global
/// nvinfer1 plugin registry. Idempotent. Call once at process start
/// before parsing any ONNX file that references a custom op.
void register_builtin_plugins();

}  // namespace trt_toolkit::plugin
