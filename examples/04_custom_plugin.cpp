// 04_custom_plugin.cpp - register a custom plugin at process start so
// that any ONNX file referencing the plugin op compiles successfully.
//
// In production you would either:
//   a) Have your network export the plugin op directly (the ONNX
//      graph carries a "plugin" node with the matching name and
//      version), OR
//   b) Use a custom ONNX symbolic to emit the node from PyTorch.
//
// Either way, the C++ side just needs to call register_builtin_plugins
// before invoking the parser.

#include <iostream>

#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/plugin/plugin_registry.hpp"
#include "trt_toolkit/utils/logger.hpp"

int main(int argc, char** argv) {
    using namespace trt_toolkit;

    utils::init_logger("info", false);
    plugin::register_builtin_plugins();

    if (argc < 2) {
        std::cerr << "Usage: 04_custom_plugin path/to/model_with_gelu.onnx\n";
        std::cerr << "\nNote: the GeluPlugin in this toolkit is a reference; production\n"
                  << "code typically uses the built-in TRT GELU op. See the comments at\n"
                  << "the top of include/trt_toolkit/plugin/gelu_plugin.hpp for details.\n";
        return 1;
    }

    builder::BuildOptions opts;
    opts.precision = builder::Precision::FP16;
    builder::TrtBuilder().build(argv[1], "with_plugin.engine", opts);
    std::cout << "engine compiled with custom plugin registration\n";
    return 0;
}
