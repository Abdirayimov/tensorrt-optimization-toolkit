// 01_resnet50_fp16.cpp - the simplest end-to-end recipe.
//
// Build:
//   ResNet50 ONNX -> FP16 .engine -> latency / throughput on random input.
//
// Prerequisite: a `resnet50.onnx` file. Easy way:
//   pip install torch torchvision
//   python -c "import torch, torchvision; m=torchvision.models.resnet50(weights=None); \
//              x=torch.randn(1,3,224,224); torch.onnx.export(m, x, 'resnet50.onnx', \
//              input_names=['data'], output_names=['logits'], opset_version=17, \
//              dynamic_axes={'data':{0:'batch'},'logits':{0:'batch'}})"

#include <iostream>

#include "trt_toolkit/benchmark/latency_probe.hpp"
#include "trt_toolkit/benchmark/throughput_probe.hpp"
#include "trt_toolkit/builder/dynamic_shapes.hpp"
#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/runner/trt_runner.hpp"
#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

int main() {
    using namespace trt_toolkit;

    utils::init_logger("info", false);

    builder::ShapeProfile profile;
    profile.add(builder::batched_shape("data", {1, 3, 224, 224}, 1, 8, 16));

    builder::BuildOptions opts;
    opts.precision = builder::Precision::FP16;
    opts.workspace_mib = 4096;
    opts.profile = std::move(profile);

    builder::TrtBuilder bldr;
    bldr.build("resnet50.onnx", "resnet50_fp16.engine", opts);

    runner::TrtRunner trt_runner("resnet50_fp16.engine");
    trt_runner.set_input_shape("data", {8, 3, 224, 224});

    utils::CudaStream stream;
    std::vector<float> input(8 * 3 * 224 * 224, 0.5f);
    trt_runner.copy_input("data", input.data(), input.size() * sizeof(float), stream.get());
    stream.synchronize();

    auto lat = benchmark::measure_latency([&] { trt_runner.infer(stream.get()); });
    std::cout << "\nResNet50 FP16 (batch=8) latency:\n  ";
    lat.format(std::cout);
    std::cout << "\n";
    return 0;
}
