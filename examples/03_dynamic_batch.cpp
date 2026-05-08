// 03_dynamic_batch.cpp - illustrate switching batch sizes at runtime.
//
// Build the same engine with a wide dynamic batch profile, then
// repeatedly call set_input_shape to flip between batch=1 and
// batch=16 - measuring latency at each.

#include <iostream>

#include "trt_toolkit/benchmark/latency_probe.hpp"
#include "trt_toolkit/builder/dynamic_shapes.hpp"
#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/runner/trt_runner.hpp"
#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

int main(int argc, char** argv) {
    using namespace trt_toolkit;

    utils::init_logger("info", false);

    if (argc < 2) {
        std::cerr << "Usage: 03_dynamic_batch path/to/model.onnx\n";
        return 1;
    }
    const std::string onnx = argv[1];
    const std::string engine_path = "dynamic_batch.engine";

    builder::ShapeProfile profile;
    profile.add(builder::batched_shape("data", {1, 3, 224, 224}, 1, 8, 32));

    builder::BuildOptions opts;
    opts.precision = builder::Precision::FP16;
    opts.profile = std::move(profile);
    builder::TrtBuilder().build(onnx, engine_path, opts);

    runner::TrtRunner trt_runner(engine_path);
    utils::CudaStream stream;

    for (std::int64_t bs : {1, 4, 8, 16, 32}) {
        trt_runner.set_input_shape("data", {bs, 3, 224, 224});
        std::vector<float> blob(static_cast<std::size_t>(bs) * 3 * 224 * 224, 0.0f);
        trt_runner.copy_input("data", blob.data(), blob.size() * sizeof(float), stream.get());
        stream.synchronize();

        auto lat = benchmark::measure_latency([&] { trt_runner.infer(stream.get()); });
        std::cout << "batch=" << bs << ":  p50=" << lat.p50_ms
                  << "ms  p99=" << lat.p99_ms
                  << "ms  samples/s=" << (1000.0 / lat.p50_ms) * static_cast<double>(bs) << "\n";
    }
    return 0;
}
