<h1 align="center">tensorrt-optimization-toolkit</h1>

<p align="center">
  <i>C++ toolkit for the unglamorous parts of TensorRT: ONNX -> engine compilation, INT8 calibration, custom plugin development, accuracy/latency benchmarking.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-8.6%2B-76B900.svg" alt="TensorRT">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/badge/status-reference%20implementation-orange.svg" alt="Status">
</p>

---

## Why this exists

Most "TensorRT tutorial" repos stop after a triumphant FP32 -> FP16
conversion of ResNet-50. Real production work needs more:

- **INT8 calibration** with a representative data set, an on-disk
  cache, and the discipline to keep that cache reproducible across
  CUDA driver upgrades.
- **Dynamic shape profiles** that match what the deployed network
  actually sees. Building an engine for `min=1 opt=8 max=64` and then
  sending it `batch=128` produces a runtime error, not a slowdown.
- **Custom plugins** because no real model is 100% covered by the
  built-in op set. Knowing the lifecycle of `IPluginV2DynamicExt`
  (or its successor `IPluginV3`) is what separates "I follow blog
  posts" from "I ship at NVIDIA".
- **Accuracy regression tracking.** INT8 quantisation that drops
  top-1 by 0.5% on validation may drop it by 5% on edge cases the
  calibration set under-represents; you need to measure that, not
  hope.
- **Benchmark hygiene.** CUDA event timing, warm-ups, percentile
  reporting - boring but easy to get wrong.

This toolkit collects the patterns I reach for repeatedly. The code
is original (no GPL'd source pulled in), uses public algorithms and
public model checkpoints, and is structured so each component is
useful in isolation.

## What's inside

- **`builder/`** - `TrtBuilder` (FP32/FP16/INT8), dynamic shape
  profile helpers, IInt8EntropyCalibrator2 with on-disk cache,
  binary-folder calibration provider
- **`runner/`** - RAII wrapper around an `ICudaEngine` +
  `IExecutionContext` with async streams and pinned memory
- **`benchmark/`** - latency probe (CUDA events, p50/p95/p99),
  throughput probe, GPU memory probe (`cudaMemGetInfo`-based)
- **`accuracy/`** - element-wise diff reporter (max_abs, max_rel,
  cosine), two-engine differ for INT8 vs FP32 tactic regression
- **`plugin/`** - reference GELU plugin (`IPluginV2DynamicExt`),
  plugin registry helper that idempotently registers stock NVIDIA
  plugins + your own
- **`debug/`** - polygraphy-style summaries for both `.onnx` and
  `.engine` files, exposed via `trt-toolkit inspect`
- **CLI**: `trt-toolkit build / benchmark / accuracy / inspect`
- **Examples**: ResNet50 FP16, YOLOv8 INT8 with calibration cache,
  dynamic batch demonstration, custom plugin registration

## Quick start

```bash
# Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 1. Compile an ONNX to an FP16 engine.
./build/trt-toolkit build \
    --onnx resnet50.onnx \
    --output resnet50_fp16.engine \
    --precision fp16 \
    --input data --min-shape 1x3x224x224 \
    --opt-shape 8x3x224x224 --max-shape 16x3x224x224

# 2. Look at what the engine actually contains.
./build/trt-toolkit inspect resnet50_fp16.engine

# 3. Measure how fast it is.
./build/trt-toolkit benchmark --engine resnet50_fp16.engine

# 4. Build an INT8 candidate and compare it to the FP16 reference.
./build/trt-toolkit build --onnx resnet50.onnx --output resnet50_int8.engine \
    --precision int8 --int8-cache calib.cache
./build/trt-toolkit accuracy \
    --reference resnet50_fp16.engine \
    --candidate resnet50_int8.engine
```

## Project structure

```
.
├── CMakeLists.txt
├── cmake/
├── include/trt_toolkit/
│   ├── builder/    engine_builder.hpp, int8_calibrator.hpp,
│   │              dynamic_shapes.hpp, binary_folder_calibrator.hpp
│   ├── runner/     trt_runner.hpp
│   ├── benchmark/  latency_probe.hpp, throughput_probe.hpp,
│   │              memory_probe.hpp
│   ├── accuracy/   diff_reporter.hpp, two_engine_diff.hpp
│   ├── plugin/     gelu_plugin.hpp, plugin_registry.hpp
│   ├── debug/      engine_inspector.hpp, onnx_inspector.hpp
│   └── utils/      logger.hpp, cuda_helpers.hpp
├── src/                   (mirrors include/)
├── tools/                 trt_toolkit_cli.cpp
├── examples/              01_resnet50_fp16.cpp
│                          02_yolov8_int8_calibration.cpp
│                          03_dynamic_batch.cpp
│                          04_custom_plugin.cpp
└── docs/
    ├── int8_calibration.md
    ├── debugging_accuracy_drops.md
    ├── plugin_development.md
    └── building.md
```

## Documentation

- [docs/building.md](docs/building.md) - native + container builds, CMake options
- [docs/int8_calibration.md](docs/int8_calibration.md) - calibration data hygiene, cache reuse
- [docs/debugging_accuracy_drops.md](docs/debugging_accuracy_drops.md) - bisecting layer precision
- [docs/plugin_development.md](docs/plugin_development.md) - lifecycle of a custom plugin

## Limitations

- The bundled GELU plugin uses `IPluginV2DynamicExt`. For TRT 11+
  migrate to `IPluginV3OneBuild` / `IPluginV3OneRuntime`.
- The accuracy comparator is single-pass: it generates one random
  input and compares the outputs. For real validation use a held-out
  data set and feed it via the `EngineDiffOptions::input_blobs`
  vector directly from C++.
- INT8 calibration uses `IInt8EntropyCalibrator2`. Newer
  per-tensor / explicit-quant flows (`IBuilderConfig::setQuant`...
  family) are out of scope.
- The toolkit assumes a single GPU; multi-GPU sharded engines need
  user-side orchestration on top.

## Roadmap

- [ ] Migrate plugin to `IPluginV3OneBuild` once the API is stable
- [ ] CLI `inspect` --json output mode for tool integration
- [ ] Layer-by-layer precision bisection (the "which layer is
      breaking accuracy" workflow)
- [ ] Optional ONNX Runtime reference path for accuracy diffing
- [ ] CUDA-language `gelu_kernel.cu` to replace the host fallback

## License

MIT - see [LICENSE](LICENSE).

## About

This repository is a reference implementation of patterns from
production TensorRT optimisation work. Algorithms are the published
originals (entropy calibration, GELU activation, ONNX parsing); the
code is written from scratch and uses only public checkpoints and
synthetic data.

Open to contract work on similar systems -
[email](mailto:khusanabdirayimov@gmail.com) -
[GitHub](https://github.com/Abdirayimov)
