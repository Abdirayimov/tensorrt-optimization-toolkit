# tensorrt-optimization-toolkit

C++ toolkit for ONNX -> TensorRT engine compilation, INT8 calibration,
and benchmarking. Reference implementation; work in progress.

## Status

- [x] Engine builder (FP32 / FP16 / INT8 with calibration)
- [x] Int8 entropy calibrator with on-disk cache
- [x] Dynamic shape profiles
- [x] TRT runner (RAII, async streams, pinned memory)
- [x] Minimal CLI: `trt-toolkit build`
- [ ] Custom plugin development example
- [ ] Latency / throughput / memory benchmark CLIs
- [ ] Accuracy diff vs ONNX Runtime
- [ ] End-to-end examples (ResNet50, YOLOv8, ArcFace)
- [ ] Polygraphy debugging helpers

See [building](docs/building.md) once docs are in place.
