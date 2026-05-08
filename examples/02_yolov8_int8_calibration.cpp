// 02_yolov8_int8_calibration.cpp - INT8 calibration walk-through.
//
// Build a YOLOv8s detector at INT8 with on-disk calibration cache and
// fall back to FP16 layers where INT8 is unsafe (default behaviour).
//
// Prerequisites:
//   1. yolov8s.onnx in the working directory. Convert from .pt with:
//        yolo export model=yolov8s.pt format=onnx imgsz=640
//   2. A directory of preprocessed calibration samples (one .bin per
//      sample, 3 * 640 * 640 floats packed C-contiguous, normalised
//      to whatever YOLOv8 expects). 200-1000 samples is typical.

#include <iostream>

#include "trt_toolkit/builder/binary_folder_calibrator.hpp"
#include "trt_toolkit/builder/dynamic_shapes.hpp"
#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/builder/int8_calibrator.hpp"
#include "trt_toolkit/utils/logger.hpp"

int main(int argc, char** argv) {
    using namespace trt_toolkit;

    utils::init_logger("info", false);

    if (argc < 2) {
        std::cerr << "Usage: 02_yolov8_int8_calibration CALIB_DIR\n";
        return 1;
    }
    const std::string calib_dir = argv[1];

    auto provider = builder::make_binary_folder_provider(
        calib_dir, "images", {3, 640, 640}, 4, 8);

    builder::Int8EntropyCalibrator calibrator(std::move(provider),
                                               "calib_cache/yolov8s_int8.cache");

    builder::ShapeProfile profile;
    profile.add(builder::batched_shape("images", {1, 3, 640, 640}, 1, 8, 16));

    builder::BuildOptions opts;
    opts.precision = builder::Precision::INT8;
    opts.workspace_mib = 4096;
    opts.calibrator = &calibrator;
    opts.profile = std::move(profile);

    builder::TrtBuilder bldr;
    bldr.build("yolov8s.onnx", "yolov8s_int8.engine", opts);

    std::cout << "wrote yolov8s_int8.engine; calibration cache reused on rebuild\n";
    return 0;
}
