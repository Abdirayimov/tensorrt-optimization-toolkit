#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "trt_toolkit/builder/int8_calibrator.hpp"

namespace trt_toolkit::builder {

/// Build a CalibrationBatchProvider that reads pre-processed binary
/// tensors from disk.
///
/// Each `.bin` file in `data_dir` is expected to hold a single sample
/// of `volume_per_sample` floats (or whatever element type is implied
/// by `provider.element_size_bytes`), packed in C-contiguous order.
///
/// The function does not preprocess images on its own; the user is
/// responsible for producing the .bin files (typically via a small
/// Python script that loads images, resizes, normalises, and dumps
/// raw bytes).
///
/// @param data_dir              Directory of .bin sample files.
/// @param input_name            TRT input binding name.
/// @param input_shape           Shape per sample (no batch dimension).
/// @param element_size_bytes    Bytes per element (4 for FP32).
/// @param batch_size            Calibration batch size.
/// @return populated CalibrationBatchProvider; the lifetime of the
///         returned object's `fill` lambda owns its own state and is
///         safe to pass into Int8EntropyCalibrator.
CalibrationBatchProvider make_binary_folder_provider(
    const std::filesystem::path& data_dir, std::string input_name,
    std::vector<std::int64_t> input_shape, std::int32_t element_size_bytes = 4,
    std::size_t batch_size = 8);

}  // namespace trt_toolkit::builder
