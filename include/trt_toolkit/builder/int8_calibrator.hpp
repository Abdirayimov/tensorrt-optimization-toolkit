#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "trt_toolkit/utils/cuda_helpers.hpp"

namespace trt_toolkit::builder {

/// Source of calibration batches.
///
/// The provider is invoked once per batch the calibrator needs. It must
/// fill `dst_host` with `batch_count * sample_size_bytes` of preprocessed
/// data using the same preprocessing the deployed network expects, and
/// return the actual batch size produced (which may be smaller than the
/// requested batch when the data set is exhausted).
///
/// Returning 0 signals end-of-data; the calibrator will then finalize.
struct CalibrationBatchProvider {
    using FillFn = std::function<std::size_t(void* dst_host, std::size_t requested_batch)>;

    std::string input_name;
    std::vector<std::int64_t> input_shape;  ///< Shape of one sample (no batch).
    std::int32_t element_size_bytes = 4;    ///< 4 for FP32, 2 for FP16, ...
    std::size_t target_batch = 8;
    FillFn fill;
};

/// IInt8EntropyCalibrator2 implementation with on-disk cache.
///
/// `calibrate()` invokes the provider repeatedly to feed the builder.
/// `readCalibrationCache()` and `writeCalibrationCache()` go to the
/// path supplied at construction; this lets the same engine be rebuilt
/// across different machines (or after a CUDA driver upgrade) without
/// re-running calibration.
///
/// Use it via `TrtBuilder::int8_with_calibrator(...)` rather than
/// directly; the builder owns the device buffer's lifetime.
class Int8EntropyCalibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(CalibrationBatchProvider provider,
                          std::filesystem::path cache_path);
    ~Int8EntropyCalibrator() override;

    Int8EntropyCalibrator(const Int8EntropyCalibrator&) = delete;
    Int8EntropyCalibrator& operator=(const Int8EntropyCalibrator&) = delete;

    // ----- nvinfer1::IInt8EntropyCalibrator2 interface -----
    std::int32_t getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], std::int32_t nb_bindings) noexcept
        override;
    const void* readCalibrationCache(std::size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, std::size_t length) noexcept override;

    /// Cache-only constructor: reuses an existing cache without ever
    /// calling the provider. Useful when you trust an artefact built
    /// elsewhere.
    static std::unique_ptr<Int8EntropyCalibrator> from_cache(
        std::filesystem::path cache_path);

private:
    CalibrationBatchProvider provider_;
    std::filesystem::path cache_path_;
    std::vector<std::uint8_t> cached_;
    bool cache_loaded_ = false;
    bool exhausted_ = false;

    std::size_t sample_volume_ = 0;
    std::size_t sample_bytes_ = 0;
    std::size_t batch_bytes_ = 0;

    std::vector<std::uint8_t> host_scratch_;
    utils::DeviceBuffer device_buffer_;
};

}  // namespace trt_toolkit::builder
