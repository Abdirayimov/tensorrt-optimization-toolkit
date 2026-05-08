#include "trt_toolkit/builder/int8_calibrator.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::builder {

namespace {

std::size_t volume_of(const std::vector<std::int64_t>& s) {
    std::size_t v = 1;
    for (auto d : s) {
        if (d <= 0) {
            throw std::invalid_argument("calibration shape must be fully concrete (no -1)");
        }
        v *= static_cast<std::size_t>(d);
    }
    return v;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> out;
    if (!std::filesystem::exists(path)) return out;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return out;
    const auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    out.resize(sz);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
    return out;
}

}  // namespace

Int8EntropyCalibrator::Int8EntropyCalibrator(CalibrationBatchProvider provider,
                                             std::filesystem::path cache_path)
    : provider_(std::move(provider)), cache_path_(std::move(cache_path)) {
    if (!provider_.fill && !std::filesystem::exists(cache_path_)) {
        throw std::invalid_argument(
            "calibrator needs either a fill function or an existing cache file");
    }

    sample_volume_ = volume_of(provider_.input_shape);
    sample_bytes_ = sample_volume_ * static_cast<std::size_t>(provider_.element_size_bytes);
    batch_bytes_ = sample_bytes_ * provider_.target_batch;

    host_scratch_.resize(batch_bytes_);
    device_buffer_ = utils::DeviceBuffer(batch_bytes_);

    if (auto blob = read_file(cache_path_); !blob.empty()) {
        cached_ = std::move(blob);
        cache_loaded_ = true;
        TRT_LOG_INFO("Int8 calibration cache found ({} bytes): {}", cached_.size(),
                     cache_path_.string());
    }
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() = default;

std::unique_ptr<Int8EntropyCalibrator> Int8EntropyCalibrator::from_cache(
    std::filesystem::path cache_path) {
    CalibrationBatchProvider stub;
    stub.input_name = "<cache-only>";
    stub.input_shape = {1};
    stub.element_size_bytes = 1;
    stub.target_batch = 1;
    return std::make_unique<Int8EntropyCalibrator>(std::move(stub), std::move(cache_path));
}

std::int32_t Int8EntropyCalibrator::getBatchSize() const noexcept {
    return static_cast<std::int32_t>(provider_.target_batch);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[],
                                     std::int32_t nb_bindings) noexcept {
    if (exhausted_) return false;
    if (cache_loaded_) {
        // Cache present: TRT does not need fresh batches.
        return false;
    }
    if (!provider_.fill) {
        return false;
    }

    try {
        const std::size_t actual = provider_.fill(host_scratch_.data(), provider_.target_batch);
        if (actual == 0) {
            exhausted_ = true;
            return false;
        }

        if (cudaMemcpy(device_buffer_.get(), host_scratch_.data(), actual * sample_bytes_,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }

        for (std::int32_t i = 0; i < nb_bindings; ++i) {
            if (provider_.input_name == names[i]) {
                bindings[i] = device_buffer_.get();
                return true;
            }
        }
        return false;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("calibrator fill threw: {}", e.what());
        return false;
    }
}

const void* Int8EntropyCalibrator::readCalibrationCache(std::size_t& length) noexcept {
    if (cached_.empty()) {
        length = 0;
        return nullptr;
    }
    length = cached_.size();
    return cached_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, std::size_t length) noexcept {
    try {
        std::filesystem::create_directories(cache_path_.parent_path());
        std::ofstream f(cache_path_, std::ios::binary);
        if (!f.is_open()) {
            SPDLOG_WARN("could not open calibration cache for write: {}", cache_path_.string());
            return;
        }
        f.write(static_cast<const char*>(cache), static_cast<std::streamsize>(length));
        TRT_LOG_INFO("wrote Int8 calibration cache ({} bytes): {}", length, cache_path_.string());
    } catch (const std::exception& e) {
        SPDLOG_WARN("calibration cache write failed: {}", e.what());
    }
}

}  // namespace trt_toolkit::builder
