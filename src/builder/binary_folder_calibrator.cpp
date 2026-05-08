#include "trt_toolkit/builder/binary_folder_calibrator.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::builder {

namespace {

struct State {
    std::vector<std::filesystem::path> files;
    std::size_t cursor = 0;
    std::size_t bytes_per_sample = 0;
};

std::vector<std::filesystem::path> list_bin_files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::is_directory(dir)) {
        throw std::invalid_argument("calibration data dir is not a directory: " +
                                    dir.string());
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".bin") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

CalibrationBatchProvider make_binary_folder_provider(const std::filesystem::path& data_dir,
                                                     std::string input_name,
                                                     std::vector<std::int64_t> input_shape,
                                                     std::int32_t element_size_bytes,
                                                     std::size_t batch_size) {
    auto state = std::make_shared<State>();
    state->files = list_bin_files(data_dir);
    if (state->files.empty()) {
        throw std::runtime_error("no .bin files in calibration dir: " + data_dir.string());
    }

    std::size_t volume = 1;
    for (auto d : input_shape) {
        if (d <= 0) {
            throw std::invalid_argument("calibration input_shape must be fully concrete");
        }
        volume *= static_cast<std::size_t>(d);
    }
    state->bytes_per_sample = volume * static_cast<std::size_t>(element_size_bytes);

    CalibrationBatchProvider p;
    p.input_name = std::move(input_name);
    p.input_shape = std::move(input_shape);
    p.element_size_bytes = element_size_bytes;
    p.target_batch = batch_size;
    p.fill = [state](void* dst_host, std::size_t requested_batch) -> std::size_t {
        if (state->cursor >= state->files.size()) return 0;
        std::size_t produced = 0;
        auto* out = static_cast<std::uint8_t*>(dst_host);
        while (produced < requested_batch && state->cursor < state->files.size()) {
            const auto& path = state->files[state->cursor];
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) {
                SPDLOG_WARN("cannot open calibration sample: {}", path.string());
                ++state->cursor;
                continue;
            }
            f.read(reinterpret_cast<char*>(out + produced * state->bytes_per_sample),
                   static_cast<std::streamsize>(state->bytes_per_sample));
            if (f.gcount() != static_cast<std::streamsize>(state->bytes_per_sample)) {
                SPDLOG_WARN("short read from {} (expected {} bytes, got {})",
                            path.string(), state->bytes_per_sample, f.gcount());
                ++state->cursor;
                continue;
            }
            ++state->cursor;
            ++produced;
        }
        return produced;
    };
    return p;
}

}  // namespace trt_toolkit::builder
