#include "trt_toolkit/accuracy/two_engine_diff.hpp"

#include <stdexcept>
#include <vector>

#include "trt_toolkit/runner/trt_runner.hpp"
#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::accuracy {

namespace {

void run_engine(runner::TrtRunner& runner,
                const std::vector<std::vector<float>>& input_blobs,
                std::vector<std::vector<float>>& output_blobs,
                std::vector<std::string>& output_names) {
    utils::CudaStream stream;

    std::size_t input_idx = 0;
    for (const auto& b : runner.bindings()) {
        if (!b.is_input) continue;
        if (input_idx >= input_blobs.size()) {
            throw std::runtime_error("not enough input blobs for engine inputs");
        }
        const auto& blob = input_blobs[input_idx];
        const std::size_t expected_floats =
            (b.element_size > 0) ? (b.volume * b.element_size) / sizeof(float) : 0;
        if (blob.size() != expected_floats && expected_floats > 0) {
            throw std::runtime_error("input blob size mismatch for binding " + b.name);
        }
        runner.copy_input(b.name, blob.data(), blob.size() * sizeof(float), stream.get());
        ++input_idx;
    }

    runner.infer(stream.get());

    output_blobs.clear();
    output_names.clear();
    for (const auto& b : runner.bindings()) {
        if (b.is_input) continue;
        const std::size_t floats =
            (b.element_size > 0) ? (b.volume * b.element_size) / sizeof(float) : 0;
        std::vector<float> out(floats, 0.0f);
        if (floats > 0) {
            runner.copy_output(b.name, out.data(), out.size() * sizeof(float), stream.get());
        }
        output_blobs.push_back(std::move(out));
        output_names.push_back(b.name);
    }
    stream.synchronize();
}

}  // namespace

EngineDiffReport diff_engines(const std::filesystem::path& reference_engine,
                              const std::filesystem::path& candidate_engine,
                              const EngineDiffOptions& options) {
    runner::TrtRunner reference(reference_engine);
    runner::TrtRunner candidate(candidate_engine);

    std::vector<std::vector<float>> ref_outputs;
    std::vector<std::vector<float>> cand_outputs;
    std::vector<std::string> ref_names;
    std::vector<std::string> cand_names;

    run_engine(reference, options.input_blobs, ref_outputs, ref_names);
    run_engine(candidate, options.input_blobs, cand_outputs, cand_names);

    if (ref_names.size() != cand_names.size()) {
        throw std::runtime_error("engines have different output binding counts");
    }

    EngineDiffReport report;
    report.binding_names = ref_names;
    report.per_output.reserve(ref_outputs.size());
    for (std::size_t i = 0; i < ref_outputs.size(); ++i) {
        if (ref_outputs[i].size() != cand_outputs[i].size()) {
            throw std::runtime_error("output binding size mismatch for " + ref_names[i]);
        }
        report.per_output.push_back(compare_arrays(ref_outputs[i], cand_outputs[i]));
    }
    return report;
}

}  // namespace trt_toolkit::accuracy
