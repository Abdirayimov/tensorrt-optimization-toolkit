// trt-toolkit CLI:  build / benchmark / accuracy / inspect subcommands.
//
//   trt-toolkit build      --onnx M.onnx --output M.engine [...]
//   trt-toolkit benchmark  --engine M.engine [--iterations 200]
//   trt-toolkit accuracy   --reference REF.engine --candidate CAND.engine
//   trt-toolkit inspect    PATH       (auto-detects .onnx vs .engine)

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "trt_toolkit/accuracy/diff_reporter.hpp"
#include "trt_toolkit/accuracy/two_engine_diff.hpp"
#include "trt_toolkit/benchmark/latency_probe.hpp"
#include "trt_toolkit/benchmark/memory_probe.hpp"
#include "trt_toolkit/benchmark/throughput_probe.hpp"
#include "trt_toolkit/builder/dynamic_shapes.hpp"
#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/builder/int8_calibrator.hpp"
#include "trt_toolkit/debug/engine_inspector.hpp"
#include "trt_toolkit/debug/onnx_inspector.hpp"
#include "trt_toolkit/runner/trt_runner.hpp"
#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace {

std::vector<std::int64_t> parse_shape(const std::string& s) {
    std::vector<std::int64_t> out;
    std::string token;
    for (const char c : s) {
        if (c == 'x' || c == 'X' || c == ',') {
            if (!token.empty()) {
                out.push_back(std::stoll(token));
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) out.push_back(std::stoll(token));
    return out;
}

trt_toolkit::builder::Precision parse_precision(const std::string& s) {
    using P = trt_toolkit::builder::Precision;
    if (s == "fp32") return P::FP32;
    if (s == "fp16") return P::FP16;
    if (s == "int8") return P::INT8;
    throw std::invalid_argument("unknown --precision: " + s);
}

void print_help() {
    std::cout
        << "trt-toolkit - ONNX -> TensorRT engine compilation, benchmarking, and diff\n"
        << "\n"
        << "Subcommands:\n"
        << "  build       Compile an ONNX file into a serialized .engine\n"
        << "  benchmark   Latency / throughput / memory for a built engine\n"
        << "  accuracy    Compare outputs of a reference and candidate engine\n"
        << "  inspect     Print metadata for a .onnx or .engine file\n"
        << "\n"
        << "Run `trt-toolkit <sub> --help` for subcommand-specific options.\n";
}

int inspect_subcommand(const std::vector<std::string>& args) {
    using namespace trt_toolkit;

    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        std::cout << "Usage: trt-toolkit inspect PATH\n"
                  << "\n"
                  << "PATH is either a .onnx or a .engine file; the format is\n"
                  << "auto-detected from the extension.\n";
        return EXIT_SUCCESS;
    }

    utils::init_logger("info", false);

    const std::filesystem::path path = args[0];
    const auto ext = path.extension().string();
    if (ext == ".onnx") {
        const auto summary = debug::inspect_onnx(path);
        summary.format(std::cout);
    } else if (ext == ".engine" || ext == ".plan" || ext == ".trt") {
        const auto summary = debug::inspect(path);
        summary.format(std::cout);
    } else {
        throw std::invalid_argument(
            "inspect: unrecognised extension '" + ext + "' (use .onnx or .engine)");
    }
    return EXIT_SUCCESS;
}

// ============================================================================
// build
// ============================================================================

void print_build_help() {
    std::cout
        << "Usage: trt-toolkit build --onnx ONNX --output ENGINE [options]\n"
        << "\n"
        << "Required:\n"
        << "  --onnx PATH               Source ONNX file\n"
        << "  --output PATH             Destination .engine\n"
        << "\n"
        << "Optional:\n"
        << "  --precision fp32|fp16|int8     Default: fp16\n"
        << "  --workspace-mib N              Default: 4096\n"
        << "  --strict-types                 Disable precision fallback\n"
        << "  --verbose                      Bump TRT logger to debug\n"
        << "\n"
        << "Dynamic shapes (repeat triplet per dynamic input):\n"
        << "  --input NAME --min-shape DxDxD --opt-shape DxDxD --max-shape DxDxD\n"
        << "\n"
        << "INT8:\n"
        << "  --int8-cache PATH              Use existing cache; do not run calibration\n";
}

int build_subcommand(const std::vector<std::string>& args) {
    using namespace trt_toolkit::builder;

    if (args.empty() || (args.size() == 1 && (args[0] == "--help" || args[0] == "-h"))) {
        print_build_help();
        return EXIT_SUCCESS;
    }

    std::string onnx_path;
    std::string output_path;
    std::string precision = "fp16";
    std::size_t workspace_mib = 4096;
    bool strict_types = false;
    bool verbose = false;
    std::string int8_cache;

    ShapeProfile profile;
    std::string current_input;
    std::vector<std::int64_t> current_min, current_opt, current_max;

    auto flush_input = [&] {
        if (current_input.empty()) return;
        if (current_min.empty() || current_opt.empty() || current_max.empty()) {
            throw std::invalid_argument("--input " + current_input +
                                        " is missing one of --min/--opt/--max-shape");
        }
        ShapeRange r;
        r.input_name = current_input;
        r.min_shape = current_min;
        r.opt_shape = current_opt;
        r.max_shape = current_max;
        profile.add(std::move(r));
        current_input.clear();
        current_min.clear();
        current_opt.clear();
        current_max.clear();
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        auto take_value = [&](const std::string& flag) {
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(flag + " expects a value");
            }
            return args[++i];
        };

        if (a == "--help" || a == "-h") { print_build_help(); return EXIT_SUCCESS; }
        else if (a == "--onnx") onnx_path = take_value(a);
        else if (a == "--output" || a == "-o") output_path = take_value(a);
        else if (a == "--precision") precision = take_value(a);
        else if (a == "--workspace-mib") workspace_mib = static_cast<std::size_t>(std::stoull(take_value(a)));
        else if (a == "--strict-types") strict_types = true;
        else if (a == "--verbose" || a == "-v") verbose = true;
        else if (a == "--int8-cache") int8_cache = take_value(a);
        else if (a == "--input") { flush_input(); current_input = take_value(a); }
        else if (a == "--min-shape") current_min = parse_shape(take_value(a));
        else if (a == "--opt-shape") current_opt = parse_shape(take_value(a));
        else if (a == "--max-shape") current_max = parse_shape(take_value(a));
        else throw std::invalid_argument("unknown build flag: " + a);
    }
    flush_input();

    if (onnx_path.empty() || output_path.empty()) {
        print_build_help();
        return EXIT_FAILURE;
    }

    trt_toolkit::utils::init_logger(verbose ? "debug" : "info", false);

    BuildOptions options;
    options.precision = parse_precision(precision);
    options.workspace_mib = workspace_mib;
    options.strict_types = strict_types;
    options.verbose = verbose;
    options.profile = std::move(profile);

    std::unique_ptr<Int8EntropyCalibrator> calibrator_owner;
    if (options.precision == Precision::INT8) {
        if (int8_cache.empty()) {
            throw std::invalid_argument(
                "INT8 build requires --int8-cache (and a calibrator with batch data)");
        }
        calibrator_owner = Int8EntropyCalibrator::from_cache(int8_cache);
        options.calibrator = calibrator_owner.get();
    }

    TrtBuilder builder;
    builder.build(onnx_path, output_path, options);
    return EXIT_SUCCESS;
}

// ============================================================================
// benchmark
// ============================================================================

void print_benchmark_help() {
    std::cout << "Usage: trt-toolkit benchmark --engine ENGINE [options]\n"
              << "\n"
              << "Optional:\n"
              << "  --batch N              Concrete batch for dynamic engines. Default: 1\n"
              << "  --warmup N             Default: 25\n"
              << "  --iterations N         Default: 200\n";
}

int benchmark_subcommand(const std::vector<std::string>& args) {
    using namespace trt_toolkit;

    std::string engine_path;
    std::size_t warmup = 25;
    std::size_t iterations = 200;
    std::int64_t batch = 1;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        auto take_value = [&](const std::string& flag) {
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(flag + " expects a value");
            }
            return args[++i];
        };
        if (a == "--help" || a == "-h") { print_benchmark_help(); return EXIT_SUCCESS; }
        else if (a == "--engine") engine_path = take_value(a);
        else if (a == "--batch") batch = std::stoll(take_value(a));
        else if (a == "--warmup") warmup = std::stoull(take_value(a));
        else if (a == "--iterations") iterations = std::stoull(take_value(a));
        else throw std::invalid_argument("unknown benchmark flag: " + a);
    }
    if (engine_path.empty()) {
        print_benchmark_help();
        return EXIT_FAILURE;
    }

    utils::init_logger("info", false);

    benchmark::MemoryProbe mem;
    mem.before();
    runner::TrtRunner trt_runner(engine_path);
    const auto mem_after_load = mem.after();

    // Resolve any dynamic dimensions to a concrete shape, then random-fill
    // every input. Engines exported with a dynamic batch (shape -1x...)
    // have no allocated device buffer until a concrete shape is set; this
    // is what `--batch` is for.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    utils::CudaStream stream;
    for (const auto& b : trt_runner.bindings()) {
        if (!b.is_input) continue;
        bool dynamic = false;
        std::vector<std::int64_t> resolved = b.shape;
        for (auto& d : resolved) {
            if (d < 0) {
                d = batch;
                dynamic = true;
            }
        }
        if (dynamic) {
            trt_runner.set_input_shape(b.name, resolved);
        }
        const auto& rb = trt_runner.binding(b.name);
        std::vector<float> blob(rb.volume * rb.element_size / sizeof(float), 0.0f);
        for (auto& v : blob) v = dist(rng);
        trt_runner.copy_input(b.name, blob.data(), blob.size() * sizeof(float), stream.get());
    }
    stream.synchronize();

    auto enqueue = [&] { trt_runner.infer(stream.get()); };

    benchmark::LatencyProbeOptions lp;
    lp.warmup = warmup;
    lp.iterations = iterations;
    auto lat = benchmark::measure_latency([&] { enqueue(); }, lp);

    benchmark::ThroughputProbeOptions tp;
    tp.warmup = warmup;
    tp.iterations = iterations;
    tp.batch_size = static_cast<std::size_t>(batch);
    auto thr = benchmark::measure_throughput([&] { enqueue(); },
                                             [&] { stream.synchronize(); }, tp);

    std::cout << "\n[memory after engine load]\n  ";
    mem_after_load.format(std::cout);
    std::cout << "\n\n[latency]\n  ";
    lat.format(std::cout);
    std::cout << "\n\n[throughput]\n  ";
    thr.format(std::cout);
    std::cout << "\n";
    return EXIT_SUCCESS;
}

// ============================================================================
// accuracy
// ============================================================================

void print_accuracy_help() {
    std::cout
        << "Usage: trt-toolkit accuracy --reference REF.engine --candidate CAND.engine\n"
        << "\n"
        << "Generates random input and reports element-wise diff statistics.\n";
}

int accuracy_subcommand(const std::vector<std::string>& args) {
    using namespace trt_toolkit;

    std::string ref;
    std::string cand;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        auto take_value = [&](const std::string& flag) {
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(flag + " expects a value");
            }
            return args[++i];
        };
        if (a == "--help" || a == "-h") { print_accuracy_help(); return EXIT_SUCCESS; }
        else if (a == "--reference") ref = take_value(a);
        else if (a == "--candidate") cand = take_value(a);
        else throw std::invalid_argument("unknown accuracy flag: " + a);
    }
    if (ref.empty() || cand.empty()) {
        print_accuracy_help();
        return EXIT_FAILURE;
    }

    utils::init_logger("info", false);

    // Determine input shape from the reference engine to size the
    // random blob we generate.
    runner::TrtRunner ref_runner(ref);
    accuracy::EngineDiffOptions options;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const auto& b : ref_runner.bindings()) {
        if (!b.is_input) continue;
        std::vector<float> blob(b.volume * b.element_size / sizeof(float), 0.0f);
        for (auto& v : blob) v = dist(rng);
        options.input_blobs.push_back(std::move(blob));
    }

    const auto report = accuracy::diff_engines(ref, cand, options);
    std::cout << "\n[accuracy diff: reference vs candidate]\n";
    for (std::size_t i = 0; i < report.binding_names.size(); ++i) {
        std::cout << "  " << report.binding_names[i] << ":  ";
        report.per_output[i].format(std::cout);
        std::cout << "\n";
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return EXIT_SUCCESS;
    }

    const std::string subcommand = argv[1];
    const std::vector<std::string> rest(argv + 2, argv + argc);

    try {
        if (subcommand == "build") return build_subcommand(rest);
        if (subcommand == "benchmark") return benchmark_subcommand(rest);
        if (subcommand == "accuracy") return accuracy_subcommand(rest);
        if (subcommand == "inspect") return inspect_subcommand(rest);
        if (subcommand == "--help" || subcommand == "-h") {
            print_help();
            return EXIT_SUCCESS;
        }
        std::cerr << "unknown subcommand: " << subcommand << "\n\n";
        print_help();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("error: {}", e.what());
        return EXIT_FAILURE;
    }
}
