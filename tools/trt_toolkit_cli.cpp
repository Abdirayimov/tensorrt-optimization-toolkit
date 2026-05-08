// trt-toolkit CLI: minimal `build` subcommand for now. Future
// subcommands (`benchmark`, `accuracy`, `inspect`) ship in later
// sessions.
//
// Usage:
//   trt-toolkit build --onnx model.onnx --output model.engine
//                     [--precision fp32|fp16|int8]
//                     [--workspace-mib N]
//                     [--input NAME --min-shape S --opt-shape S --max-shape S]
//                     [--int8-cache PATH]
//                     [--verbose]

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "trt_toolkit/builder/dynamic_shapes.hpp"
#include "trt_toolkit/builder/engine_builder.hpp"
#include "trt_toolkit/builder/int8_calibrator.hpp"
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
        << "trt-toolkit - ONNX -> TensorRT engine compilation\n"
        << "\n"
        << "Subcommands:\n"
        << "  build    Compile an ONNX file into a serialized .engine\n"
        << "\n"
        << "Run `trt-toolkit build --help` for build-specific options.\n";
}

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

        if (a == "--help" || a == "-h") {
            print_build_help();
            return EXIT_SUCCESS;
        } else if (a == "--onnx") {
            onnx_path = take_value(a);
        } else if (a == "--output" || a == "-o") {
            output_path = take_value(a);
        } else if (a == "--precision") {
            precision = take_value(a);
        } else if (a == "--workspace-mib") {
            workspace_mib = static_cast<std::size_t>(std::stoull(take_value(a)));
        } else if (a == "--strict-types") {
            strict_types = true;
        } else if (a == "--verbose" || a == "-v") {
            verbose = true;
        } else if (a == "--int8-cache") {
            int8_cache = take_value(a);
        } else if (a == "--input") {
            flush_input();
            current_input = take_value(a);
        } else if (a == "--min-shape") {
            current_min = parse_shape(take_value(a));
        } else if (a == "--opt-shape") {
            current_opt = parse_shape(take_value(a));
        } else if (a == "--max-shape") {
            current_max = parse_shape(take_value(a));
        } else {
            throw std::invalid_argument("unknown build flag: " + a);
        }
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
                "INT8 build requires --int8-cache (and a calibrator with batch data; "
                "the cache-only path is supported here for builds against pre-baked caches)");
        }
        calibrator_owner = Int8EntropyCalibrator::from_cache(int8_cache);
        options.calibrator = calibrator_owner.get();
    }

    TrtBuilder builder;
    builder.build(onnx_path, output_path, options);
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
        if (subcommand == "build") {
            return build_subcommand(rest);
        }
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
