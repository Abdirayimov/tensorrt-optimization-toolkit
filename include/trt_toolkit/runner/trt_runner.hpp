#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
}  // namespace nvinfer1

namespace trt_toolkit::runner {

struct BindingInfo {
    std::string name;
    std::vector<std::int64_t> shape;
    std::size_t element_size = 0;
    std::size_t volume = 0;
    bool is_input = false;
};

/// RAII wrapper around a deserialized engine + execution context.
///
/// Owns one device buffer per binding sized for the static (or
/// max-profile) shape; for dynamic-shape engines call
/// `set_input_shape()` per inference to reallocate the input buffer
/// and notify TRT.
///
/// Move-only. Not safe to share across threads concurrently.
class TrtRunner {
public:
    explicit TrtRunner(const std::filesystem::path& engine_path);
    ~TrtRunner();

    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;
    TrtRunner(TrtRunner&&) noexcept;
    TrtRunner& operator=(TrtRunner&&) noexcept;

    void set_input_shape(const std::string& name, const std::vector<std::int64_t>& shape);

    void copy_input(const std::string& name, const void* host_src, std::size_t bytes,
                    cudaStream_t stream);
    void copy_output(const std::string& name, void* host_dst, std::size_t bytes,
                     cudaStream_t stream) const;

    void infer(cudaStream_t stream);

    void* device_ptr(const std::string& name);
    const void* device_ptr(const std::string& name) const;

    const std::vector<BindingInfo>& bindings() const noexcept { return bindings_; }
    const BindingInfo& binding(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<BindingInfo> bindings_;
};

}  // namespace trt_toolkit::runner
