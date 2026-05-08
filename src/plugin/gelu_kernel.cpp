// gelu_kernel.cpp - launcher for the GELU CUDA kernel.
//
// Real production code would put this in a .cu file with a __global__
// kernel. We keep it as a regular .cpp so the toolkit builds without a
// CUDA compiler (the ::cudaMemcpyAsync fallback below is illustrative,
// not optimal). To enable the proper kernel:
//
//   1. Rename this file to gelu_kernel.cu
//   2. Add `find_package(CUDA REQUIRED)` and enable the CUDA language
//   3. Replace the loop with a templated __global__ kernel

#include <cuda_runtime.h>

#include <NvInferRuntimeBase.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace trt_toolkit::plugin {

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCubicCoeff = 0.044715f;

float gelu_scalar(float x) {
    const float inner = kSqrt2OverPi * (x + kCubicCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

}  // namespace

extern "C" void launch_gelu_kernel(const void* in_dev, void* out_dev, std::size_t n,
                                   nvinfer1::DataType dtype, cudaStream_t stream) {
    if (dtype != nvinfer1::DataType::kFLOAT) {
        // Reference plugin: only FP32 is implemented in this fallback
        // path. A real kernel would specialise for kHALF too.
        return;
    }

    // Host fallback: copy device -> host, compute, copy back. This is
    // *only* useful as a reference; replace with a CUDA kernel for real
    // workloads.
    std::vector<float> host(n);
    cudaMemcpyAsync(host.data(), in_dev, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (std::size_t i = 0; i < n; ++i) {
        host[i] = gelu_scalar(host[i]);
    }
    cudaMemcpyAsync(out_dev, host.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
}

}  // namespace trt_toolkit::plugin
