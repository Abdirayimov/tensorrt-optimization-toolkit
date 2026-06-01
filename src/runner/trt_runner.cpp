#include "trt_toolkit/runner/trt_runner.hpp"

#include <NvInfer.h>

#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "trt_toolkit/utils/cuda_helpers.hpp"
#include "trt_toolkit/utils/logger.hpp"

namespace trt_toolkit::runner {

namespace {

std::size_t volume_of(const std::vector<std::int64_t>& shape) {
    std::size_t v = 1;
    for (auto d : shape) {
        if (d <= 0) return 0;
        v *= static_cast<std::size_t>(d);
    }
    return v;
}

}  // namespace

struct TrtRunner::Impl {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::unordered_map<std::string, void*> device_buffers;
};

TrtRunner::TrtRunner(const std::filesystem::path& engine_path)
    : impl_(std::make_unique<Impl>()) {
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open engine: " + engine_path.string());
    }
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<std::size_t>(sz));
    f.read(blob.data(), sz);

    impl_->runtime.reset(nvinfer1::createInferRuntime(utils::tensorrt_logger()));
    if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine failed");
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) throw std::runtime_error("createExecutionContext failed");

    const std::int32_t n = impl_->engine->getNbIOTensors();
    bindings_.reserve(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
        const char* name = impl_->engine->getIOTensorName(i);
        const auto dims = impl_->engine->getTensorShape(name);
        const auto dtype = impl_->engine->getTensorDataType(name);
        const auto io = impl_->engine->getTensorIOMode(name);

        BindingInfo info;
        info.name = name;
        info.shape.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t k = 0; k < dims.nbDims; ++k) {
            info.shape.push_back(dims.d[k]);
        }
        info.element_size = utils::element_size_for(static_cast<int>(dtype));
        info.volume = volume_of(info.shape);
        info.is_input = (io == nvinfer1::TensorIOMode::kINPUT);

        if (info.volume > 0 && info.element_size > 0) {
            void* ptr = nullptr;
            TRT_CUDA_CHECK(cudaMalloc(&ptr, info.volume * info.element_size));
            impl_->device_buffers[info.name] = ptr;
            impl_->context->setTensorAddress(name, ptr);
        }

        bindings_.push_back(std::move(info));
    }
}

TrtRunner::~TrtRunner() {
    if (impl_) {
        for (auto& [_, ptr] : impl_->device_buffers) {
            if (ptr != nullptr) cudaFree(ptr);
        }
    }
}

TrtRunner::TrtRunner(TrtRunner&&) noexcept = default;
TrtRunner& TrtRunner::operator=(TrtRunner&&) noexcept = default;

void TrtRunner::set_input_shape(const std::string& name,
                                const std::vector<std::int64_t>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dims.d[i] = shape[i];
    }
    if (!impl_->context->setInputShape(name.c_str(), dims)) {
        throw std::runtime_error("setInputShape failed for " + name);
    }

    auto& info = const_cast<BindingInfo&>(binding(name));
    info.shape = shape;
    info.volume = volume_of(shape);

    void*& ptr = impl_->device_buffers[name];
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
    TRT_CUDA_CHECK(cudaMalloc(&ptr, info.volume * info.element_size));
    impl_->context->setTensorAddress(name.c_str(), ptr);

    // Setting an input shape can make previously-dynamic output shapes
    // concrete. Re-query every output binding from the context and
    // (re)allocate its device buffer so the outputs are bound before
    // enqueue. Without this, a dynamic-batch engine would have no
    // address set for its outputs and enqueueV3 would fail.
    for (auto& out : bindings_) {
        if (out.is_input) continue;
        const auto dims = impl_->context->getTensorShape(out.name.c_str());
        if (dims.nbDims < 0) continue;  // not yet resolvable
        std::vector<std::int64_t> resolved;
        resolved.reserve(static_cast<std::size_t>(dims.nbDims));
        bool concrete = true;
        for (std::int32_t k = 0; k < dims.nbDims; ++k) {
            resolved.push_back(dims.d[k]);
            if (dims.d[k] < 0) concrete = false;
        }
        if (!concrete) continue;
        const std::size_t new_vol = volume_of(resolved);
        if (new_vol == out.volume && impl_->device_buffers[out.name] != nullptr) {
            continue;  // unchanged, already allocated
        }
        out.shape = resolved;
        out.volume = new_vol;
        void*& optr = impl_->device_buffers[out.name];
        if (optr != nullptr) {
            cudaFree(optr);
            optr = nullptr;
        }
        TRT_CUDA_CHECK(cudaMalloc(&optr, out.volume * out.element_size));
        impl_->context->setTensorAddress(out.name.c_str(), optr);
    }
}

void TrtRunner::copy_input(const std::string& name, const void* host_src, std::size_t bytes,
                           cudaStream_t stream) {
    void* dst = impl_->device_buffers.at(name);
    TRT_CUDA_CHECK(cudaMemcpyAsync(dst, host_src, bytes, cudaMemcpyHostToDevice, stream));
}

void TrtRunner::copy_output(const std::string& name, void* host_dst, std::size_t bytes,
                            cudaStream_t stream) const {
    void* src = impl_->device_buffers.at(name);
    TRT_CUDA_CHECK(cudaMemcpyAsync(host_dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}

void TrtRunner::infer(cudaStream_t stream) {
    if (!impl_->context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}

void* TrtRunner::device_ptr(const std::string& name) {
    return impl_->device_buffers.at(name);
}

const void* TrtRunner::device_ptr(const std::string& name) const {
    return impl_->device_buffers.at(name);
}

const BindingInfo& TrtRunner::binding(const std::string& name) const {
    for (const auto& b : bindings_) {
        if (b.name == name) return b;
    }
    throw std::out_of_range("no such binding: " + name);
}

}  // namespace trt_toolkit::runner
