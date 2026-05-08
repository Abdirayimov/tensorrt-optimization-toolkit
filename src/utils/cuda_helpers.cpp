#include "trt_toolkit/utils/cuda_helpers.hpp"

#include <NvInfer.h>

namespace trt_toolkit::utils {

std::size_t element_size_for(int trt_data_type) {
    using DT = nvinfer1::DataType;
    switch (static_cast<DT>(trt_data_type)) {
        case DT::kFLOAT: return 4;
        case DT::kHALF:  return 2;
        case DT::kINT8:  return 1;
        case DT::kINT32: return 4;
        case DT::kBOOL:  return 1;
        case DT::kUINT8: return 1;
        case DT::kFP8:   return 1;
        default: return 0;
    }
}

}  // namespace trt_toolkit::utils
