cmake_minimum_required(VERSION 3.22)
project(trt_toolkit VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TOOLS "Build trt-toolkit CLI" ON)
option(BUILD_TESTS "Build smoke tests" OFF)

# ---- Dependencies ----
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(spdlog REQUIRED)
find_package(TensorRT REQUIRED)

# ---- Library: trt_toolkit (static) ----
file(GLOB_RECURSE TRT_TOOLKIT_SOURCES
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/builder/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/runner/*.cpp"
)

add_library(trt_toolkit STATIC ${TRT_TOOLKIT_SOURCES})
target_include_directories(trt_toolkit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(trt_toolkit
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        spdlog::spdlog
        TensorRT::nvinfer
        TensorRT::nvonnxparser
)
set_target_properties(trt_toolkit PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TOOLS)
    add_executable(trt-toolkit tools/trt_toolkit_cli.cpp)
    target_link_libraries(trt-toolkit PRIVATE trt_toolkit)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS trt_toolkit
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
if(BUILD_TOOLS)
    install(TARGETS trt-toolkit RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)

message(STATUS "")
message(STATUS "trt_toolkit ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:         ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT:     ${TensorRT_VERSION}")
message(STATUS "  Build tools:  ${BUILD_TOOLS}")
message(STATUS "  Build tests:  ${BUILD_TESTS}")
message(STATUS "")
