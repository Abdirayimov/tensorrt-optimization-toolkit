# Building

## Native (Ubuntu 22.04)

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    libspdlog-dev
```

CUDA 12.x and TensorRT 8.6+ must be installed separately from NVIDIA;
the toolkit's `cmake/FindTensorRT.cmake` searches the standard paths.
Override with `-DTensorRT_ROOT=/path/to/tensorrt` when needed.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Targets produced:

- `build/libtrt_toolkit.a` - the static library
- `build/trt-toolkit` - CLI (build / benchmark / accuracy / inspect)
- `build/01_resnet50_fp16` ... `build/04_custom_plugin` - examples

## CMake options

| Option           | Default | Effect                                     |
|------------------|--------:|--------------------------------------------|
| `BUILD_TOOLS`    | ON      | Build the `trt-toolkit` CLI binary         |
| `BUILD_EXAMPLES` | ON      | Build all `examples/*.cpp` as executables  |
| `BUILD_TESTS`    | OFF     | Build smoke tests (planned)                |

Disable examples in CI:

```bash
cmake -S . -B build -G Ninja -DBUILD_EXAMPLES=OFF
```

## Optional dependencies

`libnvinfer_plugin` is detected automatically. When absent, the
toolkit defines `TRT_NO_NVINFER_PLUGIN` and skips
`initLibNvInferPlugins`; stock NVIDIA plugins (NMS, ROIAlign, ...)
will not be available, but the toolkit's own `GeluPlugin` still
registers correctly.

## Common errors

**`Could NOT find TensorRT`** - point CMake at your install:

```bash
cmake -S . -B build -DTensorRT_ROOT=/usr/local/tensorrt
```

**`fatal error: NvInferVersion.h: No such file or directory`** - same
fix; the find module needs the include directory to contain
`NvInfer.h` and `NvInferVersion.h`.

**`-Wconversion` warnings flooding the build log** - the toolkit
opts in to a strict warning set. Use `-DCMAKE_CXX_FLAGS=-Wno-conversion`
in client projects that link against `trt_toolkit` if you do not want
the same warnings in your code.

**`enqueueV3 failed` at runtime** - the engine's expected input shape
does not match what the runner is providing. Check
`trt-toolkit inspect ENGINE` against the input shape you are
configuring.
