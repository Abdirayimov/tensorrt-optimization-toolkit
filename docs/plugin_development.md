# Plugin development

TensorRT covers most common ops out of the box, but eventually you
will need a custom layer: an exotic activation, a fused custom op,
or a domain-specific operation that the ONNX parser cannot map. This
page walks through the lifecycle using the toolkit's bundled
`GeluPlugin`.

## Choose your interface

TensorRT 8.x exposes two relevant abstractions for new plugins:

- `IPluginV2DynamicExt` - the workhorse since TRT 7.x. Mature,
  battle-tested, lots of examples on the open internet.
- `IPluginV3OneBuild` / `IPluginV3OneRuntime` - the new API in
  TRT 10.x. Cleaner separation of build-time vs runtime
  responsibilities, but tooling and examples are still catching up.

The toolkit ships `GeluPlugin` against `IPluginV2DynamicExt`. When
your minimum supported TRT version is 11, migrate to V3 - the
mechanical changes are listed at the end of this document.

## Lifecycle of `IPluginV2DynamicExt`

```
Builder phase                            Runtime phase
-----------------                        --------------------
PluginCreator::createPlugin              IRuntime deserialises engine
   v                                     PluginCreator::deserializePlugin
configurePlugin (shapes, types)             v
supportsFormatCombination                clone (per-context)
getOutputDimensions                         v
getWorkspaceSize                         configurePlugin (one-time)
                                         enqueue (per-frame)
                                         destroy (on engine teardown)
```

Things to remember:

- `clone()` is called *both* at build time (when the network is
  refined) and at runtime (per-context). Make it cheap and stateless
  except for the static config.
- `enqueue()` runs on a CUDA stream that you must respect. Synchronise
  internally only when absolutely necessary.
- `serialize()` and `getSerializationSize()` are paired; the engine
  cache stores whatever you write here and feeds it back to your
  deserializing constructor.
- `supportsFormatCombination()` is called for every position in the
  bindings array. Be defensive: return false unless the exact
  combination you implemented is requested.

## Registration

At process start, before parsing ONNX:

```cpp
trt_toolkit::plugin::register_builtin_plugins();
```

This calls `initLibNvInferPlugins` (NVIDIA's stock plugin pack) when
available, then registers `GeluPlugin`'s creator. Registration is
idempotent.

## Where the kernel lives

`src/plugin/gelu_kernel.cpp` is intentionally a `.cpp` file that
implements the kernel as a host-fallback loop. This keeps the build
toolchain minimal: you do not need `nvcc` to compile the toolkit.

For real workloads you should:

1. Rename `gelu_kernel.cpp` to `gelu_kernel.cu`.
2. Enable the CUDA language in CMake:
   ```cmake
   project(trt_toolkit VERSION 0.1.0 LANGUAGES CXX CUDA)
   ```
3. Replace the body with a `__global__` kernel and a launch
   configuration that matches your problem size.

A reasonable starting kernel:

```cuda
template <typename T>
__global__ void gelu_kernel(const T* in, T* out, std::size_t n) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float k0 = 0.7978845608028654f;
    constexpr float k1 = 0.044715f;
    const float x = static_cast<float>(in[i]);
    const float inner = k0 * (x + k1 * x * x * x);
    const float y = 0.5f * x * (1.0f + tanhf(inner));
    out[i] = static_cast<T>(y);
}
```

## Migrating to IPluginV3

The mechanical changes are roughly:

| V2DynamicExt method        | V3 successor                                    |
|----------------------------|-------------------------------------------------|
| `getOutputDimensions`      | `IPluginV3OneBuild::getOutputShapes`            |
| `supportsFormatCombination`| `IPluginV3OneBuild::supportsFormatCombination`  |
| `configurePlugin`          | `IPluginV3OneBuild::configurePlugin`            |
| `enqueue`                  | `IPluginV3OneRuntime::enqueue`                  |
| `getWorkspaceSize`         | `IPluginV3OneBuild::getWorkspaceSize`           |
| `IPluginCreator`           | `IPluginCreatorV3One`                           |

The data members and CUDA kernels are unchanged; you are essentially
splitting the existing methods across the two runtime interfaces.
