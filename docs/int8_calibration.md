# INT8 calibration guide

INT8 quantisation in TensorRT works by collecting per-tensor
activation histograms during a *calibration pass*, then fitting a
scale factor that minimises the KL divergence between the FP32 and
the quantised distribution. The toolkit implements
`IInt8EntropyCalibrator2`, which is the recommended calibrator for
post-training quantisation.

## What you need

1. **An ONNX model** that already runs in FP32.
2. **A representative calibration data set** - typically 200-1000
   samples drawn from your validation set or production traffic. The
   data must go through *exactly* the same preprocessing as
   inference time; mismatched preprocessing is the most common cause
   of "INT8 looks worse than expected" reports.
3. **Disk space for the calibration cache** - usually a few KiB per
   layer. The cache makes the calibration pass deterministic across
   rebuilds, which is useful for CI.

## Recipe

### Step 1: produce binary samples

The toolkit's `make_binary_folder_provider()` reads `*.bin` files
from a directory, each one a single C-contiguous tensor of
preprocessed values. A short Python snippet:

```python
import numpy as np
from PIL import Image
from pathlib import Path

src_dir = Path("calib_images")
dst_dir = Path("calib_bin")
dst_dir.mkdir(exist_ok=True)

for i, p in enumerate(sorted(src_dir.iterdir())[:500]):
    img = Image.open(p).convert("RGB").resize((640, 640))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr.tofile(dst_dir / f"{i:05d}.bin")
```

### Step 2: build the engine

```cpp
auto provider = builder::make_binary_folder_provider(
    "calib_bin", "images", {3, 640, 640}, /*element_size=*/4, /*batch=*/8);

builder::Int8EntropyCalibrator calibrator(std::move(provider),
                                          "calib_cache/yolov8s.cache");

builder::BuildOptions opts;
opts.precision = builder::Precision::INT8;
opts.calibrator = &calibrator;
opts.profile = ...; // dynamic shape profile

builder::TrtBuilder().build("yolov8s.onnx", "yolov8s_int8.engine", opts);
```

The first build will run calibration end-to-end. Subsequent builds
reuse the cache file unless you delete it.

## Common pitfalls

- **Too few samples.** 50 calibration samples is rarely enough;
  the histograms are too narrow and the resulting scales clip real
  activations. Aim for 500+.
- **Class-imbalanced calibration data.** If your model has 80
  classes but your 500 calibration samples only cover 5 of them,
  the rare-class tensors will be poorly quantised. Stratify your
  sampling.
- **Cache file mismatched with model.** TensorRT does not version
  the cache against the model. If you change the ONNX (even a layer
  rename), regenerate the cache; otherwise you will silently load a
  stale calibration.
- **CUDA driver / TRT version drift.** A cache produced under TRT
  8.6 may not be compatible with TRT 9.0. Treat it as an
  installation-specific artefact.
- **Thinking INT8 will "just be faster".** It usually is, but
  attention layers in transformers can fall back to FP16 internally,
  giving you a smaller speedup than expected. Use
  `trt-toolkit benchmark` on both FP16 and INT8 builds before
  committing to either.

## Validation

Always pair an INT8 build with `trt-toolkit accuracy` against the
FP16 (or FP32) reference engine. Acceptable thresholds are
domain-specific, but as a starting point:

| Workload         | Acceptable max_rel_diff |
|------------------|------------------------:|
| Image classification | 1e-2                |
| Object detection     | 5e-3                |
| Face recognition     | 1e-3                |
| Generative models    | 1e-4 or finer       |
