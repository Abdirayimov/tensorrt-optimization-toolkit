# Debugging accuracy drops

You converted your model from FP32 to FP16 (or INT8), the engine
compiles, latency is great - and the model is wrong. This page is a
short playbook for narrowing down *which* layer is the culprit.

## 1. Confirm the drop is real

Run `trt-toolkit accuracy` against an FP32 reference engine first.
Random-input diffing is a quick smoke test, but for a real signal you
want a held-out validation set.

If `cosine_similarity` is below 0.99 on random input, the problem is
almost always in the conversion (a numerically unsafe layer running
at lower precision). If cosine is high but task accuracy on real data
is poor, suspect the calibration set: it is unrepresentative.

## 2. Inspect the engine

```bash
trt-toolkit inspect model.engine
```

The summary lists every binding's dtype. If you asked for FP16 but a
binding is INT8, TRT decided that layer was unsafe at FP16 and fell
back; the resulting cast and re-cast is usually fine but worth
noting.

## 3. Bisect by layer precision

The crude-but-effective approach is to force layers above a certain
index to FP32 and rebuild:

```cpp
// Pseudo-code; the actual API is the same shape across recent TRT
// versions, just check setPrecision / setOutputType.
for (int i = 0; i < network->getNbLayers(); ++i) {
    auto* layer = network->getLayer(i);
    if (i >= cutoff_index) {
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    }
}
```

Pair with `BuilderFlag::kPREFER_PRECISION_CONSTRAINTS`
(exposed in this toolkit as `BuildOptions::strict_types = true`) so
TRT actually honours your precision request rather than silently
overriding it.

The first cutoff at which accuracy is restored tells you which layer
range is offending. From there, drop to per-layer overrides and find
the single culprit.

## 4. Common culprits

| Layer family           | Common failure mode                          |
|------------------------|----------------------------------------------|
| LayerNorm              | INT8 unsafe; force FP16 minimum              |
| Softmax (large input)  | Dynamic range too wide for INT8              |
| Sigmoid / Tanh tails   | Precision loss; accuracy drops on edge cases |
| GroupNorm with small G | Reductions amplify FP16 round-off            |
| Custom plugins         | Plugin author skipped FP16 path              |

## 5. Rebuild discipline

If you regenerate calibration caches, always rerun
`trt-toolkit accuracy`. Even small changes to the calibration data
can move per-tensor scales enough to change which layers fall back to
FP16.

## 6. When in doubt

`trtexec --onnx=model.onnx --fp16 --verbose` produces a wall of
output that explicitly states which tactic each layer chose. Reading
through it once for a model you do understand is the cheapest way to
build intuition for what TRT is doing under the hood.
