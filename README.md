# LLM Quantization Benchmark

A comprehensive, from-scratch comparison of 7 quantization methods on **Qwen2-0.5B**, including Apple's **Palletization** (CoreML-style LUT quantization). Every algorithm is implemented explicitly — no black-box wrappers.

## Methods

| Method | Type | Bits | Key Idea |
|---|---|---|---|
| FP16 Baseline | — | 16 | No quantization |
| RTN INT8 | PTQ | 8 | Round-to-nearest, per-channel scale |
| RTN INT4 | PTQ | 4 | Same, more aggressive |
| SmoothQuant | PTQ | 8 | Migrate outliers from activations → weights |
| GPTQ | PTQ | 4 | Hessian-based column-wise error compensation |
| AWQ | PTQ | 4 | Activation-aware channel scaling before quant |
| **Palletization 4-bit** | **Codebook** | **4** | **K-means LUT, Apple CoreML style** |
| QLoRA NF4 | PTQ | ~4.1 | Normal Float quantile grid + double quantization |

## Why Palletization is Different

All other methods in this list use **linear quantization**: map weights to a uniform (or calibrated-scale) numeric grid. Palletization uses a **codebook (lookup table)**:

```
Linear INT4:     weight → round(w / scale) * scale    (16 uniform levels)
Palletization:   weight → LUT[argmin_k |w - LUT[k]|]  (16 learned levels)
```

The LUT entries are optimized by k-means to cluster where the weight distribution is **dense**, not uniformly spaced. For bell-shaped weight distributions (common in attention layers), this consistently achieves lower MSE than INT4 at the same 4-bit budget.

**At runtime on Apple ANE**: weights are stored as 4-bit indices, and the ANE executes a hardware-native `gather(LUT, indices)` → compute pattern, giving real throughput gains without the overhead of dequantization kernels.

## Project Structure

```
quant-benchmark/
├── benchmark.py                    # Main runner: load → quantize → evaluate → table
├── quant_benchmark.ipynb           # Jupyter notebook with visualizations
├── quant_methods/
│   ├── rtn.py                      # Naive Round-to-Nearest (RTN INT8/INT4)
│   ├── gptq.py                     # GPTQ with Hessian/Cholesky
│   ├── awq.py                      # AWQ with activation-aware scaling
│   ├── smoothquant.py              # SmoothQuant activation migration
│   ├── palletization.py            # K-means LUT + PalletizedLinear layer
│   └── qlora_nf4.py                # NF4 + double quantization
├── evaluation/
│   ├── perplexity.py               # WikiText-2 sliding window PPL
│   ├── latency.py                  # ms/token autoregressive decode
│   └── memory.py                  # Parameter + buffer footprint
└── results/                        # JSON results + plots saved here
```

## Setup

```bash
pip install torch transformers datasets matplotlib pandas

# For real CoreML palletization (macOS only):
pip install coremltools
```

## Running

```bash
# All methods (20-40 min depending on hardware)
python benchmark.py

# Specific methods
python benchmark.py --methods fp16_baseline rtn_int4 gptq_int4 palletization_4bit

# Results saved to results/results.json
```

Or open `quant_benchmark.ipynb` for interactive exploration with visualizations.

## Metrics

| Metric | What it measures |
|---|---|
| **Perplexity** | Language model quality on WikiText-2 test (lower = better) |
| **Memory (MB)** | Actual buffer + parameter footprint of quantized model |
| **Latency (ms/tok)** | Autoregressive decode speed (50 tokens, after warmup) |
| **Compression ratio** | FP16 size / quantized size |

## Palletization vs CoreML

This repo implements palletization in **pure PyTorch** (works cross-platform). On macOS, you can use `coremltools` for production-grade ANE acceleration:

```python
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
)

config = PostTrainingPalettizerConfig.from_dict({
    "global_config": {
        "n_bits": 4,
        "granularity": "per_grouped_channel",
        "group_size": 128,
        "enable_per_channel_scale": True,
    }
})
palettizer = PostTrainingPalettizer(model, config)
palettized_model = palettizer.compress()
```

See the CoreML cell in `quant_benchmark.ipynb` for the full conversion pipeline.

## Key Takeaways

**Palletization 4-bit**
- Adapts quantization grid to weight distribution via k-means
- Achieves comparable PPL to GPTQ/AWQ without Hessian computation
- Apple ANE has native gather-LUT hardware support → real on-device speedup
- Requires no calibration data (unlike GPTQ/AWQ/SmoothQuant)

**GPTQ INT4**
- Best perplexity of all PTQ methods — preferred when accuracy is paramount
- Slow to apply: O(d²) Cholesky per layer
- No native hardware acceleration for dequantize kernel on ANE

**AWQ INT4**
- Nearly as good as GPTQ, much faster to apply (only needs activation means)
- Hardware-friendly: compatible with grouped INT4 GEMM kernels

**NF4 (QLoRA)**
- Best effective bits-per-weight: ~4.127 bpw with double quantization
- Designed for fine-tuning, not pure inference acceleration
- Good choice when you need both small footprint and LoRA adapter training

**SmoothQuant INT8**
- Enables W8A8 quantization — requires INT8 GEMM kernel for real speedup
- Better perplexity than RTN INT8 due to activation smoothing

## References

- [GPTQ](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [AWQ](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., 2022
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [CoreML Palettization](https://coremltools.readme.io/docs/palettization-overview) — Apple, 2023
