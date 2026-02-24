"""
LLM Quantization Benchmark
============================
Compares 7 quantization methods on Qwen2-0.5B:
  - FP16 baseline
  - Naive RTN  (INT8, INT4)
  - GPTQ       (INT4)
  - AWQ        (INT4)
  - SmoothQuant(INT8)
  - Palletization 4-bit (Apple CoreML-style, PyTorch simulation)
  - QLoRA / NF4

Metrics: Perplexity (WikiText-2), Memory (MB), Latency (ms/token), Compression ratio
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_methods.rtn import apply_rtn
from quant_methods.gptq import apply_gptq
from quant_methods.awq import apply_awq
from quant_methods.smoothquant import apply_smoothquant
from quant_methods.palletization import apply_palletization
from quant_methods.qlora_nf4 import apply_qlora_nf4
from evaluation.perplexity import compute_perplexity
from evaluation.latency import measure_latency
from evaluation.memory import measure_memory_mb

MODEL_ID = "Qwen/Qwen2-0.5B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

METHODS = {
    "fp16_baseline":      {"fn": None,               "bits": 16, "label": "FP16 Baseline"},
    "rtn_int8":           {"fn": apply_rtn,           "bits": 8,  "label": "RTN INT8",        "kwargs": {"bits": 8}},
    "rtn_int4":           {"fn": apply_rtn,           "bits": 4,  "label": "RTN INT4",        "kwargs": {"bits": 4}},
    "gptq_int4":          {"fn": apply_gptq,          "bits": 4,  "label": "GPTQ INT4",       "kwargs": {"bits": 4}},
    "awq_int4":           {"fn": apply_awq,           "bits": 4,  "label": "AWQ INT4",        "kwargs": {"bits": 4}},
    "smoothquant_int8":   {"fn": apply_smoothquant,   "bits": 8,  "label": "SmoothQuant INT8","kwargs": {"bits": 8}},
    "palletization_4bit": {"fn": apply_palletization, "bits": 4,  "label": "Palletization 4-bit", "kwargs": {"bits": 4}},
    "qlora_nf4":          {"fn": apply_qlora_nf4,     "bits": 4,  "label": "QLoRA NF4",       "kwargs": {}},
}


def load_base_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def run_benchmark(methods_to_run: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in methods_to_run:
        cfg = METHODS[name]
        print(f"\n{'='*60}")
        print(f"  Method: {cfg['label']}")
        print(f"{'='*60}")

        # Reload fresh model for each method
        model, tokenizer = load_base_model()

        # Apply quantization
        if cfg["fn"] is not None:
            print(f"  Applying {cfg['label']} ...")
            model = cfg["fn"](model, tokenizer, **cfg.get("kwargs", {}))

        model = model.to(DEVICE)

        # Measure memory
        mem_mb = measure_memory_mb(model)

        # Measure latency
        lat_ms = measure_latency(model, tokenizer, device=DEVICE)

        # Measure perplexity
        ppl = compute_perplexity(model, tokenizer, device=DEVICE)

        # Baseline FP16 model size (for compression ratio)
        fp16_params = sum(p.numel() for p in model.parameters()) * 2 / 1e6  # rough MB
        compression = fp16_params / mem_mb if mem_mb > 0 else 1.0

        results[name] = {
            "label":             cfg["label"],
            "bits":              cfg["bits"],
            "perplexity":        round(ppl, 3),
            "memory_mb":         round(mem_mb, 1),
            "latency_ms_token":  round(lat_ms, 2),
            "compression_ratio": round(compression, 2),
        }

        print(f"  ✓ PPL={ppl:.3f} | Mem={mem_mb:.1f}MB | Lat={lat_ms:.2f}ms/tok | Compression={compression:.2f}x")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_summary_table(results)
    return results


def print_summary_table(results: dict):
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Bits':>4} {'PPL':>8} {'Mem(MB)':>9} {'Lat(ms)':>9} {'Compress':>10}")
    print(f"{'-'*80}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["perplexity"]):
        print(f"{r['label']:<25} {r['bits']:>4} {r['perplexity']:>8.3f} "
              f"{r['memory_mb']:>9.1f} {r['latency_ms_token']:>9.2f} {r['compression_ratio']:>9.2f}x")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help="Which methods to run (default: all)")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    run_benchmark(args.methods, Path(args.output))
