"""Latency: measure ms per generated token (autoregressive decode)."""

import torch
import time


def measure_latency(model, tokenizer, device: str = "cpu",
                     prompt: str = "The future of on-device AI is",
                     n_tokens: int = 50, n_warmup: int = 3) -> float:
    """
    Measure autoregressive decode latency in ms/token.

    Args:
        n_tokens:  number of tokens to generate for timing
        n_warmup:  warmup runs to avoid cold-start noise

    Returns:
        float: milliseconds per token
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    # Timed run
    # Sync before timing (MPS/CUDA)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False)

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / n_tokens
