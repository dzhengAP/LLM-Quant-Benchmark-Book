"""Memory: measure model parameter footprint in MB."""

import torch
import torch.nn as nn


def measure_memory_mb(model) -> float:
    """
    Compute total memory footprint of all model buffers and parameters.

    This measures the actual storage used by quantized weights (e.g., int8
    indices for palletization, int8 weights for RTN) rather than the logical
    parameter count, giving a true compressed size estimate.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / 1e6  # bytes -> MB
