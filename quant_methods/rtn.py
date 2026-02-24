"""
Naive Round-to-Nearest (RTN) Quantization
==========================================
Simplest PTQ baseline. Quantizes each weight tensor independently using
symmetric min-max scaling. No calibration data needed.

Math:
    scale = max(|W|) / (2^(bits-1) - 1)
    W_q   = clamp(round(W / scale), -128, 127)   # for INT8
    W_deq = W_q * scale                            # dequantized for inference

Pros: Zero overhead, instant, no data needed
Cons: No error compensation — worst perplexity of all PTQ methods
      Sensitive to outliers in weight distribution
"""

import torch
import torch.nn as nn
from copy import deepcopy


class RTNLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores INT4/INT8 weights
    and dequantizes on the fly during forward pass.
    """
    def __init__(self, weight: torch.Tensor, bias, scale: torch.Tensor, bits: int):
        super().__init__()
        self.bits = bits
        self.register_buffer("weight_q", weight)   # quantized integer weights
        self.register_buffer("scale", scale)        # per-output-channel scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly: INT -> FP16 for compute
        weight_deq = self.weight_q.to(x.dtype) * self.scale.unsqueeze(1)
        return nn.functional.linear(x, weight_deq, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int) -> "RTNLinear":
        W = linear.weight.data.float()   # [out, in]

        # Symmetric per-output-channel quantization
        max_val = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        qmax = 2 ** (bits - 1) - 1
        scale = max_val / qmax           # [out, 1]

        W_q = (W / scale).round().clamp(-qmax - 1, qmax)

        if bits <= 8:
            W_q = W_q.to(torch.int8)
        # For INT4: store as int8, wastes a bit but avoids custom packing
        # In production you'd pack two INT4 into one INT8

        return cls(
            weight=W_q,
            bias=linear.bias,
            scale=scale.squeeze(1).half(),
            bits=bits,
        )


def apply_rtn(model, tokenizer=None, bits: int = 8):
    """
    Replace all nn.Linear layers with RTN-quantized equivalents.
    Skips the LM head (last projection) to preserve output distribution.
    """
    model = deepcopy(model)

    def _replace(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in full_name:
                setattr(module, name, RTNLinear.from_linear(child, bits=bits))
            else:
                _replace(child, full_name)

    _replace(model)
    print(f"  RTN INT{bits}: replaced all Linear layers with {bits}-bit quantized versions")
    return model
