"""
SmoothQuant: Smooth the Quantization Difficulty from Activations to Weights
=============================================================================
Paper: "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"
       Xiao et al., 2022  https://arxiv.org/abs/2211.10438

Problem it solves:
  - Weight quantization (INT8) is easy — weights are smooth
  - Activation quantization (INT8) is hard — activations have extreme outliers
    (e.g., specific channels with magnitude 100x larger than others)
  - W8A8 is needed for real speedup on hardware (INT8 matmul), but naive A8 kills quality

Key idea: mathematically equivalent reparameterization
    Y = (X * diag(s)^{-1}) * (diag(s) * W^T)
        \_____  ______/     \______  _______/
              smooth X'           smooth W'
  
  where s_j = max(|X_j|)^alpha / max(|W_j|)^{1-alpha}
  
  - Divide activations by s  → reduces activation outliers
  - Multiply weights by s    → weights absorb the difficulty
  - alpha controls migration: 0.5 means equal share

After smoothing, both X' and W' are quantization-friendly for INT8.

Implementation notes:
  - s is absorbed into the preceding LayerNorm (scale the LN weight/bias by 1/s)
  - This makes the transform truly "free" — no extra runtime ops
  - We implement the weight-side absorption here (LN absorption is model-specific)

Pros:  Enables W8A8 quantization with minimal perplexity loss
       Hardware-efficient: INT8 matmul on NVIDIA/Apple hardware
Cons:  Still requires activation calibration
       Slightly worse than GPTQ/AWQ for weight-only quant
"""

import torch
import torch.nn as nn
from copy import deepcopy
from datasets import load_dataset


# ── Calibration: collect per-channel activation max ──────────────────────────

def collect_activation_maxima(model, tokenizer, linear_names: list[str],
                               n_samples: int = 128, seq_len: int = 512,
                               device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    For each Linear layer, collect per-input-channel absolute maximum
    across calibration data. Returns dict: name -> [in_features] tensor.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    maxima: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float().reshape(-1, inp[0].size(-1))
            channel_max = x.abs().max(dim=0).values   # [in_features]
            if name not in maxima:
                maxima[name] = channel_max
            else:
                maxima[name] = torch.maximum(maxima[name], channel_max)
        return hook

    model.to(device)
    for name, module in model.named_modules():
        if name in linear_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        n = 0
        for i in range(0, input_ids.size(0) - seq_len, seq_len):
            model(input_ids[i : i + seq_len].unsqueeze(0).to(device))
            n += 1
            if n >= n_samples:
                break

    for h in hooks:
        h.remove()

    return maxima


# ── SmoothQuant scaling ───────────────────────────────────────────────────────

def compute_smooth_scale(act_max: torch.Tensor, W: torch.Tensor,
                          alpha: float = 0.5) -> torch.Tensor:
    """
    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    W:  [out_features, in_features], we look at per-input-channel max
    """
    w_max = W.abs().max(dim=0).values.clamp(min=1e-8)   # [in_features]
    act_max = act_max.clamp(min=1e-8)
    s = act_max.pow(alpha) / w_max.pow(1 - alpha)
    return s


def smooth_and_quantize(W: torch.Tensor, s: torch.Tensor,
                         bits: int = 8) -> torch.Tensor:
    """
    Apply smooth scale to W, then quantize.
    W' = diag(s) * W^T  ->  W'^T = W * diag(s)  (column-wise multiply)
    i.e., W'[:, j] *= s[j]
    Returns dequantized weight (float16).
    """
    qmax = 2 ** (bits - 1) - 1

    # W: [out, in]. Multiply each input channel by s
    W_smooth = W.float() * s.unsqueeze(0)          # [out, in]

    # Symmetric per-channel quantization
    max_val = W_smooth.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = max_val / qmax

    W_q = (W_smooth / scale).round().clamp(-qmax - 1, qmax)
    W_deq = W_q * scale                             # dequantized [out, in]

    return W_deq.half()


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_smoothquant(model, tokenizer, bits: int = 8,
                       alpha: float = 0.5, n_calib_samples: int = 128):
    """
    Apply SmoothQuant INT{bits} to all Linear layers (except lm_head).

    Note: Full SmoothQuant would also quantize activations (W8A8). Here we
    quantize weights only (W8A32) since runtime activation quantization
    requires custom kernels. The smooth scaling still benefits weight quantization.
    """
    model = deepcopy(model)
    device = next(model.parameters()).device

    target_layers = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in name
    ]

    print(f"  Collecting activation maxima for {len(target_layers)} layers ...")
    act_maxima = collect_activation_maxima(
        model, tokenizer, target_layers,
        n_samples=n_calib_samples, device=device
    )

    print(f"  Applying smooth scaling + INT{bits} weight quantization (alpha={alpha}) ...")
    for name, module in model.named_modules():
        if name not in act_maxima:
            continue

        s = compute_smooth_scale(act_maxima[name].to(device),
                                  module.weight.data.float(), alpha=alpha)
        module.weight.data = smooth_and_quantize(module.weight.data, s, bits=bits)

    return model
