"""
AWQ: Activation-Aware Weight Quantization
==========================================
Paper: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
       Lin et al., 2023  https://arxiv.org/abs/2306.00978

Core insight: Not all weight channels are equally important.
Channels corresponding to large activation magnitudes matter more —
getting them wrong causes disproportionate output error.

Strategy:
  1. Collect per-channel activation magnitudes s_x = mean(|X|) over calibration data
  2. For important channels (large s_x), scale UP the weights before quantization
     so the quantizer allocates more resolution to them
  3. Scale DOWN the activations correspondingly (absorbed into previous layer's bias/norm)
     so the output is unchanged: W_q(W * s) * (X / s) ≈ W_q(W) * X

The scaling factor alpha controls the trade-off:
  scale = s_x^alpha,  alpha in [0, 1]  (tuned per layer via grid search)

Key difference from GPTQ:
  - GPTQ compensates error AFTER quantization using Hessian
  - AWQ prevents error BEFORE quantization by protecting important channels

Pros:  Excellent INT4 perplexity, hardware-friendly (no special dequant kernels needed)
       Works well with group quantization (group_size=128)
Cons:  Requires activation statistics, alpha search adds overhead
"""

import torch
import torch.nn as nn
from copy import deepcopy
from datasets import load_dataset


# ── Calibration helpers ───────────────────────────────────────────────────────

def get_calibration_activations(model, tokenizer, layer_names: list[str],
                                 n_samples: int = 128, seq_len: int = 512,
                                 device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    Run calibration forward passes and collect per-channel activation magnitudes.
    Returns dict: layer_name -> mean |activation| per input channel [in_features]
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    act_stats: dict[str, list] = {n: [] for n in layer_names}
    hooks = []

    def make_hook(name):
        def hook(_, inp, __):
            x = inp[0].detach().float()           # [batch, seq, in]
            x = x.reshape(-1, x.size(-1))         # [batch*seq, in]
            act_stats[name].append(x.abs().mean(dim=0))  # [in]
        return hook

    model.to(device)
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_samples * seq_len, input_ids.size(0) - seq_len), seq_len):
            ids = input_ids[i : i + seq_len].unsqueeze(0).to(device)
            model(ids)
            if i // seq_len >= n_samples:
                break

    for h in hooks:
        h.remove()

    # Average across calibration batches
    return {name: torch.stack(stats).mean(0) for name, stats in act_stats.items()
            if stats}


# ── AWQ scale search ─────────────────────────────────────────────────────────

def find_best_alpha(W: torch.Tensor, s_x: torch.Tensor, bits: int,
                    alphas: list[float] = None) -> float:
    """
    Grid search over alpha to minimize quantization error:
        min_alpha  ||W_q(W * diag(s_x^alpha)) * diag(s_x^{-alpha}) - W||_F
    """
    if alphas is None:
        alphas = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9

    qmax = 2 ** (bits - 1) - 1
    best_alpha, best_err = 0.5, float("inf")

    for alpha in alphas:
        scale = s_x.pow(alpha).clamp(min=1e-4)  # [in_features]

        # Scale weights
        W_scaled = W * scale.unsqueeze(0)  # [out, in]

        # Quantize scaled weights (per-channel)
        max_val = W_scaled.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        q_scale = max_val / qmax
        W_q = (W_scaled / q_scale).round().clamp(-qmax - 1, qmax) * q_scale

        # Undo the scale (as if activations were divided by scale)
        W_deq = W_q / scale.unsqueeze(0)

        err = (W_deq - W).pow(2).mean().item()
        if err < best_err:
            best_err = err
            best_alpha = alpha

    return best_alpha


# ── Weight quantization with AWQ scale ───────────────────────────────────────

def awq_quantize_linear(W: torch.Tensor, s_x: torch.Tensor,
                         bits: int, alpha: float) -> torch.Tensor:
    """
    Quantize W using the AWQ-scaled scheme.
    The activation scale s_x^alpha is absorbed into the previous layer,
    so here we just quantize the pre-scaled weight and return the dequantized result.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = s_x.pow(alpha).clamp(min=1e-4)

    W_scaled = W * scale.unsqueeze(0)

    max_val = W_scaled.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    q_scale = max_val / qmax

    W_q = (W_scaled / q_scale).round().clamp(-qmax - 1, qmax)
    W_deq = (W_q * q_scale) / scale.unsqueeze(0)

    return W_deq.half()


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_awq(model, tokenizer, bits: int = 4, n_calib_samples: int = 128):
    model = deepcopy(model)
    device = next(model.parameters()).device

    # Identify all Linear layers (except lm_head)
    target_layers = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in name
    ]

    print(f"  Collecting activation statistics for {len(target_layers)} layers ...")
    act_stats = get_calibration_activations(
        model, tokenizer, target_layers,
        n_samples=n_calib_samples, device=device
    )

    print(f"  Searching optimal alpha and applying AWQ INT{bits} ...")
    for name, module in model.named_modules():
        if name not in act_stats:
            continue

        s_x = act_stats[name].to(device)
        W = module.weight.data.float()

        alpha = find_best_alpha(W, s_x, bits=bits)
        module.weight.data = awq_quantize_linear(W, s_x, bits=bits, alpha=alpha)

    return model
