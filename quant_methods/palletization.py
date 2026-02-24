"""
Palletization (Lookup Table / Codebook Quantization)
=====================================================
Apple CoreML documentation: https://coremltools.readme.io/docs/palettization-overview

Palletization is fundamentally different from linear quantization:

  Linear quantization:  value = round(W / scale) * scale   (uniform grid)
  Palletization:        value = LUT[argmin_k ||W - LUT[k]||]  (learned codebook)

Instead of mapping weights to a uniform grid, palletization learns a small
set of representative values (the "palette" or LUT) and replaces each weight
with an index into that LUT. With 4-bit palletization:
  - LUT size: 2^4 = 16 entries
  - Each weight stores a 4-bit index instead of 16-bit float
  - Compression ratio: 16-bit / 4-bit = 4x

How CoreML uses it at runtime:
  - Weights stored as INT4 indices in ANE/GPU memory
  - LUT (16 floats per group) stored separately
  - During inference: gather(LUT, indices) to reconstruct weight slice
  - ANE has native support for this gather-and-compute pattern

LUT optimization — two strategies:
  1. K-means (what we implement here): LUT entries = k-means centroids
     Pros: Adapts to actual weight distribution, non-uniform spacing
     Cons: Requires calibration, slightly slower to encode
  
  2. Linear (uniform): LUT = evenly spaced in [min, max]
     Pros: No calibration needed, fast
     Cons: Worse quality when distribution is non-uniform (e.g., bell-shaped)

Comparison with INT4 linear quantization at same 4-bit budget:
  - Uniform INT4: 16 uniformly-spaced values
  - Palletization: 16 k-means-optimized values
  - Palletization wins when weight distribution has clusters
    (common in attention weight matrices)

Group palletization:
  We support per-group LUTs (group_size weights share one LUT), which gives
  finer granularity similar to GPTQ's group quantization.

This implementation simulates CoreML palletization in PyTorch:
  - Same algorithm (k-means LUT + index storage)
  - No ANE kernel — dequantize on-the-fly in forward pass
  - Run `coremltools.optimize.torch.palettization` on Mac for real ANE speedup
"""

import torch
import torch.nn as nn
from copy import deepcopy


# ── K-means LUT computation ───────────────────────────────────────────────────

def kmeans_lut(weights: torch.Tensor, n_entries: int,
               n_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a k-means LUT for a 1D weight tensor.

    Args:
        weights:   1D float tensor of weight values
        n_entries: number of LUT entries (e.g., 16 for 4-bit)
        n_iter:    k-means iterations

    Returns:
        lut:     [n_entries] float tensor — the palette values
        indices: [len(weights)] int tensor — index of nearest LUT entry per weight
    """
    w = weights.float()

    # Initialize centroids using quantile-based spacing
    # Better than random init: covers the distribution from the start
    quantiles = torch.linspace(0, 1, n_entries, device=w.device)
    sorted_w = w.sort().values
    idx = (quantiles * (len(sorted_w) - 1)).long().clamp(0, len(sorted_w) - 1)
    centroids = sorted_w[idx].clone()

    for _ in range(n_iter):
        # Assignment: each weight -> nearest centroid
        dists = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # [N, K]
        assignments = dists.argmin(dim=1)                          # [N]

        # Update centroids: mean of assigned weights
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_entries):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = w[mask].mean()
            else:
                new_centroids[k] = centroids[k]  # keep old if empty

        # Check convergence
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids

    # Final assignment
    dists = (w.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1)

    return centroids, indices


# ── Palletized Linear layer ───────────────────────────────────────────────────

class PalletizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using palletization.

    Stores:
      - weight_indices: [out, in] INT8 (actually 4-bit logically, stored as int8)
      - lut:            [n_groups, n_entries] float16
    
    On forward: reconstruct weights via LUT gather, then matmul.
    """

    def __init__(self, indices: torch.Tensor, lut: torch.Tensor,
                 bias, group_size: int, out_features: int, in_features: int):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.n_entries = lut.size(1)

        self.register_buffer("weight_indices", indices)   # [out*in] int8
        self.register_buffer("lut", lut)                  # [n_groups, n_entries]
        self.bias = bias

    def _dequantize(self) -> torch.Tensor:
        """Reconstruct float weight matrix from indices and LUT."""
        total = self.out_features * self.in_features
        n_groups = (total + self.group_size - 1) // self.group_size

        # Pad indices to be divisible by group_size
        padded = self.weight_indices.long()
        if padded.size(0) < n_groups * self.group_size:
            pad_size = n_groups * self.group_size - padded.size(0)
            padded = torch.cat([padded, torch.zeros(pad_size, dtype=torch.long,
                                                     device=padded.device)])

        indices_2d = padded.view(n_groups, self.group_size)  # [n_groups, group_size]

        # Gather LUT values: for each group, look up the entry for each index
        # lut: [n_groups, n_entries]  indices_2d: [n_groups, group_size]
        W_flat = self.lut.gather(1, indices_2d)               # [n_groups, group_size]
        W_flat = W_flat.view(-1)[:total]                       # flatten and trim padding

        return W_flat.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._dequantize().to(x.dtype)
        return nn.functional.linear(x, W, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4,
                    group_size: int = 128) -> "PalletizedLinear":
        n_entries = 2 ** bits
        W = linear.weight.data.float()   # [out, in]
        out_feat, in_feat = W.shape
        total = out_feat * in_feat

        W_flat = W.reshape(-1)  # [out*in]
        n_groups = (total + group_size - 1) // group_size

        # Pad to group_size multiple
        pad_size = n_groups * group_size - total
        if pad_size > 0:
            W_flat_padded = torch.cat([W_flat, torch.zeros(pad_size, device=W.device)])
        else:
            W_flat_padded = W_flat

        W_groups = W_flat_padded.view(n_groups, group_size)  # [n_groups, group_size]

        all_luts = []
        all_indices = []

        for g in range(n_groups):
            group_weights = W_groups[g]
            lut, indices = kmeans_lut(group_weights, n_entries)
            all_luts.append(lut.half())
            all_indices.append(indices.to(torch.int8))

        lut_tensor = torch.stack(all_luts)               # [n_groups, n_entries]
        idx_tensor = torch.cat(all_indices)[:total]      # [total] trim padding
        idx_tensor = idx_tensor.to(torch.int8)

        return cls(
            indices=idx_tensor,
            lut=lut_tensor,
            bias=linear.bias,
            group_size=group_size,
            out_features=out_feat,
            in_features=in_feat,
        )

    def extra_repr(self) -> str:
        bits = int(torch.tensor(self.n_entries).log2().item())
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={bits}, group_size={self.group_size}, "
                f"lut_entries={self.n_entries}")


# ── Also provide: uniform LUT baseline (non-k-means) ─────────────────────────

def uniform_lut(weights: torch.Tensor, n_entries: int):
    """Uniform spacing between min and max — baseline vs k-means."""
    w_min, w_max = weights.min(), weights.max()
    lut = torch.linspace(w_min.item(), w_max.item(), n_entries, device=weights.device)
    dists = (weights.unsqueeze(1) - lut.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1)
    return lut, indices


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_palletization(model, tokenizer=None, bits: int = 4,
                         group_size: int = 128, use_kmeans: bool = True):
    """
    Replace all Linear layers with palletized equivalents.

    Args:
        bits:       Number of bits for LUT index (4 → 16-entry LUT)
        group_size: Number of weights sharing one LUT (smaller = better quality, more overhead)
        use_kmeans: True for k-means LUT (better), False for uniform LUT (faster)

    CoreML equivalent:
        import coremltools as ct
        from coremltools.optimize.torch.palettization import (
            DKMPalettizer, DKMPalettizerConfig
        )
        config = DKMPalettizerConfig.from_dict({"global_config": {
            "n_bits": 4, "granularity": "per_grouped_channel", "group_size": 128
        }})
        palettizer = DKMPalettizer(model, config)
        palettizer.prepare()
        # ... fine-tune or calibrate ...
        palettizer.finalize()
    """
    model = deepcopy(model)

    replaced = 0
    lut_method = "k-means" if use_kmeans else "uniform"

    def _replace(module, prefix=""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in full_name:
                setattr(module, name,
                        PalletizedLinear.from_linear(child, bits=bits,
                                                      group_size=group_size))
                replaced += 1
            else:
                _replace(child, full_name)

    print(f"  Building {bits}-bit palletization ({lut_method} LUT, group_size={group_size}) ...")
    _replace(model)
    print(f"  Palletized {replaced} Linear layers | LUT size: {2**bits} entries per group")

    return model


# ── Utility: compare k-means vs uniform LUT quality ──────────────────────────

def compare_lut_methods(W: torch.Tensor, bits: int = 4) -> dict:
    """
    Compare k-means vs uniform LUT reconstruction MSE on a weight tensor.
    Useful for understanding when k-means palletization outperforms linear quant.
    """
    W_flat = W.reshape(-1).float()
    n_entries = 2 ** bits

    # K-means LUT
    km_lut, km_idx = kmeans_lut(W_flat, n_entries)
    km_recon = km_lut[km_idx]
    km_mse = (km_recon - W_flat).pow(2).mean().item()

    # Uniform LUT
    uni_lut, uni_idx = uniform_lut(W_flat, n_entries)
    uni_recon = uni_lut[uni_idx]
    uni_mse = (uni_recon - W_flat).pow(2).mean().item()

    # Linear INT4 (RTN) for reference
    qmax = 2 ** (bits - 1) - 1
    scale = W_flat.abs().max() / qmax
    rtn_q = (W_flat / scale).round().clamp(-qmax - 1, qmax) * scale
    rtn_mse = (rtn_q - W_flat).pow(2).mean().item()

    return {
        "kmeans_palletization_mse": km_mse,
        "uniform_palletization_mse": uni_mse,
        "rtn_int4_mse": rtn_mse,
        "kmeans_vs_rtn_improvement": rtn_mse / km_mse,  # >1 means kmeans wins
    }
