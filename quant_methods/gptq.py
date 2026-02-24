"""
GPTQ: Post-Training Quantization via Hessian-Based Error Compensation
======================================================================
Paper: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
       Frantar et al., 2022  https://arxiv.org/abs/2210.17323

Core idea: RTN treats all weights as equally important. GPTQ uses second-order
information (the Hessian of the layer's output w.r.t. weights) to compensate
for quantization error in already-quantized weights.

Algorithm per layer:
    1. Collect calibration activations X  [batch, seq, in_features]
    2. Compute Hessian  H = 2 * X^T X     [in, in]
    3. Apply Cholesky decomposition for numerical stability
    4. For each column j (weight dimension):
       a. Quantize column j: w_q = quant(w_j)
       b. Compute error: delta = w_q - w_j
       c. Propagate error to remaining columns using H:
          W[:, j+1:] -= delta * (H^{-1}[j, j+1:] / H^{-1}[j, j])

This is the "Optimal Brain Surgeon" update applied column-by-column, O(d^2) per layer.

Pros: Best perplexity among pure PTQ methods at INT4
Cons: Requires calibration data (~128 samples), Cholesky can be unstable for small H
"""

import torch
import torch.nn as nn
from copy import deepcopy
from datasets import load_dataset


# ── Quantization helpers ─────────────────────────────────────────────────────

def quantize_tensor(W: torch.Tensor, bits: int, per_channel: bool = True) -> tuple:
    """Symmetric linear quantization. Returns (W_q, scale)."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        max_val = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    else:
        max_val = W.abs().max().clamp(min=1e-8)
    scale = max_val / qmax
    W_q = (W / scale).round().clamp(-qmax - 1, qmax)
    return W_q, scale


def dequantize(W_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return W_q * scale


# ── Calibration data ─────────────────────────────────────────────────────────

def get_calibration_data(tokenizer, n_samples: int = 128, seq_len: int = 512):
    """Load WikiText-2 train split as calibration samples."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    samples = []
    for i in range(0, min(n_samples * seq_len, input_ids.size(0) - seq_len), seq_len):
        samples.append(input_ids[i : i + seq_len].unsqueeze(0))
        if len(samples) >= n_samples:
            break
    return samples


# ── GPTQ core for one Linear layer ───────────────────────────────────────────

class GPTQQuantizer:
    def __init__(self, layer: nn.Linear, bits: int = 4, block_size: int = 128):
        self.layer = layer
        self.bits = bits
        self.block_size = block_size  # columns processed at once
        self.H = None                 # accumulated Hessian
        self.n_samples = 0

    def add_batch(self, x: torch.Tensor):
        """Accumulate Hessian from calibration batch. x: [batch*seq, in_features]"""
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        x = x.float()
        if self.H is None:
            self.H = torch.zeros(x.size(1), x.size(1), dtype=torch.float32,
                                 device=x.device)
        self.H += x.T @ x
        self.n_samples += x.size(0)

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run GPTQ on the stored weight matrix.
        Returns (W_q, scale) per output channel.
        """
        W = self.layer.weight.data.float().clone()  # [out, in]
        out_feat, in_feat = W.shape

        # Normalize Hessian
        H = self.H.clone()
        H /= self.n_samples
        H += 1e-8 * torch.eye(in_feat, device=H.device)  # damping for stability

        # Cholesky decomposition: H = L L^T, work with H^{-1} via Cholesky solve
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            print("  Warning: Cholesky failed, falling back to RTN for this layer")
            W_q, scale = quantize_tensor(W, self.bits)
            return W_q.to(torch.int8), scale

        qmax = 2 ** (self.bits - 1) - 1

        # Per-output-channel scale (compute once from full W)
        max_val = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = max_val / qmax  # [out, 1]

        W_q = torch.zeros_like(W, dtype=torch.float32)

        # Process in blocks of columns for memory efficiency
        for col_start in range(0, in_feat, self.block_size):
            col_end = min(col_start + self.block_size, in_feat)
            W_block = W[:, col_start:col_end].clone()

            for j in range(col_end - col_start):
                col = col_start + j

                # 1. Quantize this column
                w_col = W[:, col]
                w_q = (w_col / scale.squeeze(1)).round().clamp(-qmax - 1, qmax)
                W_q[:, col] = w_q

                # 2. Error = quantized - original (in float space)
                err = (w_q * scale.squeeze(1)) - w_col  # [out]

                # 3. Propagate error to remaining columns in block
                if col + 1 < in_feat:
                    h_row = H_inv[col, col + 1 :]  # [remaining]
                    h_diag = H_inv[col, col].clamp(min=1e-8)
                    W[:, col + 1 :] -= torch.outer(err, h_row / h_diag)

        return W_q.to(torch.int8), scale.squeeze(1).half()


# ── Hook-based calibration runner ────────────────────────────────────────────

class GPTQRunner:
    def __init__(self, model, bits: int = 4):
        self.model = model
        self.bits = bits
        self.quantizers: dict[str, GPTQQuantizer] = {}
        self.hooks = []

    def register_hooks(self):
        """Attach forward hooks to all Linear layers to capture activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                q = GPTQQuantizer(module, bits=self.bits)
                self.quantizers[name] = q

                def make_hook(quantizer):
                    def hook(_, inp, __):
                        quantizer.add_batch(inp[0].detach())
                    return hook

                self.hooks.append(module.register_forward_hook(make_hook(q)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def run_calibration(self, samples, device):
        self.model.to(device)
        with torch.no_grad():
            for ids in samples:
                self.model(ids.to(device))

    def apply_quantization(self):
        """Replace weights with GPTQ-quantized versions."""
        for name, module in self.model.named_modules():
            if name in self.quantizers:
                q = self.quantizers[name]
                W_q, scale = q.quantize()
                # Store quantized weight + scale back into the layer
                module.weight.data = (W_q.float() * scale.unsqueeze(1))
                module.weight.data = module.weight.data.half()


def apply_gptq(model, tokenizer, bits: int = 4, n_calib_samples: int = 128):
    model = deepcopy(model)
    device = next(model.parameters()).device

    print(f"  Collecting {n_calib_samples} calibration samples ...")
    samples = get_calibration_data(tokenizer, n_samples=n_calib_samples)

    runner = GPTQRunner(model, bits=bits)
    runner.register_hooks()

    print(f"  Running calibration forward passes ...")
    runner.run_calibration(samples, device)
    runner.remove_hooks()

    print(f"  Applying GPTQ INT{bits} quantization ...")
    runner.apply_quantization()

    return model
