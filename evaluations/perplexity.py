"""
Perplexity Evaluation on WikiText-2
=====================================
Standard benchmark for LLM quantization papers.
PPL measures how "surprised" the model is by the test set.
Lower = better.

We follow the exact same setup as most quant papers:
  - WikiText-2 test split
  - Sliding window with stride = seq_len/2 to handle long sequences
  - seq_len = 2048 (or model max)
"""

import torch
from datasets import load_dataset
import math


def compute_perplexity(model, tokenizer, device: str = "cpu",
                        seq_len: int = 512, stride: int = 256,
                        max_tokens: int = 20000) -> float:
    """
    Compute perplexity on WikiText-2 test set using sliding window.

    Args:
        seq_len:    context window per forward pass
        stride:     sliding window stride (seq_len/2 avoids boundary effects)
        max_tokens: cap total tokens evaluated (for speed)

    Returns:
        perplexity (float) — lower is better
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    # Cap for speed
    if input_ids.size(0) > max_tokens:
        input_ids = input_ids[:max_tokens]

    total_log_likelihood = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for begin_loc in range(0, input_ids.size(0) - seq_len, stride):
            end_loc = begin_loc + seq_len
            ids = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)

            # Labels: same as input, shifted by model internally
            outputs = model(ids, labels=ids)
            log_likelihood = outputs.loss.item()  # cross-entropy (mean over tokens)

            # Count only the tokens not covered by previous window
            n_new_tokens = min(stride, ids.size(1))
            total_log_likelihood += log_likelihood * n_new_tokens
            total_tokens += n_new_tokens

    avg_nll = total_log_likelihood / total_tokens
    return math.exp(avg_nll)
