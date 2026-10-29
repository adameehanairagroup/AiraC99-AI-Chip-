#!/usr/bin/env python3
"""
============================================================================
AIRA CORE - MODEL WEIGHT EXPORTER (HuggingFace / PyTorch -> Aira Binary Format)

Exports Transformer parameters into a contiguous zero-copy memory-mappable
binary file for ultra-fast execution on low-end hardware.

Architect: Er. Adam Eehan (Aira Group of Technology)
============================================================================
"""

import struct
import sys
import numpy as np

def export_model_to_bin(config, state_dict, output_path):
    print(f"[*] Packaging Model Weights to Aira Binary Format: {output_path}")

    # Header parameters:
    # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]

    with open(output_path, "wb") as f:
        # Write 7 config integers
        header = struct.pack(
            "7i",
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len
        )
        f.write(header)

        # Helper to serialize float32 array
        def serialize_tensor(tensor):
            d = tensor.detach().cpu().numpy().astype(np.float32)
            f.write(d.tobytes())

        print("[+] Writing Token Embedding Table...")
        serialize_tensor(state_dict["tok_embeddings.weight"])

        print("[+] Writing Layer Weights (Attention, FFN, RMSNorm)...")
        for l in range(n_layers):
            serialize_tensor(state_dict[f"layers.{l}.attention_norm.weight"])
            serialize_tensor(state_dict[f"layers.{l}.attention.wq.weight"])
            serialize_tensor(state_dict[f"layers.{l}.attention.wk.weight"])
            serialize_tensor(state_dict[f"layers.{l}.attention.wv.weight"])
            serialize_tensor(state_dict[f"layers.{l}.attention.wo.weight"])
            serialize_tensor(state_dict[f"layers.{l}.ffn_norm.weight"])
            serialize_tensor(state_dict[f"layers.{l}.feed_forward.w1.weight"])
            serialize_tensor(state_dict[f"layers.{l}.feed_forward.w2.weight"])
            serialize_tensor(state_dict[f"layers.{l}.feed_forward.w3.weight"])

        print("[+] Writing Final Normalization & Classifier...")
        serialize_tensor(state_dict["norm.weight"])

        # RoPE freqs
        head_size = dim // n_heads
        freqs = 1.0 / (10000.0 ** (np.arange(0, head_size, 2)[: (head_size // 2)] / head_size))
        freqs_cis = np.outer(np.arange(seq_len), freqs).astype(np.float32)
        f.write(freqs_cis.tobytes())

        # Unshared classifier head if present
        if "output.weight" in state_dict:
            serialize_tensor(state_dict["output.weight"])

    print(f"[SUCCESS] Export complete! Model ready for Aira Core.")

if __name__ == "__main__":
    print("Aira Core Model Exporter v1.0.0")
