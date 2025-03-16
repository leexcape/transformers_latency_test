"""
Testing latency for single layer of torch.nn.linear. Input x, weights and bias are recovered from the PEFT, specifically
, the lora. 96 samples of the set (x, weights and bias) are grabbed from the PEFT lora breakpoint, and one of 96 samples
are drawn and used to run 1 point out of 1000 in total. Further, the GPU cache are cleared after each run.
"""

import torch.nn as nn
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import random

matplotlib.use("Agg")  # Avoids GUI issues

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Path to weight files
weights_dir = "data/weights/"

# Get all available .pth files in the directory
weight_files = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith(".pth")]

if not weight_files:
    raise FileNotFoundError("No .pth weight files found in directory: " + weights_dir)

print(f"Found {len(weight_files)} weight files.")

# Function to load a random checkpoint
def load_random_checkpoint():
    chosen_file = random.choice(weight_files)
    checkpoint = torch.load(chosen_file, map_location="cpu")  # Load on CPU first

    # Extract dimensions
    in_features = checkpoint["base_weight"].shape[1]
    out_features = checkpoint["base_weight"].shape[0]
    lora_features = checkpoint["lora_A_weight"].shape[0]

    # Create new linear layers
    base_layer = nn.Linear(in_features, out_features, bias=checkpoint["base_bias"] is not None)
    lora_A_layer = nn.Linear(in_features, lora_features, bias=checkpoint["lora_A_bias"] is not None)
    lora_B_layer = nn.Linear(lora_features, out_features, bias=checkpoint["lora_B_bias"] is not None)

    # Load weights and biases
    base_layer.weight.data = checkpoint["base_weight"]
    if checkpoint["base_bias"] is not None:
        base_layer.bias.data = checkpoint["base_bias"]

    lora_A_layer.weight.data = checkpoint["lora_A_weight"]
    if checkpoint["lora_A_bias"] is not None:
        lora_A_layer.bias.data = checkpoint["lora_A_bias"]

    lora_B_layer.weight.data = checkpoint["lora_B_weight"]
    if checkpoint["lora_B_bias"] is not None:
        lora_B_layer.bias.data = checkpoint["lora_B_bias"]

    # Load input x
    x = checkpoint["layer_input"]

    return base_layer, lora_A_layer, lora_B_layer, x

# Function to measure latency with GPU cache clearing
def measure_latency(num_runs=1000):
    latencies = {
        "base": [],
        "lora_0": [],
        "lora_1": [],
        "lora_2": [],
        "lora_3": []
    }

    dropout = nn.Dropout(p=0.1)  # Simulating dropout
    scaling = 0.1  # Example scaling factor

    for _ in range(num_runs):
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        # Load a random checkpoint
        base_layer, lora_A_layer, lora_B_layer, x = load_random_checkpoint()

        # Move model and input to GPU
        base_layer = base_layer.to(device)
        lora_A_layer = lora_A_layer.to(device)
        lora_B_layer = lora_B_layer.to(device)
        x = x.to(device)

        y = base_layer(x)

        # Ensure GPU synchronization before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure Base Layer Latency
        start_time = time.time()
        _ = base_layer(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies["base"].append((time.time() - start_time) * 1000)

        # Measure LoRA 0 Latency
        start_time = time.time()
        _ = lora_B_layer(lora_A_layer(dropout(x))) * scaling + y
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies["lora_0"].append((time.time() - start_time) * 1000)

        # Measure LoRA 1 Latency
        start_time = time.time()
        _ = lora_B_layer(lora_A_layer(dropout(x))) * scaling
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies["lora_1"].append((time.time() - start_time) * 1000)

        # Measure LoRA 2 Latency
        start_time = time.time()
        _ = lora_B_layer(lora_A_layer(dropout(x)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies["lora_2"].append((time.time() - start_time) * 1000)

        # Measure LoRA 3 Latency
        start_time = time.time()
        _ = lora_B_layer(lora_A_layer(x))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies["lora_3"].append((time.time() - start_time) * 1000)

    return {key: np.array(value) for key, value in latencies.items()}

# Measure latency with random weight selection for 1000 runs
latency_results = measure_latency()

# Compute statistics
stats = {key: (np.mean(values), np.std(values)) for key, values in latency_results.items()}

# Print latency summary
for key, (mean, std) in stats.items():
    print(f"{key.replace('_', '-').title()} Layer: Mean={mean:.3f} ms, Std={std:.3f}")

# Plot results
plt.figure(figsize=(10, 5))
for key, values in latency_results.items():
    plt.plot(values, label=f'{key.replace("_", "-").title()} (Mean: {stats[key][0]:.3f} ms, Std: {stats[key][1]:.3f})', alpha=0.7)

plt.xlabel("Iteration")
plt.ylabel("Latency (ms)")
plt.title("Inference Latency per Iteration (Randomized Weights, Cache Cleared)")
plt.legend()
plt.savefig("latency_plot_cache_cleared.png")  # ✅ Save figure as an image

print("Latency analysis completed and saved!")