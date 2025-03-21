"""
Testing latency for single layer of torch.nn.linear. Input x, weights and bias are recovered from the PEFT, specifically
, the lora. 1 sample of the set (x, weights and bias) is grabbed from the PEFT lora breakpoint, and this sample are
reused in each of the 1000 runs in total
"""

import torch.nn as nn
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from datetime import datetime

matplotlib.use("Agg")  # Requires X11 forwarding
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("using device: ", device)

# Load saved matrices
checkpoint = torch.load("/home/cc/MyWorkspace/transformers_latency_test/data/weights/lora_matrices_20250321_17_32_19_794.pth")

# Reconstruct layers
in_features = checkpoint["base_weight"].shape[1]
out_features = checkpoint["base_weight"].shape[0]
lora_features = checkpoint["lora_A_weight"].shape[0]

# Create new linear layers
base_layer = nn.Linear(in_features, out_features, bias=checkpoint["base_bias"] is not None).to(device)
lora_A_layer = nn.Linear(in_features, lora_features, bias=checkpoint["lora_A_bias"] is not None).to(device)
lora_B_layer = nn.Linear(lora_features, out_features, bias=checkpoint["lora_B_bias"] is not None).to(device)

# Load weights and biases
base_layer.weight.data = checkpoint["base_weight"].to(device)
if checkpoint["base_bias"] is not None:
    base_layer.bias.data = checkpoint["base_bias"].to(device)

lora_A_layer.weight.data = checkpoint["lora_A_weight"].to(device)
if checkpoint["lora_A_bias"] is not None:
    lora_A_layer.bias.data = checkpoint["lora_A_bias"].to(device)

lora_B_layer.weight.data = checkpoint["lora_B_weight"].to(device)
if checkpoint["lora_B_bias"] is not None:
    lora_A_layer.bias.data = checkpoint["lora_B_bias"].to(device)

print("Matrices loaded successfully!")

# function to measure the layer-wise latency
def measure_latency(func, num_runs=1000):

    # Warm-up runs
    for _ in range(10):
        _ = func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []

    # Measure latency for each run
    for _ in range(num_runs):
        start_time = time.time()
        _ = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    return np.array(latencies)

# Example input size
batch_size, seq_len, dim, rank = 128, 197, 768, 16  # Modify as needed

# Define Layers
dropout = torch.nn.Dropout(p=0.1).to(device)  # Simulating dropout
scaling = 0.1  # Example scaling factor

# Load input value
x = checkpoint["layer_input"].to(device)
y = base_layer(x)


# Function to measure latency of base layer
def base_layer_inference():
    return base_layer(x)


# Function to measure latency of LoRA operation
def lora_inference_0():
    return lora_B_layer(lora_A_layer(dropout(x))) * scaling + y


def lora_inference_1():
    return lora_B_layer(lora_A_layer(dropout(x))) * scaling


def lora_inference_2():
    return lora_B_layer(lora_A_layer(dropout(x)))


def lora_inference_3():
    return lora_B_layer(lora_A_layer(x))


# Measure latency
latencies_base = measure_latency(base_layer_inference)
latencies_lora_0 = measure_latency(lora_inference_0)
latencies_lora_1 = measure_latency(lora_inference_1)
latencies_lora_2 = measure_latency(lora_inference_2)
latencies_lora_3 = measure_latency(lora_inference_3)

# Compute statistics
mean_base, std_base = np.mean(latencies_base), np.std(latencies_base)
mean_lora_0, std_lora_0 = np.mean(latencies_lora_0), np.std(latencies_lora_0)
mean_lora_1, std_lora_1 = np.mean(latencies_lora_1), np.std(latencies_lora_1)
mean_lora_2, std_lora_2 = np.mean(latencies_lora_2), np.std(latencies_lora_2)
mean_lora_3, std_lora_3 = np.mean(latencies_lora_3), np.std(latencies_lora_3)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(latencies_base, label=f'Base Layer (Mean: {mean_base:.3f} ms, Std: {std_base:.3f})', alpha=0.7)
plt.plot(latencies_lora_0, label=f'0-LoRA Layer (Mean: {mean_lora_0:.3f} ms, Std: {std_lora_0:.3f})', alpha=0.7)
plt.plot(latencies_lora_1, label=f'1-LoRA Layer (Mean: {mean_lora_1:.3f} ms, Std: {std_lora_1:.3f})', alpha=0.7)
plt.plot(latencies_lora_2, label=f'2-LoRA Layer (Mean: {mean_lora_2:.3f} ms, Std: {std_lora_2:.3f})', alpha=0.7)
plt.plot(latencies_lora_3, label=f'3-LoRA Layer (Mean: {mean_lora_3:.3f} ms, Std: {std_lora_3:.3f})', alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Latency (ms)")
plt.title("Inference Latency per Iteration")
plt.legend()

# Save figure as a high-quality JPG file
timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]
filename = "results/latency_test_3_" + timestamp + ".jpg"
plt.savefig(filename, dpi=300, bbox_inches='tight')

print(f"Plot saved as {filename}")

