import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set device (Modify here if you want to force CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define matrix sizes
a = 128  # Input batch size
b = 1024  # Large dimension
r = 64  # Smaller rank for LoRA

# Initialize random tensors on the selected device
x = torch.randn(a, b, device=device)
M = torch.randn(b, b, device=device)  # Full-rank matrix
lora_A = torch.randn(b, r, device=device)  # LoRA A (Low-rank)
lora_B = torch.randn(r, b, device=device)  # LoRA B (Low-rank)
dropout = torch.nn.Dropout(p=0.1).to(device)  # Dropout layer
scaling = 0.1
y = torch.randn(a, b, device=device)  # Bias term for 0-LoRA

# Define different inference functions
def base_layer_inference():
    return torch.matmul(x, M)

def lora_inference_0():
    return torch.matmul(torch.matmul(dropout(x), lora_A), lora_B) * scaling + y

def lora_inference_1():
    return torch.matmul(torch.matmul(dropout(x), lora_A), lora_B) * scaling

def lora_inference_2():
    return torch.matmul(torch.matmul(dropout(x), lora_A), lora_B)

def lora_inference_3():
    return torch.matmul(x, lora_A).matmul(lora_B)

# Measure latency for each function
def measure_latency(func, iterations=1000):
    latencies = []
    for _ in range(iterations):
        start = time.time()
        _ = func()
        torch.cuda.synchronize() if device.type == "cuda" else None  # Ensure accurate timing
        latencies.append((time.time() - start) * 1000)  # Convert to ms
    return np.array(latencies)

# Run latency measurements
iterations = 1000
lat_base = measure_latency(base_layer_inference, iterations)
lat_0 = measure_latency(lora_inference_0, iterations)
lat_1 = measure_latency(lora_inference_1, iterations)
lat_2 = measure_latency(lora_inference_2, iterations)
lat_3 = measure_latency(lora_inference_3, iterations)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(lat_base, label=f"Base Layer (Mean: {lat_base.mean():.3f} ms, Std: {lat_base.std():.3f})", alpha=0.8)
plt.plot(lat_0, label=f"0-LoRA Layer (Mean: {lat_0.mean():.3f} ms, Std: {lat_0.std():.3f})", alpha=0.8)
plt.plot(lat_1, label=f"1-LoRA Layer (Mean: {lat_1.mean():.3f} ms, Std: {lat_1.std():.3f})", alpha=0.8)
plt.plot(lat_2, label=f"2-LoRA Layer (Mean: {lat_2.mean():.3f} ms, Std: {lat_2.std():.3f})", alpha=0.8)
plt.plot(lat_3, label=f"3-LoRA Layer (Mean: {lat_3.mean():.3f} ms, Std: {lat_3.std():.3f})", alpha=0.8)

# Customize plot
plt.title("Inference Latency per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)

# Save figure as a high-quality JPG file
timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]
filename = "results/latency_test_1_" + timestamp + ".jpg"
plt.savefig(filename, dpi=300, bbox_inches='tight')

print(f"Plot saved as {filename}")
