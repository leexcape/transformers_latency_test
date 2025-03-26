import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load the JSON file
with open("/home/cc/MyWorkspace/transformers_latency_test/results/id_record_20250321_18_47_34_221.json", "r") as f:  # Replace with actual filename
    data = json.load(f)

# Identify keys for "base" and "lora"
base_key = next(k for k in data.keys() if "base" in k)
lora_key = next(k for k in data.keys() if "lora" in k)

# Extract the time series data
base_times = data[base_key]
lora_times = data[lora_key]

# Generate x-axis (run indices)
runs_base = list(range(len(base_times)))
runs_lora = list(range(len(lora_times)))

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(runs_base[10:-1], base_times[10:-1], label="Base", marker="o", linestyle="-", alpha=0.7)
plt.plot(runs_lora[10:-1], lora_times[10:-1], label="LoRA", marker="s", linestyle="--", alpha=0.7)

# Labels and title
plt.xlabel("Runs")
plt.ylabel("Time (s)")
plt.title("Comparison of Base vs. LoRA Time Traces")
plt.legend()
plt.grid(True)

# Save plot
timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]
filename = "results/id_record_time_" + timestamp + ".jpg"
plt.savefig(filename)
plt.close()
