import torch
import torch.nn as nn
import time
import pandas as pd
import os
import numpy as np  # 用于 NaN 处理
import time
import glob
import inspect

save_path = "vit_layer_data"
num_vit_layers = 12  # 假设 ViT 共有 12 层
num_linear_layers = 10  # 每个 ViT 层有 10 个 Linear 层

# 初始化文件的函数
def log_vit_layer_new_file():
    return None
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    latency_file_prefix = timestamp + "-LayerLatency.csv"
    latency_file = os.path.join(save_path, latency_file_prefix)
    size_file_prefix = timestamp + "-LayerSize.csv"
    size_file = os.path.join(save_path, size_file_prefix)
    latency_df = pd.DataFrame(np.nan, index=range(num_vit_layers), columns=range(num_linear_layers))
    size_df = pd.DataFrame(np.nan, index=range(num_vit_layers), columns=range(num_linear_layers))
    latency_df.to_csv(latency_file)
    size_df.to_csv(size_file)

# 记录数据的函数（支持多个 ViT 层）
def log_vit_layer_data(index, start_time, end_time, input_tensor, output_tensor):
    # Get stack
    return None
    matching_paths = [
        "/home/cc/MyWorkspace/lora_test/",
        "/home/cc/miniconda3/envs/transformers_env/lib/python3.10/site-packages/transformers"
    ]
    stack = inspect.stack()
    relevant_frames = [frame for frame in stack if any(path in frame.filename for path in matching_paths)]

    if index == 0 and "value" in relevant_frames[2][4][0]:
        index = 3
    if index == 1 and "value" in relevant_frames[2][4][0]:
        index = 4

    # 计算时延
    latency = end_time - start_time
    input_size = tuple(input_tensor.shape)
    output_size = tuple(output_tensor.shape)

    # 确保目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 定义文件路径
    latency_file_pattern = os.path.join(save_path, f"*-LayerLatency.csv")
    latency_file_list = glob.glob(latency_file_pattern)
    size_file_pattern = os.path.join(save_path, f"*-LayerSize.csv")
    size_file_list = glob.glob(size_file_pattern)
    if not latency_file_list:
        print('Error: There is no file')
    if not size_file_list:
        print('Error: There is no file')
    latency_file = max(latency_file_list, key=os.path.getctime)
    size_file = max(size_file_list, key=os.path.getctime)

    # 读取已有数据（如果文件存在）
    latency_df = pd.read_csv(latency_file, index_col=0)
    latency_df.columns = latency_df.columns.astype(int)
    latency_df.index = latency_df.index.astype(int)

    size_df = pd.read_csv(size_file, index_col=0, dtype=str)  # 由于是 tuple，需要存为字符串
    size_df.columns = size_df.columns.astype(int)
    size_df.index = size_df.index.astype(int)

    # **查找第一个 NaN 的 vit_layer_index**
    available_layer = None
    for vit_layer_index in range(num_vit_layers):
        if pd.isna(latency_df.loc[vit_layer_index, index]):  # 找到第一个 NaN
            available_layer = vit_layer_index
            break

    if available_layer is None:
        print(f"Warning: No available slot for index {index}, skipping update.")
        return  # 如果所有 ViT 层的 `index` 位置都填充了，就跳过

    # **更新数据**
    latency_df.loc[available_layer, index] = latency
    size_df.loc[available_layer, index] = str((input_size, output_size))  # 需转字符串存入 CSV

    # **保存文件**
    latency_df.to_csv(latency_file)
    size_df.to_csv(size_file)

# ViT 层的示例
class ExampleVitLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.base_layer = nn.Linear(input_dim, output_dim)
        self.lora_A = nn.Linear(input_dim, input_dim // 2)
        self.lora_B = nn.Linear(input_dim // 2, output_dim)

    def forward(self, x):
        # 记录 base_layer 计算时间
        start_time = time.time()
        result = self.base_layer(x)
        end_time = time.time()
        log_vit_layer_data(index=0, start_time=start_time, end_time=end_time, input_tensor=x, output_tensor=result, new_file=True)

        # 记录 lora_A 计算时间
        start_time = time.time()
        result_A = self.lora_A(x)
        end_time = time.time()
        log_vit_layer_data(index=1, start_time=start_time, end_time=end_time, input_tensor=x, output_tensor=result_A)

        # 记录 lora_B 计算时间
        start_time = time.time()
        result_B = self.lora_B(result_A)
        end_time = time.time()
        log_vit_layer_data(index=2, start_time=start_time, end_time=end_time, input_tensor=result_A, output_tensor=result_B)

        return result_B

# 测试代码
if __name__ == "__main__":
    vit_layer = ExampleVitLayer(128, 64)
    input_tensor = torch.randn(1, 128)

    # **模拟多个 ViT 层的调用**
    for _ in range(15):  # 15 次调用，模拟 15 个不同 ViT 层的计算
        vit_layer(input_tensor)
