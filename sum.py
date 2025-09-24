import os
import torch
from collections import defaultdict

# 设置主文件夹路径
main_folder_path = "/home/chenjj/workspace/vscode/UDA/share/experiments/pipeline/advSample/attackedLidar_Kitti100/train/ori_labels_pt"

# 初始化字典来存储所有类别的点数
total_category_counts = defaultdict(int)

# 遍历主文件夹中的所有子文件夹
for subfolder in range(14):  # 子文件夹从 00 到 13
    subfolder_name = f"{subfolder:02d}"  # 格式化子文件夹名称为 '00', '01', ..., '13'
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    if os.path.isdir(subfolder_path):
        print(f"Processing folder: {subfolder_name}")
        
        # 遍历子文件夹中的所有 .pt 文件
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".pt"):
                file_path = os.path.join(subfolder_path, file_name)
                
                # 加载 .pt 文件
                data = torch.load(file_path)
                
                # 确保 data 是一个一维 Tensor
                if data.dim() == 1:
                    # 统计每个类别的点数
                    unique_labels, counts = torch.unique(data, return_counts=True)
                    
                    # 更新总类别点数
                    for label, count in zip(unique_labels.tolist(), counts.tolist()):
                        total_category_counts[label] += count
                else:
                    print(f"Unexpected data format in {file_name}. Expected a 1D Tensor.")

# 输出最终统计结果
print("Total points per category across all .pt files in the adv folder and its subfolders:")
for category, count in total_category_counts.items():
    print(f"Category {category}: {count} points")
