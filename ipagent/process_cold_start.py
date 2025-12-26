import pandas as pd
from collections import Counter

# 读取原始数据
input_file = '/data2/tencent/MicroLens-100k/MicroLens-100k_pairs.tsv'
output_file = '/data2/tencent/MicroLens-100k/MicroLens-100k_pairs_colduser.tsv'

# 读取文件并处理
with open(input_file, 'r') as f:
    lines = f.readlines()

# 处理数据
user_interactions = {}  # 用户交互物品数量
original_lines = {}    # 存储原始行，以保持格式一致

# 先检查文件格式，读取前几行查看结构
print("检查文件格式:")
for i in range(min(5, len(lines))):
    print(f"第{i}行: {lines[i].strip()}")

for i, line in enumerate(lines):
    if i == 0:  # 保存标题行
        header = line
        continue
        
    parts = line.strip().split('\t')
    
    # 如果行格式是「用户ID \t 视频ID1 视频ID2 视频ID3...」
    if len(parts) == 2:
        user_id = parts[0]
        video_ids = parts[1].split()
        interaction_count = len(video_ids)
    # 如果行格式是「用户ID 视频ID1 视频ID2 视频ID3...」（单行无制表符）
    elif len(parts) == 1 and ' ' in parts[0]:
        all_ids = parts[0].split()
        user_id = all_ids[0]
        video_ids = all_ids[1:]
        interaction_count = len(video_ids)
    else:
        continue  # 跳过不符合格式的行
    
    user_interactions[user_id] = interaction_count
    original_lines[user_id] = line

# 输出一些统计信息来帮助理解数据格式
unique_interaction_counts = set(user_interactions.values())
print(f"交互数量的不同值: {sorted(unique_interaction_counts)}")
print(f"最小交互数: {min(user_interactions.values()) if user_interactions else 'N/A'}")
print(f"最大交互数: {max(user_interactions.values()) if user_interactions else 'N/A'}")

# 找出交互物品数量恰好等于5的用户
target_users = {user: count for user, count in user_interactions.items() if count == 5}

# 保存目标用户数据，完全保持原文件格式
with open(output_file, 'w') as f:
    # 写入标题行
    f.write(header)
    
    # 写入目标用户数据，直接使用原始行
    for user_id in target_users:
        f.write(original_lines[user_id])

# 打印统计信息
print(f"原始用户总数: {len(user_interactions)}")
print(f"交互物品数量恰好为5的用户数: {len(target_users)}")
print(f"原始记录总行数: {len(lines)-1}")  # 减去标题行
print(f"交互物品数量恰好为5的用户记录行数: {len(target_users)}")

# 打印用户交互物品数量的分布统计
interaction_counts = Counter(user_interactions.values())
print("\n用户交互物品数量分布:")
for count, frequency in sorted(interaction_counts.items()):
    print(f"交互物品数量为 {count} 的用户数: {frequency}")

# 打印一些示例数据
print("\n交互物品数量为5的用户数据示例:")
with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        if i > 0 and i <= 5:  # 打印前5个用户数据
            print(line.strip()) 