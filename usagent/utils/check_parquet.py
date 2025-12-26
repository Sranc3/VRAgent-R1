#!/usr/bin/env python
"""
检查parquet文件的内容
"""

import pandas as pd
import json

# 读取parquet文件
train_file = "data/processed/microlens_video_preference/train.parquet"
df = pd.read_parquet(train_file)

# 显示数据集的基本信息
print(f"数据集大小: {len(df)} 条记录")
print(f"列名: {df.columns.tolist()}")

# 显示第一条记录的详细信息
first_record = df.iloc[0]
print("\n第一条记录:")
print(f"提示内容类型: {type(first_record['prompt'])}")

# 如果prompt是列表，解析第一个元素
if isinstance(first_record['prompt'], list) and len(first_record['prompt']) > 0:
    first_prompt = first_record['prompt'][0]
    print(f"提示角色: {first_prompt.get('role', 'N/A')}")
    print(f"提示内容前100个字符: {first_prompt.get('content', 'N/A')[:100]}...")
else:
    print("提示内容格式不符合预期")

# 显示ground truth信息
if 'reward_model' in first_record and isinstance(first_record['reward_model'], dict):
    reward_model = first_record['reward_model']
    if 'ground_truth' in reward_model and isinstance(reward_model['ground_truth'], dict):
        ground_truth = reward_model['ground_truth']
        print("\nGround Truth信息:")
        for key, value in ground_truth.items():
            # 如果值太长，只显示部分
            if isinstance(value, str) and len(value) > 100:
                value_display = value[:100] + "..."
            else:
                value_display = value
            print(f"  {key}: {value_display}")
    else:
        print("\nGround Truth信息不可用")
else:
    print("\nReward Model信息不可用")

# 显示前3条记录的摘要信息
print("\n前3条记录的摘要:")
for i in range(min(3, len(df))):
    record = df.iloc[i]
    next_rec = record['reward_model']['ground_truth']['next_recommendation']
    next_rec_short = next_rec[:50] + "..." if len(next_rec) > 50 else next_rec
    print(f"记录 {i+1}: next_recommendation = {next_rec_short}") 