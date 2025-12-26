#!/usr/bin/env python
"""
准备视频偏好预测的数据集，应用结构化提示词模板
使用单个候选视频格式，要求模型分析用户偏好并预测用户是否会喜欢该视频
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

def make_prefix(item):
    """创建带有系统提示的前缀"""
    history_str = "\n".join([f"- {content}" for content in item['history_videos']])
    
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the user's video watching history, analyze their preferences, and predict if they would like the given candidate video, just give Yes or No. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. After thinking, when you finally reach a conclusion, give the user preference and your prediction within <answer> </answer> tags. i.e., <answer> (1) User_preference:... \n(2) Prediction: Yes/No  </answer>.
    <|im_end|>\n<|im_start|>user\n
    User's viewing history:
    {history_str}
    
    Candidate video:
    {item['candidate_video']}
    <|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def no_thinking(item):
    history_str = "\n".join([f"- {content}" for content in item['history_videos']])
    
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant thinks about the user's video watching history, and predict if they would like the given candidate video. The answer are enclosed within <answer> </answer> tags, respectively, i.e., <answer> answer here </answer>. 
    <|im_end|>\n<|im_start|>user\n
    User's viewing history:
    {history_str}
    
    Candidate video:
    {item['candidate_video']}
    <|im_end|>\n<|im_start|>assistant\n<answer>"""
    return prefix


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式的数据集
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_parquet_dataset(data: List[Dict[str, Any]], 
                           output_dir: str,
                           split_name: str) -> None:
    """创建Parquet格式的训练/测试数据集
    
    Args:
        data: 原始数据列表
        output_dir: 输出目录
        split_name: 数据集分割名称 ('train' 或 'test')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    for item in data:
        # 应用模板
        prompt = make_prefix(item)
        #prompt = no_thinking(item)
        
        # 创建记录
        record = {
            'prompt': [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "preference",
            'reward_model': {
                'ground_truth': {
                    'label': item['label'],  # 1表示喜欢，0表示不喜欢
                    'candidate_video': item['candidate_video'],
                    'history_videos': item['history_videos']
                }
            },
            'data_source': 'video_like'
        }
        formatted_data.append(record)
    
    # 转换为DataFrame并保存为Parquet
    df = pd.DataFrame(formatted_data)
    output_path = os.path.join(output_dir, f'{split_name}.parquet')
    df.to_parquet(output_path, index=False)
    print(f"已创建 {split_name} 数据集: {output_path} ({len(df)} 条记录)")

def main():
    # 设置路径
    data_dir = "data/raw"
    output_dir = "data/processed/microlens_like_rl"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载MicroLens偏好数据集
    microlens_data_path = os.path.join(data_dir, "microlens_like_data.jsonl")
    microlens_data = load_jsonl_data(microlens_data_path)
    print(f"加载了 {len(microlens_data)} 条MicroLens视频偏好记录")
    
    # 分割训练集和测试集 (90% 训练, 10% 测试)
    split_idx = int(len(microlens_data) * 0.9)
    train_data = microlens_data[:split_idx]
    test_data = microlens_data[split_idx:]
    
    # 创建数据集
    print("\n创建MicroLens视频偏好数据集...")
    create_parquet_dataset(train_data, output_dir, "train")
    create_parquet_dataset(test_data, output_dir, "test")
    
    print("\n数据集准备完成！")
    print(f"数据集保存在: {output_dir}")

if __name__ == "__main__":
    main() 