#!/usr/bin/env python
"""
准备视频偏好预测的数据集，应用结构化提示词模板
简化版本：只包含历史视频内容和下一个视频的真实内容
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

def make_prefix(item):
    history = item['watch_history']
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the user's video watching history, analyze their preferences, and recommend the next video content they might be interested in. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to recommend a video. After thinking, when you finally reach a conclusion, give the user perference and recommend the next video content within <answer> </answer> tags. i.e., <answer> (1) User_preferences:... \n(2) Next_video:...  </answer>.\n<|im_end|>\n<|im_start|>user\n{history}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
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

def load_prompt_template(template_path: str) -> str:
    """加载提示词模板
    
    Args:
        template_path: 模板文件路径
        
    Returns:
        模板字符串
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def apply_template(template: str, watch_history: str) -> str:
    """将观看历史应用到模板中
    
    Args:
        template: 提示词模板
        watch_history: 用户观看历史
        
    Returns:
        填充后的提示词
    """
    return template.replace('{watch_history}', watch_history)

def create_parquet_dataset(data: List[Dict[str, Any]], 
                        #    template: str, 
                           output_dir: str,
                           split_name: str) -> None:
    """创建Parquet格式的训练/测试数据集
    
    Args:
        data: 原始数据列表
        template: 提示词模板
        output_dir: 输出目录
        split_name: 数据集分割名称 ('train' 或 'test')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    for item in data:
        # 应用模板
        quesiton = make_prefix(item)
        #prompt = apply_template(template, item['watch_history'])
        
        # 创建记录
        record = {
            'prompt': [{
                "role": "user",
                "content": quesiton,
            }],
            "ability": "recommend",
            'reward_model': {
                'ground_truth': {
                    # 'user_preferences': item['preferences'],
                    'next_recommendation': item['next_recommendation']
                }
            },
            'data_source': 'video_preference'
            
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
    output_dir = "data/processed/microlens_video_preference"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载MicroLens数据集
    microlens_data_path = os.path.join(data_dir, "microlens_user_watch_history.jsonl")
    microlens_data = load_jsonl_data(microlens_data_path)
    print(f"加载了 {len(microlens_data)} 条MicroLens用户观看历史记录")
    
    # 分割训练集和测试集 (80% 训练, 20% 测试)
    split_idx = int(len(microlens_data) * 0.8)
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