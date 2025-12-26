#!/usr/bin/env python
"""
从MicroLens-100k数据集中提取用户观看历史数据，并为视频偏好预测任务构建数据集
每个用户的前n-1条视频作为输入序列，最后一条视频作为ground truth
"""

import os
import csv
import json
import pandas as pd
from typing import List, Dict, Any, Tuple

# 文件路径
PAIRS_PATH = "/data2/tencent/MicroLens-100k/MicroLens-100k_pairs.tsv"
ITEMS_PATH = "/data2/tencent/MicroLens-100k/complete_copy.csv"
OUTPUT_DIR = "/data2/RL/data/raw"
OUTPUT_FILE = "microlens_user_watch_history.jsonl"

def load_item_data(file_path: str) -> Dict[str, str]:
    """加载视频内容数据
    
    Args:
        file_path: 视频内容数据文件路径
        
    Returns:
        视频ID到标题的映射字典
    """
    item_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        reader = csv.reader(f)
        
        for row in reader:
            if len(row) >= 2:
                item_id = row[0]
                title = row[1].strip('"')
                item_data[item_id] = title
    
    print(f"加载了 {len(item_data)} 个视频内容信息")
    return item_data

def load_user_data(file_path: str, num_users: int = 100, min_history_length: int = 3) -> List[Tuple[str, List[str]]]:
    """加载用户观看历史数据
    
    Args:
        file_path: 用户观看历史数据文件路径
        num_users: 要提取的用户数量
        min_history_length: 最小观看历史长度，确保可以分割成输入和ground truth
        
    Returns:
        用户ID和观看历史列表的元组列表
    """
    user_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(user_data) >= num_users:
                break
                
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                user_id = parts[0]
                watch_history = parts[1].split()
                
                # 确保观看历史足够长，至少有min_history_length个视频
                if len(watch_history) >= min_history_length:
                    user_data.append((user_id, watch_history))
    
    print(f"提取了 {len(user_data)} 个用户的观看历史")
    return user_data

def generate_dataset(user_data: List[Tuple[str, List[str]]], 
                   item_data: Dict[str, str]) -> List[Dict[str, Any]]:
    """生成用户观看历史数据集
    
    Args:
        user_data: 用户ID和观看历史列表的元组列表
        item_data: 视频ID到标题的映射字典
        
    Returns:
        用户观看历史数据集
    """
    dataset = []
    
    for user_id, watch_history_ids in user_data:
        # 确保历史记录中有足够的视频
        if len(watch_history_ids) < 2:
            continue
            
        # 将历史记录分为输入序列(前n-1个)和ground truth(最后一个)
        input_history_ids = watch_history_ids[:-1]
        ground_truth_id = watch_history_ids[-1]
        
        # 收集输入序列的视频内容
        input_history_titles = []
        valid_input = True
        
        for item_id in input_history_ids:
            if item_id in item_data:
                input_history_titles.append(item_data[item_id])
            else:
                valid_input = False
                break
        
        # 确保ground truth ID在项目数据中
        if valid_input and ground_truth_id in item_data:
            ground_truth_title = item_data[ground_truth_id]
            
            # 创建格式化的观看历史文本
            if input_history_titles:
                watch_history_text = "\n".join(input_history_titles)
                
                # 创建完整记录
                record = {
                    "user_id": user_id,  # 直接使用数字ID
                    "watch_history": watch_history_text,
                    "next_recommendation": ground_truth_title  # 仅使用最后一个视频的标题作为ground truth
                }
                
                dataset.append(record)
    
    return dataset

def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """将数据保存为JSONL格式
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据已保存到: {output_path}")

def main():
    # 加载视频内容数据
    print("加载视频内容数据...")
    item_data = load_item_data(ITEMS_PATH)
    
    # 加载用户观看历史数据
    print("\n加载用户观看历史数据...")
    user_data = load_user_data(PAIRS_PATH, num_users=100, min_history_length=3)
    
    # 生成数据集
    print("\n生成数据集...")
    dataset = generate_dataset(user_data, item_data)
    
    # 保存数据
    print("\n保存数据...")
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    save_jsonl(dataset, output_path)
    
    print(f"\n共生成 {len(dataset)} 条用户观看历史记录")
    print(f"数据已保存到: {output_path}")

if __name__ == "__main__":
    main() 