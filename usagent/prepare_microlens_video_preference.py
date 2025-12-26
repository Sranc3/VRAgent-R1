#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import re

# 数据路径
MICROLENS_PAIRS_PATH = "/data2/tencent/MicroLens-100k/MicroLens-100k_pairs.tsv"
MICROLENS_VIDEO_INFO_PATH = "/data2/tencent/MicroLens-100k/complete_copy.csv"
RECOMMENDATION_LIST_PATH = "/data2/tencent/MicroLens/Code/VideoRec/SASRec/results/recommendations.csv"

OUTPUT_DIR = "/data2/RL/data/raw"
COMMENT_DIR = '/data2/tencent/MicroLens-100k_comment_en.txt'

def load_video_info():
    """加载视频内容信息"""
    try:
        video_df = pd.read_csv(
            MICROLENS_VIDEO_INFO_PATH,
            usecols=[0, 1],
            names=['video_id', 'title'],
            header=0,
            quoting=1,
            escapechar='\\',
            encoding='utf-8',
            engine='python',
            on_bad_lines='skip'
        )
        
        video_df['video_id'] = video_df['video_id'].astype(str)
        video_dict = dict(zip(video_df['video_id'], video_df['title']))
        print(f"成功加载 {len(video_dict)} 条视频信息")
        return video_dict
        
    except Exception as e:
        print(f"CSV读取错误: {str(e)}")
        print("尝试手动解析CSV文件...")
        
        video_dict = {}
        with open(MICROLENS_VIDEO_INFO_PATH, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                try:
                    parts = line.split(',', 1)
                    if len(parts) >= 2:
                        video_id = parts[0].strip('"')
                        title = parts[1].strip().strip('"')
                        video_dict[video_id] = title
                except Exception as e:
                    if i < 10:
                        print(f"跳过行 {i+2}: {str(e)}")
                    continue
        
        print(f"手动解析完成，加载 {len(video_dict)} 条视频信息")
        return video_dict

def load_user_sequences(max_users=10000, max_seq_length=8):
    """加载用户观看序列，限制最大长度"""
    user_sequences = []
    try:
        with open(MICROLENS_PAIRS_PATH, 'r') as f:
            print("检查文件格式...")
            
            for i, line in enumerate(f):
                if i >= max_users:
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    print(f"警告: 行 {i+1} 格式不正确: {line.strip()}")
                    continue
                    
                user_id = parts[0]
                video_ids_raw = parts[1].split()
                video_ids = [vid.strip() for vid in video_ids_raw]
                
                if 2 <= len(video_ids) <= max_seq_length + 1:
                    user_sequences.append({
                        'user_id': user_id,
                        'video_ids': video_ids
                    })
                    
                if i < 3:
                    print(f"用户 {user_id} 视频序列: {video_ids}")
    except Exception as e:
        print(f"加载用户序列时出错: {str(e)}")
    
    print(f"成功加载 {len(user_sequences)} 个用户序列")            
    return user_sequences

def load_recommendation_list():
    """加载用户推荐列表"""
    rec_dict = {}
    try:
        df = pd.read_csv(RECOMMENDATION_LIST_PATH)
        
        for _, row in df.iterrows():
            import ipdb; ipdb.set_trace()
            user_id = str(row['user_id'])
            video_ids = row['recommended_items'].strip('"').split(',')
            video_ids = [str(vid) for vid in video_ids]
            rec_dict[user_id] = video_ids
            
        print(f"成功加载 {len(rec_dict)} 个用户的推荐列表")
        return rec_dict
    except Exception as e:
        print(f"加载推荐列表时出错: {str(e)}")
        return {}

def load_comments():
    """加载用户评论数据"""
    comments_dict = {}
    try:
        with open(COMMENT_DIR, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t', 2)
                    if len(parts) >= 3:
                        user_id = parts[0].strip()
                        video_id = parts[1].strip()
                        comment = parts[2].strip()
                        comments_dict[(user_id, video_id)] = comment
                except Exception as e:
                    continue
        
        print(f"成功加载 {len(comments_dict)} 条评论")
        return comments_dict
    except Exception as e:
        print(f"加载评论数据时出错: {str(e)}")
        return {}

def create_preference_data(user_sequences, video_dict, max_samples=100):
    """创建偏好预测任务数据"""
    all_video_ids = list(video_dict.keys())
    data = []
    
    rec_dict = load_recommendation_list()
    comments_dict = load_comments()
    print(f"已加载推荐列表，包含 {len(rec_dict)} 个用户")
    
    random.shuffle(user_sequences)
    
    for user in tqdm(user_sequences, desc="Processing user data"):
        if len(data) >= max_samples:
            break
            
        user_id = user['user_id']
        video_ids = user['video_ids']
        
        # 将最后一个视频作为正样本
        history_videos = video_ids[:-1]
        positive_video_id = video_ids[-1]
        
        # 确保历史记录和正样本都在视频字典中
        if any(vid not in video_dict for vid in history_videos) or positive_video_id not in video_dict:
            continue
            
        # 从所有视频中随机选择3个负样本
        available_videos = [vid for vid in all_video_ids if vid not in video_ids]
        negative_videos = random.sample(available_videos, min(1, len(available_videos)))
        
        # 如果可用的负样本不足3个，从所有视频中随机补充
        while len(negative_videos) < 1:
            negative_videos.append(random.choice(all_video_ids))
            
        # 确保所有视频都在视频字典中
        if any(vid not in video_dict for vid in negative_videos):
            continue
            
        # 构建历史视频内容列表
        history_contents = []
        for vid in history_videos:
            content = video_dict[vid]
            comment = comments_dict.get((user_id, vid), "")
            history_contents.append(content)
        
        # 为每个视频（正样本和负样本）创建一个样本
        all_samples = [(positive_video_id, 1)] + [(vid, 0) for vid in negative_videos]
        
        for video_id, label in all_samples:
            data.append({
                'user_id': user_id,
                'history_videos': history_contents,
                'candidate_video': video_dict[video_id],
                'label': label
            })
    
    return data

def format_prompt(item):
    """格式化提示，创建偏好预测任务格式"""
    history_str = "\n".join([f"- {content}" for content in item['history_videos']])
    
    prompt = f"""Based on the user's viewing history, analyze their preferences and predict if they would like the following video.

User's viewing history:
{history_str}

Candidate video:
{item['candidate_video']}

First analyze the user's interest patterns and preferences, then predict if they would like this video. Format your answer as follows:
<think>
Write your analysis process here, including the user's interest preferences and reasoning for your prediction
</think>
<answer>
(1) User_preference: briefly summarize the user's interests and preferences
(2) Prediction: Yes/No - whether you predict the user would like this video
</answer>
"""
    return prompt

def save_to_jsonl(data, output_path):
    """保存数据到JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to {output_path}")

def main():
    # 设置随机种子确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 加载视频信息
    print("Loading video information...")
    video_dict = load_video_info()
    print(f"Loaded {len(video_dict)} video entries")
    
    # 加载用户序列
    print("Loading user sequences...")
    user_sequences = load_user_sequences(max_users=20000, max_seq_length=8)
    print(f"Loaded {len(user_sequences)} user sequences")
    
    # 创建偏好预测任务数据
    print("Creating preference prediction data...")
    preference_data = create_preference_data(user_sequences, video_dict, max_samples=2000)
    print(f"Created {len(preference_data)} preference examples")
    
    # 保存为JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, 'microlens_like_data.jsonl')
    save_to_jsonl(preference_data, jsonl_path)
    
    print("数据处理完成！")

if __name__ == "__main__":
    main() 