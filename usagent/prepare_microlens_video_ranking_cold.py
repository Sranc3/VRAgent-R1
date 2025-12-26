#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import re
import json
all_dict = json.load(open('/data2/tencent/MicroLens/Code/VideoRec/SASRec/results/sorted_dict_ori_pos.json'))

all_dict_rev = json.load(open('/data2/tencent/MicroLens/Code/VideoRec/SASRec/results/sorted_dict.json'))
# 数据路径
MICROLENS_PAIRS_PATH = "/data2/tencent/MicroLens-100k/MicroLens-100k_pairs_colduser.tsv"
MICROLENS_VIDEO_INFO_PATH = "/data2/tencent/MicroLens-100k/complete_copy.csv"
# MICROLENS_VIDEO_INFO_PATH = "/data2/tencent/MLLM_sum_used_video.csv"
RECOMMENDATION_LIST_PATH = "/data2/tencent/MicroLens/Code/VideoRec/SASRec/results/recommendations.csv"

OUTPUT_DIR = "/data2/RL/data/raw"
#PROCESSED_DIR = "/data2/RL/data/processed/microlens_ranking"
COMMENT_DIR = '/data2/tencent/MicroLens-100k_comment_en.txt'

# 创建输出目录
#os.makedirs(OUTPUT_DIR, exist_ok=True)
#os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_video_info():
    """加载视频内容信息"""
    try:
        # 使用更宽松的参数读取CSV
        video_df = pd.read_csv(
            MICROLENS_VIDEO_INFO_PATH,
            usecols=[0, 1],  # 只读取前两列
            names=['video_id', 'title'],  # 指定列名
            header=0,  # 第一行是标题
            quoting=1,  # 引号内的逗号视为文本
            escapechar='\\',  # 处理转义字符
            encoding='utf-8',  # 指定编码
            engine='python',  # 使用Python引擎
            on_bad_lines='skip'  # 新版pandas用on_bad_lines代替error_bad_lines
        )
        
        # 处理视频ID为数字
        video_df['video_id'] = video_df['video_id'].astype(str)
        
        # 转换为字典
        video_dict = dict(zip(video_df['video_id'], video_df['title']))
        print(f"成功加载 {len(video_dict)} 条视频信息")
        return video_dict
        
    except Exception as e:
        print(f"CSV读取错误: {str(e)}")
        print("尝试手动解析CSV文件...")
        
        # 手动解析CSV
        video_dict = {}
        with open(MICROLENS_VIDEO_INFO_PATH, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for i, line in enumerate(f):
                try:
                    # 简单分割，假设第一个逗号前是ID
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
    import csv

    with open(MICROLENS_PAIRS_PATH, 'r', newline='') as file1:
        # 读取 TSV 文件
        tsv_reader = csv.reader(file1, delimiter='\t')

        for i, line1 in enumerate(tsv_reader):
            # 处理第一个文件的行
            user_id = line1[-1].split(' ')
            if len(user_id) >= 13:
                user_id = user_id[-13:]
            user_sequences.append({
                'user_id': line1[0],
                'video_ids': user_id
            })
            if i < 3:  # 打印前几个用户的处理结果用于调试
                print(f"用户 {line1[0]} 视频序列: {user_id}")
            
    return user_sequences

def load_recommendation_list():
    """加载用户推荐列表"""
    rec_dict = {}
    try:
        # 读取推荐列表CSV文件
        df = pd.read_csv(RECOMMENDATION_LIST_PATH)
        
        # 处理每一行数据
        for _, row in df.iterrows():
            user_id = str(int(row['user_id'] + 1))  # 使用正确的列名
            # 将推荐项目字符串分割成列表
            video_ids = row['recommended_items'].strip('"').split(',')
            video_ids = [str(vid) for vid in video_ids]  # 确保所有ID都是字符串
            rec_dict[user_id] = video_ids
            
        print(f"成功加载 {len(rec_dict)} 个用户的推荐列表")
        return rec_dict
    except Exception as e:
        print(f"加载推荐列表时出错: {str(e)}")
        return {}

def load_comments():
    """加载用户评论数据"""
    comments_dict = {}  # 格式: {(user_id, video_id): comment}
    try:
        with open(COMMENT_DIR, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 由于评论中可能包含制表符，我们只分割前两个字段
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

def format_prompt(history_str, candidates_str):
    """格式化提示，创建视频排序任务格式"""
    history_str = "\n".join([f"- {content}" for content in history_str])
    candidates_str = "\n".join([f"- {content}" for content in candidates_str])
    
    prompt = f"""Based on the user's viewing history, analyze their preferences and interests, then predict the next two videos which the user is most likely to watch from the following candidates.

User's viewing history:
{history_str}

Candidate videos for the next watch:
{candidates_str}

First analyze the user's interest patterns and preferences, then select the video that best matches these patterns. Format your answer as follows:
<think>
Write your analysis process here, including the user's interest preferences and reasoning for your choice
</think>
<answer>
(1) User_preference: briefly summarize the user's interests and preferences
(2) Next_video: copy the exact title of the two videos you predict the user will watch next
</answer>
"""
    return prompt

def create_cold_data(user_sequences, video_dict, max_samples=100):
    """创建排序任务数据，限制样本数量"""
    all_video_ids = list(video_dict.keys())
    data = []
    # 加载推荐列表和评论数据
    # 这个数据是比较小的数据
    rec_dict = load_recommendation_list()
    comments_dict = load_comments()
    print(f"已加载推荐列表，包含 {len(rec_dict)} 个用户")
    
    for user in tqdm(user_sequences, desc="Processing user data"):
        if len(data) >= max_samples:
            break
            
        user_id = user['user_id']
        video_ids = user['video_ids']
        
        # 将最后一个视频作为正确答案
        history_videos = video_ids[:-1]
        correct_video_id = video_ids[-1]
        if user_id not in rec_dict.keys(): continue
        candidate_vids = rec_dict[user_id]
        candidate_video_ids = []
        for idx in candidate_vids:
            if idx in all_dict_rev.keys():
                candidate_video_ids.append(all_dict_rev[idx])
            else:
                candidate_video_ids.append(idx)
            
        
        # 找到正确答案在候选列表中的索引位置
        
        correct_index = False
        if correct_video_id in candidate_video_ids:
            print(candidate_video_ids)
            print(correct_video_id)
            correct_index = True

        
        # 构建历史视频内容列表（包含评论）
        history_contents = []
        for vid in history_videos:
            if vid not in video_dict.keys(): continue
            content = video_dict[vid]
            # 添加评论（如果存在）
            comment = comments_dict.get((user_id, vid), "")
            if comment:
                content = f"{content}, user comment: {comment}"
            history_contents.append(content)
        
        # 构建候选视频内容列表（不包含评论）
        candidate_contents = [video_dict[vid] for vid in candidate_video_ids]

        # 添加到数据集
        data.append({
            'candidate_video_id': candidate_video_ids,
            'history_id': history_videos,
            'corr_id':correct_video_id,
            'user_id': user_id,
            'history_videos': history_contents,
            'candidate_videos': candidate_contents,
            'correct_index': correct_index,
            'ground_truth': video_dict[correct_video_id],
            # 'format_prompt':format_prompt(history_contents, candidate_contents)
        })
    
    return data



def save_to_jsonl(data, output_path):
    """保存数据到JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to {output_path}")

def manual_split(data, test_ratio=0.2):
    """手动分割数据为训练集和测试集"""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def prepare_parquet_files(jsonl_path):
    """准备训练和测试的parquet文件"""
    # 读取JSONL数据
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 分割训练和测试集
    train_data, test_data = manual_split(data, test_ratio=0.2)
    
    # 准备数据帧
    train_df = pd.DataFrame({
        'prompt': [format_prompt(item) for item in train_data],
        'response': ["" for _ in train_data],  # 空响应，将由模型生成
        'history_videos': [item['history_videos'] for item in train_data],
        'candidate_videos': [item['candidate_videos'] for item in train_data],
        'correct_index': [item['correct_index'] for item in train_data],
        'ground_truth': [item['ground_truth'] for item in train_data]
    })
    
    test_df = pd.DataFrame({
        'prompt': [format_prompt(item) for item in test_data],
        'response': ["" for _ in test_data],  # 空响应，将由模型生成
        'history_videos': [item['history_videos'] for item in test_data],
        'candidate_videos': [item['candidate_videos'] for item in test_data],
        'correct_index': [item['correct_index'] for item in test_data],
        'ground_truth': [item['ground_truth'] for item in test_data]
    })
    
    # 保存为parquet
    train_df.to_parquet(os.path.join(PROCESSED_DIR, 'train.parquet'), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DIR, 'test.parquet'), index=False)
    
    print(f"Training set: {len(train_data)} samples, Test set: {len(test_data)} samples")
    print(f"Saved to {PROCESSED_DIR}")

def prepare_test_csv(ranking_data, output_path, video_dict):
    """准备测试集CSV文件，格式为：user,item,label"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("user,item,label\n")
        
        # 为每个样本写入数据
        for item in ranking_data:
            user_id = item['user_id']
            candidate_video_ids = []
            
            # 获取正样本视频ID（需要从视频内容反向查找视频ID）
            for video_id, content in video_dict.items():
                if content == item['ground_truth']:
                    candidate_video_ids.append((video_id, 1))  # 正样本
                    break
            
            # 获取负样本视频ID
            for i, content in enumerate(item['candidate_videos']):
                if i != item['correct_index']:  # 不是正样本
                    for video_id, vid_content in video_dict.items():
                        if content == vid_content:
                            candidate_video_ids.append((video_id, 0))  # 负样本
                            break
            
            # 写入数据
            for video_id, label in candidate_video_ids:
                f.write(f"{user_id},{video_id},{label}\n")

def save_used_video_titles(ranking_data, video_dict, output_path):
    """保存数据集中使用到的视频ID和标题信息，格式与MicroLens-50k_titles.json保持一致"""
    used_videos = []
    
    # 收集所有使用到的视频
    for item in ranking_data:
        # 从历史记录中收集
        for content in item['history_videos']:
            # 通过内容反向查找视频ID
            for video_id, title in video_dict.items():
                if title == content:
                    used_videos.append({
                        "id": video_id,
                        "title": title
                    })
                    break
        
        # 从候选视频中收集
        for content in item['candidate_videos']:
            for video_id, title in video_dict.items():
                if title == content:
                    used_videos.append({
                        "id": video_id,
                        "title": title
                    })
                    break
    
    # 去重
    unique_videos = []
    seen_ids = set()
    for video in used_videos:
        if video['id'] not in seen_ids:
            seen_ids.add(video['id'])
            unique_videos.append(video)
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_videos, f, ensure_ascii=False, indent=2)
    print(f"已保存使用到的视频信息到: {output_path}")
    print(f"共保存了 {len(unique_videos)} 个视频的信息")

def main():
    # 设置随机种子确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 加载视频信息
    print("Loading video information...")
    # video介绍
    video_dict = load_video_info()
    print(f"Loaded {len(video_dict)} video entries")
    
    # 加载用户序列
    print("Loading user sequences...")
    # 加载冷启动用户的所有信息
    user_sequences = load_user_sequences(max_users=30000, max_seq_length=10)
    print(f"Loaded {len(user_sequences)} user sequences")
    
    # 创建排序任务数据
    print("Creating ranking data...")
    ranking_data = create_cold_data(user_sequences, video_dict, max_samples=10000)
    print(f"Created {len(ranking_data)} ranking examples")
    
    # 保存为JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, 'microlens_ranking_data_for_cold.jsonl')
    save_to_jsonl(ranking_data, jsonl_path)
    
    # 生成测试集CSV文件
    # test_csv_path = os.path.join(OUTPUT_DIR, 'test_pairs.csv')
    # print("生成测试集CSV文件...")
    # prepare_test_csv(ranking_data, test_csv_path, video_dict)
    # print(f"测试集CSV文件已保存到: {test_csv_path}")
    
    # 保存使用到的视频ID和标题信息
    #save_used_video_titles(ranking_data, video_dict, os.path.join(OUTPUT_DIR, 'used_video_titles.json'))
    
    print("数据处理完成！")

if __name__ == "__main__":
    main() 