#!/usr/bin/python python3
import warnings
warnings.filterwarnings("ignore")
import re
import os
import csv
import json
import time
import types
import random
import textwrap
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from vllm import LLM, SamplingParams
from datetime import datetime
from typing import Dict, Tuple, Optional, List

first_round_anno = json.load(open('/data2/RL/Logic-RL/eval_video/res_first_round.json'))
def search_user_id(user_id):
    for anno in first_round_anno:
        if anno['user_id'] == user_id:
            return anno["pred_idx_1"][0]

def parse_recommendation(text: str) -> str:
    """解析文本中的推荐内容。
    
    Args:
        text: 包含推荐的文本
        
    Returns:
        提取的推荐内容
    """
    print("\n[推荐解析]")
    
    # 首先尝试提取标准格式的推荐（优先）
    # 支持英文模板的格式: "Next_video: recommendation"
    standard_pattern = r'next_video:?\s*(.+?)(?:\n|$|\.|;)'
    standard_match = re.search(standard_pattern, text.lower(), re.DOTALL)
    
    if standard_match:
        rec = standard_match.group(1).strip()
        # 去除可能的引号
        rec = rec.strip('"\'')
        print(f"  标准格式推荐: {rec}")
        return rec
    
    # 没有找到标准格式，尝试通用匹配
    english_patterns = [
        r'recommend(?:ed)? (?:watching |viewing |)([\w\s&,:\-\'".]+)',
        r'suggestion: ([\w\s&,:\-\'".]+)',
        r'might enjoy ([\w\s&,:\-\'".]+)',
        r'would (?:like|enjoy) ([\w\s&,:\-\'".]+)',
        r'next (?:video|content|course): ([\w\s&,:\-\'".]+)',
        r'recommended (?:next |)(?:content|video|course): ([\w\s&,:\-\'".]+)',
        r'should watch ([\w\s&,:\-\'".]+)',
        r'ideal next (?:video|content) (?:is |would be |)([\w\s&,:\-\'".]+)',
        r'predict (?:the user will watch |)([\w\s&,:\-\'".]+)'
    ]
    
    # chinese_patterns = [
    #     r'推荐(?:观看|)([^。；\n]*)',
    #     r'建议(?:观看|)([^。；\n]*)',
    #     r'可以(?:观看|尝试)([^。；\n]*)',
    #     r'接下来(?:可以|应该)(?:观看|学习)([^。；\n]*)',
    #     r'下一个视频(?:应该是|可以是|:)([^。；\n]*)'
    # ]
    
    # 合并所有模式
    all_patterns = english_patterns #+ chinese_patterns
    
    for pattern in all_patterns:
        match = re.search(pattern, text.lower())
        if match:
            rec = match.group(1).strip()
            print(f"  找到非标准推荐: {rec}")
            return rec
    
    print("  [警告] 未找到明确的推荐")
    return ""

def find_matching_video(prediction: str, candidate_videos: List[str]) -> int:
    """在候选视频中找到与预测最匹配的视频索引。
    
    Args:
        prediction: 预测的视频标题
        candidate_videos: 候选视频列表
        
    Returns:
        匹配视频的索引，如果没有找到则返回-1
    """
    # if not prediction or  candidate_videos==None:
    #     return -1
    
    pred_lower = prediction.lower().strip()
    
    # 首先尝试完全匹配
    for i, video in enumerate(candidate_videos):
        if video.lower().strip() == pred_lower:
            print(f"  完全匹配: 预测选择了候选项 {i}")
            return i
    
    # 如果没有完全匹配，尝试部分匹配（一个是另一个的子字符串）
    for i, video in enumerate(candidate_videos):
        if pred_lower in video.lower().strip() or video.lower().strip() in pred_lower:
            print(f"  部分匹配: 预测选择了候选项 {i}")
            return i
    
    # 如果仍然没有匹配，计算单词重叠度
    best_match = -1
    best_score = 0.0
    
    for i, video in enumerate(candidate_videos):
        video_lower = video.lower().strip()
        
        # 分词比较
        pred_words = set(re.findall(r'\b\w+\b', pred_lower))
        video_words = set(re.findall(r'\b\w+\b', video_lower))
        
        # 计算相似单词
        common_words = pred_words.intersection(video_words)
        
        # 计算Jaccard相似度
        if pred_words and video_words:
            jaccard = len(common_words) / len(pred_words.union(video_words))
            
            if jaccard > 0.3 and jaccard > best_score:
                best_score = jaccard
                best_match = i
    
    if best_match >= 0:
        print(f"  单词重叠匹配: 预测选择了候选项 {best_match} (相似度: {best_score:.2f})")
        return best_match
    
    print("  [警告] 无法匹配预测到任何候选视频", prediction, candidate_videos)
    return -1

def format_prompt(history_str, candidates_str):
    """格式化提示，创建视频排序任务格式"""
    history_str = "\n".join([f"- {content}" for content in history_str])
    candidates_str = "\n".join([f"- {content}" for content in candidates_str])
    
    prompt = f"""Based on the user's viewing history, analyze their preferences and interests, then predict the next video which the user is most likely to watch from the following candidates.

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
(2) Next_video: copy the exact title of the video you predict the user will watch next
</answer>
"""
    return prompt

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=str)
    parser.add_argument('--model_path', default='/data2/RL/Logic-RL/checkpoints/verl_examples/video_ranking_top10_10000/actor/global_step_1450', type=str)
    parser.add_argument('--json_path', default='/data2/RL/data/raw/microlens_ranking_data_for_cold.jsonl', type=str)
    args = parser.parse_args()
   

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4,
        max_model_len=20000
    )
    
    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=0.8,
        top_p=0.95,
    )

    # 读取parquet文件
    # 全部都弄到一个parquet里面，然后批量处理。
    with open(args.json_path, 'r') as f:

        cnt = 0
        total_time = 0
        results = json.load(open('res_second_round.json'))
        idx = 1

        for d in tqdm(f):
            cnt += 1
            print(cnt)
            if cnt <= 5885:
                continue
            d = json.loads(d)
            first_round_id = d['candidate_video_id'][search_user_id(d['user_id'])]
            del d['candidate_videos'][search_user_id(d['user_id'])]
            del d['candidate_video_id'][search_user_id(d['user_id'])]
            messages = format_prompt(d['history_videos'], d['candidate_videos'])
            messages = [{'content':messages, 'role':'user'}]
            tokenizer = llm.get_tokenizer()
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # correct_index = d['correct_index']

            start_time = time.time()
            outputs = llm.generate([text], sampling_params)
            time_taken = time.time() - start_time
            response = outputs[0].outputs[0].text.strip()
            if '<answer>' in response:
                result = re.split(r'<answer>', response)[1]
            else:
                result = response[len(response) - 30:]
            
            
            pred_recommendation = parse_recommendation(result)
            #pred_recommendation = result
            
            predicted_index = find_matching_video(pred_recommendation, d['candidate_videos'])
            # print(correct_index, predicted_index)
            results.append({'user_id':d['user_id'], 'output': result, 'pred_idx_2': [predicted_index], 'pred_id_in_recommend':[first_round_id, d['candidate_video_id'][predicted_index]]})
            # print('haha')
            with open('res_second_round.json', 'w') as file_json:
                json.dump(results, file_json)
                


if __name__ == "__main__":
    main()