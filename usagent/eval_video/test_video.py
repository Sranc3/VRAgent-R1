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


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=str)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--json_path', type=str)
    # parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    # print(args.model_path)
    #step = re.search(r'(\d+)$', args.model_path).group(1)

    # # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-stage3-len8192-step1800-t0_7-001/actor/global_step_{args.step}"
    # # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-test-01/actor/global_step_{args.step}"
    
    # # if args.stage == 0:
    # #     model_path = "/volume/ailab4sci/models/Qwen2.5-7B-Instruct-1M"
    # # elif args.stage == 1:
    # #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-002/actor/global_step_{args.step}"
    # # elif args.stage == 2:
    # #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-step1320-t0_7-001/actor/global_step_{args.step}"

    # model_path = "/volume/ailab4sci/models/Qwen2.5-7B-Instruct"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-12k-832"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-LC2k-1088"
    # model_name = args.model
    # model_path = f"/volume/ailab4sci/models/{model_name}"
    # model_path = "/volume/ailab4sci/ztgao/checkpoints/GRPO_logic_KK/rpp_qwen32b_5ppl_2e-6_16gpu/actor/global_step_120"

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
    df = pd.read_parquet("/data2/RL/data/processed/microlens_ranking_rl_top10_10000/test.parquet")
    data = df.to_dict('records')
    
    #加载测试集数据
    #data = data[:-100]
    
    cnt = 0
    total_time = 0
    results = []
    idx = 1

    for d in tqdm(data):
        
        #prompt = d["question"]
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\\boxed{}\\) </answer>."},
        #     {"role": "user", "content": prompt}
        # ]
        messages = d["prompt"]
        
        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        expected_answer = d['reward_model']['ground_truth']['next_recommendation']
        candidate_videos = d['reward_model']['ground_truth']['candidate_videos']
        correct_index = d['reward_model']['ground_truth']['correct_index']

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
        
        predicted_index = find_matching_video(pred_recommendation, candidate_videos)
        print(correct_index, predicted_index)
        

        correct = correct_index == predicted_index
        if not correct :
            print("Wrong Prediction:", idx)
       
        #correct = expected_answer in result
        
        # result = {
        #     "question": d['question'],
        #     "generated_output": response,
        #     "expected_expected_answer": expected_answer,
        #     "correct": correct,
        #     "time_taken": time_taken
        # }

        #results.append(result)

        if correct:
            cnt += 1
        print('current_acc:', cnt/idx)
        idx+=1

        total_time += time_taken
    
    acc = cnt / len(data)
    print(f"ACC: {acc}")
    # with open(f"{step}.json", 'w') as outfile:
    #     json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()