""" Preprocess dataset for video preference prediction task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    # 提供一个简单的替代函数，如果verl模块不可用
    def copy(src, dst):
        print(f"模拟HDFS复制操作: 从 {src} 到 {dst}")
    
    def makedirs(path):
        print(f"模拟HDFS创建目录: {path}")

import argparse
import json

def make_prefix(dp, template_type):
    user_history = dp['watch_history']
    if template_type == 'base':
        prefix = f"""The user has a history of watching videos, and the Assistant analyzes their preferences. The assistant first thinks about the reasoning process in the mind and then provides insights about user preferences and predicts what they might want to watch next. The reasoning process and prediction are enclosed within <think> </think> and <predict> </predict> tags, respectively, i.e., <think> reasoning process here </think><predict> prediction here </predict>. Now the user wants you to analyze their watching history. After thinking, clearly summarize the user's preferences and predict what category or specific content they might want to watch next within <predict> </predict> tags.\n\nUser watch history:{user_history}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first analyzes the user's video watching history to understand their preferences, then provides insights and recommendations. The analysis process and prediction are enclosed within <think> </think> and <predict> </predict> tags, respectively, i.e., <think> analysis process here </think><predict> prediction here </predict>. After thinking, clearly summarize the user's preferences and predict what category or specific content they might want to watch next within <predict> </predict> tags.\n<|im_end|>\n<|im_start|>user\nHere is my watching history: {user_history}\nCan you analyze my preferences and suggest what I might want to watch next?\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/video_preference/instruct')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='./data/raw/user_watch_history.jsonl')
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'video_preference'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(f"加载数据集总数: {len(raw_dataset)}")

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE, "数据集大小不足以满足指定的训练集和测试集大小"
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            # 创建输入提示
            question = make_prefix(example, template_type=args.template_type)
            
            # 定义解决方案（用于评估模型预测）
            solution = {
                "user_preferences": example['preferences'],
                "next_recommendation": example['next_recommendation']
            }
            
            # 构建完整数据结构
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "preference_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'user_id': example.get('user_id', f'user_{idx}')
                }
            }
            return data
        return process_fn

    # 应用转换函数到训练集和测试集
    print("正在处理训练集...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    print("正在处理测试集...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 创建本地目录（如果不存在）
    print(f"创建本地目录: {local_dir}")
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    # 保存处理后的数据集
    print("保存训练集...")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print("保存测试集...")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 如果指定了HDFS目录，则复制数据
    if hdfs_dir is not None:
        print(f"正在复制数据到HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        
    print("数据预处理完成！") 