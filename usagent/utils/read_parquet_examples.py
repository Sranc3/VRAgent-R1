#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import random

# 默认的parquet文件路径
DEFAULT_TRAIN_PATH = "Logic-RL/data/kk/instruct/3ppl/train.parquet"
#DEFAULT_TRAIN_PATH = "/data2/RL/data/processed/microlens_ranking_rl/train.parquet"
DEFAULT_TEST_PATH = "Logic-RL/data/kk/instruct/3ppl/test.parquet"

def read_parquet_sample(file_path, num_samples=5, random_select=False, specific_indices=None, columns=None):
    """
    读取parquet文件并返回指定数量的样例
    
    参数:
        file_path (str): parquet文件路径
        num_samples (int): 要展示的样例数量
        random_select (bool): 是否随机选择样例
        specific_indices (list): 指定要展示的样例索引列表
        columns (list): 只显示指定的列
    
    返回:
        DataFrame: 选定的样例数据
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    # 读取parquet文件
    print(f"正在读取文件: {file_path}")
    df = pd.read_parquet(file_path)
    total_rows = len(df)
    all_columns = df.columns.tolist()
    print(f"总样本数: {total_rows}")
    print(f"所有列名: {all_columns}")
    
    # 如果指定了列，验证列名是否有效
    if columns:
        valid_columns = [col for col in columns if col in all_columns]
        if not valid_columns:
            print(f"警告: 所有指定的列名都无效，将显示所有列")
        else:
            if len(valid_columns) != len(columns):
                invalid_columns = set(columns) - set(valid_columns)
                print(f"警告: 以下列名无效，将被忽略: {invalid_columns}")
            df = df[valid_columns]
            print(f"已选择列: {valid_columns}")
    
    # 选择样例
    if specific_indices:
        # 确保索引在有效范围内
        valid_indices = [i for i in specific_indices if 0 <= i < total_rows]
        if not valid_indices:
            raise ValueError("所有指定索引都超出范围")
        selected_df = df.iloc[valid_indices]
    elif random_select:
        # 随机选择样例
        if num_samples > total_rows:
            num_samples = total_rows
        selected_indices = random.sample(range(total_rows), num_samples)
        selected_df = df.iloc[selected_indices]
    else:
        # 选择前N个样例
        num_samples = min(num_samples, total_rows)
        selected_df = df.head(num_samples)
    
    return selected_df

def main():
    parser = argparse.ArgumentParser(description="读取parquet文件并展示样例")
    parser.add_argument("--file_path", "-f", default=DEFAULT_TRAIN_PATH,
                        help=f"parquet文件路径 (默认: {DEFAULT_TRAIN_PATH})")
    parser.add_argument("--test", "-t", action="store_true", 
                        help=f"使用测试集 (路径: {DEFAULT_TEST_PATH})")
    parser.add_argument("--num", "-n", type=int, default=5, help="要展示的样例数量 (默认: 5)")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择样例")
    parser.add_argument("--indices", "-i", type=int, nargs="+", help="指定要展示的样例索引")
    parser.add_argument("--columns", "-c", type=str, nargs="+", help="只显示指定的列名")
    parser.add_argument("--output", "-o", help="输出结果到文件")
    
    args = parser.parse_args()
    
    # 如果指定了使用测试集，则覆盖文件路径
    if args.test:
        args.file_path = DEFAULT_TEST_PATH
    
    try:
        selected_samples = read_parquet_sample(
            args.file_path, 
            args.num, 
            args.random, 
            args.indices,
            args.columns
        )
        
        # 输出结果
        if args.output:
            selected_samples.to_csv(args.output, index=False)
            print(f"结果已保存到 {args.output}")
        else:
            # 打印每个样例
            for idx, row in selected_samples.iterrows():
                print("\n" + "="*80)
                print(f"样例 #{idx}")
                print("="*80)
                for col in row.index:
                    print(f"{col}:")
                    # 如果值是字符串且较长，则格式化输出
                    if isinstance(row[col], str) and len(row[col]) > 100:
                        print("-"*40)
                        print(row[col])
                        print("-"*40)
                    else:
                        print(f"{row[col]}")
                    print()
    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()

"""
使用示例:

1. 显示训练集的前5个样例:
   python read_parquet_examples.py

2. 显示测试集的前5个样例:
   python read_parquet_examples.py --test

3. 随机显示10个样例:
   python read_parquet_examples.py --random --num 10

4. 显示特定索引的样例:
   python read_parquet_examples.py --indices 0 10 20

5. 只显示特定列:
   python read_parquet_examples.py --columns input output

6. 将结果保存到CSV文件:
   python read_parquet_examples.py --output samples.csv

7. 组合使用:
   python read_parquet_examples.py --test --random --num 3 --columns input output
""" 