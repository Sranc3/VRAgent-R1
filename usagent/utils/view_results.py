#!/usr/bin/env python
""" 查看生成的parquet文件内容 """

import argparse
import json
import pandas as pd
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="查看处理后的parquet文件内容")
    parser.add_argument("--file", default="./data/processed/video_preference/train.parquet", 
                        help="要查看的parquet文件路径")
    
    args = parser.parse_args()
    
    # 读取parquet文件
    df = pd.read_parquet(args.file)
    
    # 显示文件基本信息
    print(f"文件路径: {args.file}")
    print(f"数据条数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    print("\n===================== 数据示例 =====================\n")
    
    # 显示第一条数据的内容
    for idx, row in df.iterrows():
        print(f"样本 #{idx}:")
        
        # 处理prompt部分
        prompt = row['prompt'][0]['content']
        print("提示内容前100个字符:", prompt[:100] + "...")
        
        # 处理ground_truth部分
        print("用户偏好标签:", row['reward_model']['ground_truth']['user_preferences'])
        print("下一步推荐:", row['reward_model']['ground_truth']['next_recommendation'])
        
        # 显示用户ID
        print("用户ID:", row['extra_info']['user_id'])
        print("-" * 50)
        
        # 只显示第一条作为示例
        if idx == 0:
            break
    
    print("\n完整的第一条数据结构:")
    pprint(df.iloc[0].to_dict(), width=100, compact=False)

if __name__ == "__main__":
    main() 