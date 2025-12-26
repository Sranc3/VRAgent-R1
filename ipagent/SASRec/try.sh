#!/bin/bash

# 输出文件
output_file="output.log"

# 基础命令
base_command="python run_text_test.py --load_ckpt_name /data2/tencent/MicroLens/Code/VideoRec/SASRec/checkpoint/checkpoint_MicroLens-100k_pairs_with_cold_text/cpt_v1_sasrec_blocknum_2_tau_0.07_bs_256_ed_512_lr_0.0001_l2_0.1_flrText_5e-05_bert-base-chinese_freeze_165_maxLen_10/epoch-"

# 循环从 82 到 120，步长为 2
for epoch in $(seq 82 2 120); do
    # 构建完整命令
    full_command="${base_command}${epoch}.pt"
    echo "Running command: $full_command" >> $output_file
    # 运行命令并将输出追加到日志文件
    $full_command >> $output_file 2>&1
    echo "Finished running epoch $epoch" >> $output_file
    echo "-------------------------" >> $output_file
done