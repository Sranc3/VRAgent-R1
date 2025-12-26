#!/bin/bash

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 检查视频偏好奖励函数模块位置
# if [ ! -f "Logic-RL/verl/utils/reward_score/video_preference.py" ]; then
#     echo "复制奖励函数到Logic-RL框架..."
#     cp video_preference.py Logic-RL/verl/utils/reward_score/
# fi

# 检查初始化文件
if [ ! -f "Logic-RL/verl/utils/reward_score/__init__.py" ]; then
    echo "创建初始化文件..."
    cat > Logic-RL/verl/utils/reward_score/__init__.py << EOF
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module provides reward scoring functions for different tasks
# Import the functions here to make them available at the module level

# Uncomment the line below when the file exists
# from . import gsm8k, math, multiply, countdown, kk, video_preference

# Add video_preference to the imports
from . import video_preference

# Export patterns
__all__ = [
    # 'gsm8k',
    # 'math',
    # 'multiply',
    # 'countdown',
    # 'kk',
    'video_preference',
]
EOF
fi

# 创建必要的目录
# mkdir -p ./experiments/video_preference/grpo
# mkdir -p ./checkpoints/verl_examples/video_preference

# 指定输出目录
OUTPUT_DIR="outputs/video_preference_grpo"
mkdir -p $OUTPUT_DIR

# 指定模型路径（根据实际情况修改）
MODEL_PATH=~/models/Qwen2.5-7B

# 可选：设置XFORMERS作为注意力后端以提高性能
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 输出运行信息
echo "开始视频偏好预测模型GRPO训练..."
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 转到Logic-RL目录
cd Logic-RL
set -x

# 使用GRPO算法进行训练
# 参数设置参考main_grpo.sh并针对视频偏好任务进行了适配
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data2/RL/data/processed/microlens_ranking_rl_top10_10000/train.parquet \
    data.val_files=/data2/RL/data/processed/microlens_ranking_rl_top10_10000/test.parquet \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=80 \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=80 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.total_training_steps=null \
    trainer.project_name=verl_examples \
    trainer.experiment_name=video_preference_structured \
    trainer.logger=['wandb'] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.critic_warmup=0 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=./checkpoints/verl_examples/video_ranking_top10_10000 \
    trainer.total_epochs=3 $@ 2>&1 | tee video_preference_grpo.log

