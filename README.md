# VRAgent-R1

A reinforcement learning-based video recommendation agent system that combines large language models with sequential recommendation models for personalized video recommendations.

## Overview

VRAgent-R1 consists of two main components:

- **ipagent**: Multi-modal sequential recommendation model based on SASRec
- **usagent**: Reinforcement learning training framework (PPO/GRPO) for video preference prediction

## Project Structure

```
VRAgent-r1/
├── ipagent/                    # Sequential recommendation module
│   ├── SASRec/                 # SASRec model implementation
│   └── qwen.py                 # Qwen2-VL model integration
│
└── usagent/                    # RL training module
    ├── verl/                   # veRL framework integration
    ├── utils/                  # Utility functions and reward functions
    ├── run_video_like_ppo.sh   # Video like PPO training
    ├── run_video_preference_ppo.sh  # Video preference PPO training
    └── main_grpo_video.sh     # GRPO training
```

## Features

- **Multi-modal Support**: ID, image, text, and video inputs
- **RL Training**: PPO and GRPO algorithms for preference learning
- **Video Preference Prediction**: Predict user preferences from watch history
- **Cold Start Handling**: Support for cold-start users

## Quick Start

### Installation

```bash
git clone https://github.com/Sranc3/VRAgent-R1.git
cd VRAgent-r1

pip install torch transformers datasets accelerate wandb
pip install pytorchvideo decord
```

### Data Preparation

Prepare video preference data in JSONL format:

```json
{
  "user_id": "user_001",
  "watch_history": "...",
  "preferences": ["category1", "category2"],
  "next_recommendation": "..."
}
```

Process data:

```bash
cd usagent
python prepare_microlens_video_preference.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/processed
```

### Training

**PPO Training:**
```bash
cd usagent
bash run_video_preference_ppo.sh
```

**GRPO Training:**
```bash
cd usagent
bash main_grpo_video.sh
```

### Evaluation

```bash
cd usagent/eval_video
python test_video.py --model_path /path/to/checkpoint
```

## Configuration

Key parameters in training scripts:

- `data.train_files`: Training data path
- `actor_rollout_ref.model.path`: Base model path
- `actor_rollout_ref.actor.optim.lr`: Learning rate
- `trainer.n_gpus_per_node`: Number of GPUs per node
- `trainer.total_epochs`: Training epochs

## Results

Results are saved in `usagent/outputs/`. Training logs are tracked with WandB (project: `GRPO_logic_video`).

## License

Apache License 2.0

## Acknowledgments

- [veRL](https://github.com/volcengine/verl): Efficient RL training framework
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo): Video understanding library
- [SASRec](https://github.com/kang205/SASRec): Self-Attentive Sequential Recommendation
