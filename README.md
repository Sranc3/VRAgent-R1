# VRAgent-R1

A reinforcement learning-based video recommendation agent system that combines large language models with sequential recommendation models for personalized video recommendations.

## Overview

VRAgent-R1 consists of two main components:

- **IP Agent**: Multi-modal item perception based on MLLM, e.g., Qwen-2.5VL
- **US Agent**: User Simulation agent based on GRPO reinforcement learning


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


### Training


**GRPO Training:**
```bash
cd usagent
bash main_grpo_video.sh
bash bash run_video_preference_ppo.sh
...
```


## License

Apache License 2.0

## Acknowledgments

- [veRL](https://github.com/volcengine/verl): Efficient RL training framework
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo): Video understanding library
- [SASRec](https://github.com/kang205/SASRec): Self-Attentive Sequential Recommendation
