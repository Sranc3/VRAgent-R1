# 视频偏好预测任务数据预处理

此脚本用于预处理用户视频观看历史数据，准备用于训练大语言模型来分析用户偏好并预测下一步推荐内容。

## 数据格式

### 输入数据格式 (JSONL)

输入数据应为JSONL格式，每行包含一个用户的观看历史记录。示例：

```json
{
  "user_id": "user_001",
  "watch_history": "1月1日: 《如何建立个人理财计划》(财经教育) 观看时长: 15分钟\n1月2日: 《股市入门指南》(财经教育) 观看时长: 25分钟...",
  "preferences": ["财经教育", "编程教程", "实用知识"],
  "next_recommendation": "投资理财进阶课程"
}
```

字段说明：
- `user_id`: 用户唯一标识
- `watch_history`: 用户观看历史的文本描述
- `preferences`: 用户偏好标签列表（用于评估）
- `next_recommendation`: 下一步推荐内容（用于评估）

### 输出数据格式

脚本将生成处理后的训练集和测试集，保存为Parquet格式。处理后的数据结构为：

```
{
  "data_source": "video_preference",
  "prompt": [{
    "role": "user",
    "content": "包含模板和用户历史的提示"
  }],
  "ability": "preference_prediction",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "user_preferences": ["标签1", "标签2", ...],
      "next_recommendation": "推荐内容"
    }
  },
  "extra_info": {
    "split": "train/test",
    "index": 数字索引,
    "user_id": "用户ID"
  }
}
```

## 使用方法

### 安装依赖

首先安装必要的依赖：

```bash
pip install datasets tqdm
```

### 准备数据

1. 准备符合上述格式的JSONL文件，存放在`./data/raw/`目录下

### 运行脚本

```bash
python video.py --data_path ./data/raw/user_watch_history.jsonl --local_dir ./data/processed/video_preference
```

### 参数说明

- `--local_dir`: 处理后数据的本地保存路径 (默认: "./data/video_preference/instruct")
- `--hdfs_dir`: HDFS存储路径 (可选)
- `--data_path`: 原始数据路径 (默认: "./data/raw/user_watch_history.jsonl")
- `--train_size`: 训练集大小 (默认: 900)
- `--test_size`: 测试集大小 (默认: 100)
- `--template_type`: 模板类型，可选 "base" 或 "qwen-instruct" (默认: "qwen-instruct")

## 预期输出

运行成功后，脚本将在指定的本地目录生成两个文件：
- `train.parquet`: 训练集数据
- `test.parquet`: 测试集数据

如果指定了HDFS路径，数据也会被复制到HDFS中。

## 模型输出格式

使用此数据集训练的模型预期输出为：

```
<think>
分析用户观看习惯的思考过程...
</think>
<predict>
(1) 用户偏好总结: ...
(2) 推荐的下一步内容: ...
</predict>
```

## 示例用例

以下是一个简单的示例，展示如何使用该脚本处理少量数据：

```bash
# 处理4条示例数据，分为3条训练数据和1条测试数据
python video.py --data_path ./data/raw/user_watch_history.jsonl --local_dir ./data/test_run --train_size 3 --test_size 1
```

## 后续使用

处理后的数据可以用于：

1. 训练大语言模型识别用户偏好
2. 评估模型对用户观看习惯的分析能力
3. 测试模型的内容推荐能力

可以根据实际需求扩展数据集大小和复杂度，添加更多观看类型和用户行为模式。 