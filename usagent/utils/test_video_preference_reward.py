#!/usr/bin/env python
"""测试视频偏好预测任务的奖励函数"""

import sys
import os
import json

# 添加父目录到sys.path，确保能够导入verl模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from verl.utils.reward_score.video_preference import compute_score
except ImportError:
    # 如果没有verl模块，直接从当前目录导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from video_preference import compute_score

# 示例模型响应（好的响应）
GOOD_RESPONSE = """<|im_start|>assistant
<think>
Looking at this user's watching history, I notice several patterns:

1. The user has watched multiple videos related to programming and development:
   - "JavaScript for Beginners" (complete video)
   - "Web Development Basics" (45 minutes)
   - "CSS Styling Tutorial" (complete video)
   - "React.js Introduction" (40 minutes)
   - "Building Your First Website" (complete video)

2. The user seems particularly interested in web development technologies, with a focus on:
   - Frontend frameworks (React.js)
   - Core web technologies (JavaScript, CSS)
   - Practical application (Building websites)

3. The user tends to watch programming tutorials completely, showing dedication to learning these topics thoroughly.

4. The user appears to be building their skills progressively, from basic JavaScript to more advanced frameworks like React.

5. There's a clear focus on practical skills that can be applied to building websites rather than theoretical computer science concepts.
</think>

<predict>
(1) User preference summary: Based on your watching history, you have a strong interest in web development and programming, particularly focused on frontend technologies. You've shown consistent interest in JavaScript, CSS, and React.js, and seem to prefer complete, practical tutorials that help you build real websites.

(2) Recommended next content: I would recommend watching "Advanced JavaScript Frameworks" tutorials or a "Full-Stack Web Development Course" that builds on your frontend knowledge and expands into backend technologies. This would be a natural progression from your current interests.
</predict>
<|im_end|>"""

# 示例模型响应（格式错误）
BAD_FORMAT_RESPONSE = """<|im_start|>assistant
I'll analyze your watching history.

Based on your watching history, you have a strong interest in web development and programming, particularly focused on frontend technologies. You've shown consistent interest in JavaScript, CSS, and React.js, and seem to prefer complete, practical tutorials that help you build real websites.

I would recommend watching "Advanced JavaScript Frameworks" tutorials or a "Full-Stack Web Development Course" that builds on your frontend knowledge and expands into backend technologies. This would be a natural progression from your current interests.
<|im_end|>"""

# 示例模型响应（内容相关但不精确）
MEDIOCRE_RESPONSE = """<|im_start|>assistant
<think>
Looking at this user's watching history:
- "JavaScript for Beginners" (complete video)
- "Web Development Basics" (45 minutes)
- "CSS Styling Tutorial" (complete video)
- "React.js Introduction" (40 minutes)
- "Building Your First Website" (complete video)

The user seems to be interested in technology topics, especially around computers and software development. They watch a lot of tutorial content.
</think>

<predict>
(1) User preference summary: You enjoy technology tutorials, particularly related to computer programming. You seem to like learning new digital skills.

(2) Recommended next content: I think you would enjoy some more advanced technology courses, perhaps related to software engineering or computer science fundamentals.
</predict>
<|im_end|>"""

# 示例真实数据
GROUND_TRUTH = {
    "user_preferences": ["Web Development", "Programming", "Frontend Technologies"],
    "next_recommendation": "Advanced JavaScript Frameworks or Full-Stack Development Course"
}

def main():
    print("="*80)
    print(" 视频偏好预测任务奖励函数测试 ".center(80, '='))
    print("="*80)
    
    print("\n\n【测试1：优秀响应】")
    compute_score(GOOD_RESPONSE, GROUND_TRUTH)
    
    print("\n\n【测试2：格式错误响应】")
    compute_score(BAD_FORMAT_RESPONSE, GROUND_TRUTH)
    
    print("\n\n【测试3：内容相关但不精确响应】")
    compute_score(MEDIOCRE_RESPONSE, GROUND_TRUTH)
    
    # 从英文数据集中读取一个真实样本进行测试
    try:
        print("\n\n【测试4：使用真实数据集样本】")
        with open('./data/raw/user_watch_history_english.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i == 7:  # 使用第8个样本 (索引为7的样本)
                    sample = json.loads(line)
                    break
            
        # 构造模拟响应
        mock_response = f"""<|im_start|>assistant
<think>
Analyzing the user's viewing history:
{sample['watch_history']}

This user has clearly shown interest in web development and programming technologies.
</think>

<predict>
(1) User preference summary: Based on your watching history, you have a strong interest in web development, particularly JavaScript and frontend technologies. You appear to be systematically learning website building skills.

(2) Recommended next content: I would recommend "Advanced JavaScript Frameworks Course" or "Backend Web Development with Node.js" to complement your frontend skills.
</predict>
<|im_end|>"""
        
        # 构造真实数据
        gt = {
            "user_preferences": sample['preferences'],
            "next_recommendation": sample['next_recommendation']
        }
        
        compute_score(mock_response, gt)
    except Exception as e:
        print(f"读取数据集测试失败: {e}")

if __name__ == "__main__":
    main() 