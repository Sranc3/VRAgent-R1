import re
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import Counter

def extract_prediction(solution_str: str) -> Tuple[Optional[str], str]:
    """提取模型响应中的最终预测部分。
    
    Args:
        solution_str: 模型的原始响应字符串
        
    Returns:
        包含(提取的预测, 处理后的字符串)的元组
    """
    # 分割响应以隔离助手输出
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] 无法定位模型响应头")
        return None, solution_str

    # 使用XML样式标签提取最终预测
    prediction_pattern = r'<predict>(.*?)</predict>'
    matches = list(re.finditer(prediction_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] 未找到有效的predict标签")
        return None, processed_str
        
    final_prediction = matches[-1].group(1).strip()
    return final_prediction, processed_str

def parse_preferences(text: str) -> List[str]:
    """解析文本中提到的用户偏好。
    
    Args:
        text: 包含用户偏好的文本
        
    Returns:
        提取的偏好标签列表
    """
    preferences = []
    # 尝试从文本中提取关键偏好标签
    # 这里可以使用更复杂的NLP技术，但为简单起见，我们使用一种基于模式匹配的方法
    
    print("\n[偏好解析]")
    
    # 匹配常见的偏好表达方式
    patterns = [
        r'prefers? ([\w\s&]+)',
        r'interested in ([\w\s&]+)',
        r'enjoys? ([\w\s&]+)',
        r'likes? ([\w\s&]+)',
        r'preference for ([\w\s&]+)',
        r'fan of ([\w\s&]+)',
        r'([\w\s&]+) (videos|content|movies|tutorials)',
        r'interest in ([\w\s&]+)',
        r'strong interest in ([\w\s&]+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            pref = match.group(1).strip()
            if len(pref) > 3 and pref not in preferences:  # 避免太短的词和重复
                preferences.append(pref)
                print(f"  找到偏好: {pref}")
    
    return preferences

def parse_recommendation(text: str) -> str:
    """解析文本中的推荐内容。
    
    Args:
        text: 包含推荐的文本
        
    Returns:
        提取的推荐内容
    """
    # 尝试从文本中提取推荐
    print("\n[推荐解析]")
    
    # 匹配常见的推荐表达方式
    patterns = [
        r'recommend (?:watching |)([\w\s&,:\-\'".]+)',
        r'suggestion: ([\w\s&,:\-\'".]+)',
        r'might enjoy ([\w\s&,:\-\'".]+)',
        r'would (?:like|enjoy) ([\w\s&,:\-\'".]+)',
        r'next (?:video|content|course): ([\w\s&,:\-\'".]+)',
        r'recommended next content: ([\w\s&,:\-\'".]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            rec = match.group(1).strip()
            print(f"  找到推荐: {rec}")
            return rec
    
    print("  [警告] 未找到明确的推荐")
    return ""

def validate_response_structure(processed_str: str) -> bool:
    """进行响应结构的全面验证。
    
    Args:
        processed_str: 模型的处理后响应字符串
        
    Returns:
        表示是否满足所有格式要求的布尔值
    """
    print("\n[结构验证]")
    validation_passed = True

    # 检查必需的标签
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'predict_start': ('<predict>', 1),
        'predict_end': ('</predict>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: 出现次数={count}, 位置={pos}")
        
        if count != expected_count:
            print(f"  [错误] {tag_str} 出现 {count} 次 (预期 {expected_count})")
            validation_passed = False

    # 验证标签顺序
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['predict_start'] or
        positions['predict_start'] > positions['predict_end']):
        print("  [错误] 标签顺序不正确: 预期 <think>...</think><predict>...</predict>")
        validation_passed = False
    else:
        print("  标签序列验证通过")

    return validation_passed

def calculate_preference_match(pred_preferences: List[str], gt_preferences: List[str]) -> float:
    """计算预测偏好与真实偏好的匹配度。
    
    Args:
        pred_preferences: 预测的偏好列表
        gt_preferences: 真实的偏好列表
        
    Returns:
        匹配分数 (0-1)
    """
    if not pred_preferences or not gt_preferences:
        return 0.0
    
    # 将偏好转换为小写以进行不区分大小写的比较
    pred_lower = [p.lower() for p in pred_preferences]
    gt_lower = [g.lower() for g in gt_preferences]
    
    # 计算直接匹配的偏好
    direct_matches = set(pred_lower).intersection(set(gt_lower))
    
    # 计算部分匹配的偏好（一个偏好是另一个的子字符串）
    partial_matches = 0
    for pred in pred_lower:
        if pred in direct_matches:
            continue
        for gt in gt_lower:
            if pred in gt or gt in pred:
                partial_matches += 0.5
                break
    
    # 计算总分数 (直接匹配优先于部分匹配)
    score = (len(direct_matches) + 0.5 * partial_matches) / len(gt_preferences)
    
    # 限制最大分数为1.0
    return min(score, 1.0)

def calculate_recommendation_match(pred_rec: str, gt_rec: str) -> float:
    """计算预测推荐与真实推荐的匹配度。
    
    Args:
        pred_rec: 预测的推荐
        gt_rec: 真实的推荐
        
    Returns:
        匹配分数 (0-1)
    """
    if not pred_rec or not gt_rec:
        return 0.0
    
    # 转换为小写以进行不区分大小写的比较
    pred_lower = pred_rec.lower()
    gt_lower = gt_rec.lower()
    
    # 检查直接包含关系
    if pred_lower in gt_lower or gt_lower in pred_lower:
        # 如果一个是另一个的子字符串，给予高分
        return 0.9
    
    # 分词比较
    pred_words = set(re.findall(r'\b\w+\b', pred_lower))
    gt_words = set(re.findall(r'\b\w+\b', gt_lower))
    
    # 计算相似单词
    common_words = pred_words.intersection(gt_words)
    
    # 计算Jaccard相似度
    if not pred_words or not gt_words:
        return 0.0
    
    jaccard = len(common_words) / len(pred_words.union(gt_words))
    
    # 根据Jaccard相似度返回分数
    if jaccard > 0.7:
        return 0.8
    elif jaccard > 0.5:
        return 0.6
    elif jaccard > 0.3:
        return 0.4
    elif jaccard > 0.1:
        return 0.2
    else:
        return 0.0

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 content_reward: float = 2.0) -> float:
    """计算模型响应的综合分数。
    
    Args:
        solution_str: 原始模型响应字符串
        ground_truth: 包含真实数据的字典
        format_reward: 格式正确性的奖励/扣分点数
        content_reward: 内容正确性的奖励/扣分点数
        
    Returns:
        总分 (格式和内容奖励的总和)
    """
    print("\n" + "="*80)
    print(" 处理新样本 ".center(80, '='))
    
    # 解析真实数据
    gt_preferences = ground_truth.get('user_preferences', [])
    gt_recommendation = ground_truth.get('next_recommendation', '')
    print(f"[真实数据] 偏好: {gt_preferences}")
    print(f"[真实数据] 推荐: {gt_recommendation}")

    # 提取模型预测
    prediction_text, processed_str = extract_prediction(solution_str)
    print(f"\n[模型响应]\n{processed_str}")

    # 验证响应结构
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  格式验证: {'通过' if format_correct else '失败'}")
    print(f"  格式分数: {format_score}")

    # 验证内容正确性
    content_score = 0
    if format_correct and prediction_text:
        # 解析预测文本中的偏好和推荐
        pred_preferences = parse_preferences(prediction_text)
        pred_recommendation = parse_recommendation(prediction_text)
        
        # 计算偏好匹配分数
        preference_score = calculate_preference_match(pred_preferences, gt_preferences)
        
        # 计算推荐匹配分数
        recommendation_score = calculate_recommendation_match(pred_recommendation, gt_recommendation)
        
        # 计算总内容分数
        content_score = content_reward * (0.6 * preference_score + 0.4 * recommendation_score)
        
        print(f"\n[内容验证]")
        print(f"  预期偏好: {gt_preferences}")
        print(f"  预测偏好: {pred_preferences}")
        print(f"  偏好匹配分数: {preference_score:.2f}")
        
        print(f"  预期推荐: {gt_recommendation}")
        print(f"  预测推荐: {pred_recommendation}")
        print(f"  推荐匹配分数: {recommendation_score:.2f}")
        
        print(f"  内容分数: {content_score:.2f}")
    else:
        content_score = -1.5
        print("\n[内容验证] 由于格式错误或缺少预测而跳过")

    total_score = format_score + content_score
    print("\n" + "-"*80)
    print(f" 最终分数 ".center(80, '-'))
    print(f"  格式: {format_score}")
    print(f"  内容: {content_score:.2f}")
    print(f"  总分: {total_score:.2f}")
    print("="*80 + "\n")

    return total_score 