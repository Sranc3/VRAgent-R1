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
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
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
    print("\n[偏好解析]")
    
    # 首先尝试提取标准格式的偏好（优先）
    # 支持英文模板的格式: "User_preferences: item1, item2, item3"
    standard_pattern = r'user_preference(?:s)?:?\s*(.+?)(?:\n|$|\.|next_video)'
    standard_matches = re.finditer(standard_pattern, text.lower(), re.DOTALL)
    found_standard = False
    
    for match in standard_matches:
        found_standard = True
        # 分割可能的多个偏好（以逗号或分号分隔）
        raw_prefs = match.group(1).strip()
        for pref in re.split(r'[;,]', raw_prefs):
            clean_pref = pref.strip()
            # 去除可能的编号 (如 "1. Programming" 中的 "1.")
            clean_pref = re.sub(r'^\d+\.\s*', '', clean_pref)
            # 去除引号
            clean_pref = clean_pref.strip('"\'')
            
            if len(clean_pref) > 2 and clean_pref not in preferences:
                preferences.append(clean_pref)
                print(f"  标准格式偏好: {clean_pref}")
    
    # 如果找到了标准格式，直接返回
    if found_standard:
        return preferences
        
    # 否则使用通用模式匹配
    english_patterns = [
        r'interested in ([\w\s&-]+)',
        r'interest in ([\w\s&-]+)',
        r'enjoys? ([\w\s&-]+)',
        r'likes? ([\w\s&-]+)',
        r'preference for ([\w\s&-]+)',
        r'fan of ([\w\s&-]+)',
        r'loves? ([\w\s&-]+)',
        r'passionate about ([\w\s&-]+)',
        r'([\w\s&-]+) (videos|content|movies|tutorials|courses)',
        r'focuses? on ([\w\s&-]+)',
        r'engages? with ([\w\s&-]+)'
    ]
    
    chinese_patterns = [
        r'喜欢([\w\s&\u4e00-\u9fff]+)',
        r'偏好([\w\s&\u4e00-\u9fff]+)',
        r'感兴趣的([\w\s&\u4e00-\u9fff]+)',
        r'观看([\w\s&\u4e00-\u9fff]+)视频',
        r'热衷于([\w\s&\u4e00-\u9fff]+)',
        r'关注([\w\s&\u4e00-\u9fff]+)',
        r'([\w\s&\u4e00-\u9fff]+)爱好者'
    ]
    
    # 合并所有模式
    all_patterns = english_patterns + chinese_patterns
    
    for pattern in all_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            pref = match.group(1).strip()
            if len(pref) > 2 and pref not in preferences:  # 避免太短的词和重复
                preferences.append(pref)
                print(f"  找到非标准偏好: {pref}")
    
    return preferences

def parse_recommendation(text: str) -> str:
    """解析文本中的推荐内容。
    
    Args:
        text: 包含推荐的文本
        
    Returns:
        提取的推荐内容
    """
    print("\n[推荐解析]")
    
    # 首先尝试提取标准格式的推荐（优先）
    # 支持英文模板的格式: "Next_video: recommendation"
    standard_pattern = r'next_video:?\s*(.+?)(?:\n|$|\.|;)'
    standard_match = re.search(standard_pattern, text.lower(), re.DOTALL)
    
    if standard_match:
        rec = standard_match.group(1).strip()
        # 去除可能的引号
        rec = rec.strip('"\'')
        print(f"  标准格式推荐: {rec}")
        return rec
    
    # 没有找到标准格式，尝试通用匹配
    english_patterns = [
        r'recommend(?:ed)? (?:watching |viewing |)([\w\s&,:\-\'".]+)',
        r'suggestion: ([\w\s&,:\-\'".]+)',
        r'might enjoy ([\w\s&,:\-\'".]+)',
        r'would (?:like|enjoy) ([\w\s&,:\-\'".]+)',
        r'next (?:video|content|course): ([\w\s&,:\-\'".]+)',
        r'recommended (?:next |)(?:content|video|course): ([\w\s&,:\-\'".]+)',
        r'should watch ([\w\s&,:\-\'".]+)',
        r'ideal next (?:video|content) (?:is |would be |)([\w\s&,:\-\'".]+)',
        r'predict (?:the user will watch |)([\w\s&,:\-\'".]+)'
    ]
    
    # chinese_patterns = [
    #     r'推荐(?:观看|)([^。；\n]*)',
    #     r'建议(?:观看|)([^。；\n]*)',
    #     r'可以(?:观看|尝试)([^。；\n]*)',
    #     r'接下来(?:可以|应该)(?:观看|学习)([^。；\n]*)',
    #     r'下一个视频(?:应该是|可以是|:)([^。；\n]*)'
    # ]
    
    # 合并所有模式
    all_patterns = english_patterns #+ chinese_patterns
    
    for pattern in all_patterns:
        match = re.search(pattern, text.lower())
        if match:
            rec = match.group(1).strip()
            print(f"  找到非标准推荐: {rec}")
            return rec
    
    print("  [警告] 未找到明确的推荐")
    return ""

def validate_response_structure(processed_str: str) -> Tuple[bool, float]:
    """进行响应结构的全面验证。
    
    Args:
        processed_str: 模型的处理后响应字符串
        
    Returns:
        表示是否满足所有格式要求的布尔值及格式质量分数
    """
    print("\n[结构验证]")
    validation_passed = True
    format_quality_score = 1.0  # 默认满分

    # 检查必需的标签
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: 出现次数={count}, 位置={pos}")
        
        if count != expected_count:
            print(f"  [错误] {tag_str} 出现 {count} 次 (预期 {expected_count})")
            validation_passed = False
            format_quality_score *= 0.5  # 标签缺失降低一半分数

    # 验证标签顺序
    if (
        positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [错误] 标签顺序不正确: 预期 <think>...</think><answer>...</answer>")
        validation_passed = False
        format_quality_score *= 0.5  # 标签顺序错误降低一半分数
    else:
        print("  标签序列验证通过")

    # 检查预测中是否包含了结构化格式
    predict_content = ""
    if positions['answer_start'] >= 0 and positions['answer_end'] >= 0:
        start_idx = positions['answer_start'] + len('<answer>')
        end_idx = positions['answer_end']
        predict_content = processed_str[start_idx:end_idx].strip().lower()
        
        # 检查结构化格式元素 - 支持英文和中文格式
        has_user_preference = bool(re.search(r'(user_preference(?:s)?:|用户(?:偏好|喜好):|偏好:)', predict_content))
        has_next_video = bool(re.search(r'(next_video:|下一个视频:|推荐视频:|视频推荐:)', predict_content))
        
        # 检查是否有编号格式 (1) 和 (2)
        has_numbering = bool(re.search(r'\(1\)|\（1\）|\(2\)|\（2\）', predict_content))
        
        print(f"  结构化格式检查: User_preference标记: {'存在' if has_user_preference else '缺失'}")
        print(f"  结构化格式检查: Next_video标记: {'存在' if has_next_video else '缺失'}")
        print(f"  结构化格式检查: 编号格式: {'存在' if has_numbering else '缺失'}")
        
        # 根据结构化元素的存在情况调整质量分数
        # if not has_user_preference:
        #     format_quality_score *= 0.7
        #     print("  [警告] 缺少User_preference标记")
        if not has_next_video:
            format_quality_score *= 0.7
            print("  [警告] 缺少Next_video标记")
        if not has_numbering:
            format_quality_score *= 0.9
            print("  [警告] 缺少编号格式")
        
    print(f"  格式质量得分: {format_quality_score:.2f}")
    return validation_passed, format_quality_score

def find_matching_video(prediction: str, candidate_videos: List[str]) -> int:
    """在候选视频中找到与预测最匹配的视频索引。
    
    Args:
        prediction: 预测的视频标题
        candidate_videos: 候选视频列表
        
    Returns:
        匹配视频的索引，如果没有找到则返回-1
    """
    # if not prediction or  candidate_videos==None:
    #     return -1
    
    pred_lower = prediction.lower().strip()
    print("模型选择内容：",pred_lower)
    # 首先尝试完全匹配
    for i, video in enumerate(candidate_videos):
        if video.lower().strip() == pred_lower:
            print(f"  完全匹配: 预测选择了候选项 {i}")
            return i
    
    # 如果没有完全匹配，尝试部分匹配（一个是另一个的子字符串）
    for i, video in enumerate(candidate_videos):
        if pred_lower in video.lower().strip() or video.lower().strip() in pred_lower:
            print(f"  部分匹配: 预测选择了候选项 {i}")
            return i
    
    # 如果仍然没有匹配，计算单词重叠度
    best_match = -1
    best_score = 0.0
    
    for i, video in enumerate(candidate_videos):
        video_lower = video.lower().strip()
        
        # 分词比较
        pred_words = set(re.findall(r'\b\w+\b', pred_lower))
        video_words = set(re.findall(r'\b\w+\b', video_lower))
        
        # 计算相似单词
        common_words = pred_words.intersection(video_words)
        
        # 计算Jaccard相似度
        if pred_words and video_words:
            jaccard = len(common_words) / len(pred_words.union(video_words))
            
            if jaccard > 0.3 and jaccard > best_score:
                best_score = jaccard
                best_match = i
    
    if best_match >= 0:
        print(f"  单词重叠匹配: 预测选择了候选项 {best_match} (相似度: {best_score:.2f})")
        return best_match
    
    print("  [警告] 无法匹配预测到任何候选视频")
    return -1

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, any],
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
    correct_index = ground_truth.get('correct_index', -1)
    #print('correct_index',type(correct_index),correct_index)
    candidate_videos = ground_truth.get('candidate_videos', [])
    ground_truth_video = ground_truth.get('next_recommendation', '')
    
    # 验证必要的字段
    if correct_index < 0 : #or not candidate_videos:
        print("[错误] 真实数据中缺少必要的字段 (correct_index 或 candidate_videos)")
        return 0.0
        
    print(f"正确答案索引: {correct_index}")
    print(f"正确视频: {ground_truth_video}")
    print(f"候选视频数量: {len(candidate_videos)}")
    
    # 提取预测，如果extract_prediction函数失败，prediction将为None
    prediction, processed_str = extract_prediction(solution_str)
    #print("处理中.....",prediction, processed_str)
    if prediction is None:
        print("[错误] 无法提取预测内容")
        return 0.0
        
    # 格式验证
    format_valid, format_quality_score = validate_response_structure(processed_str)
    
    # 计算格式分数 - 结合基础验证结果和质量分数
    if format_valid:
        format_score = format_reward * format_quality_score
    else:
        # 即使基本格式不通过，我们仍然给予一定的分数基于格式质量
        format_score = format_reward * format_quality_score * 0.5
    
    print(f"\n格式得分: {format_score:.2f}/{format_reward}")
    
    # 内容评估
    print("\n[内容评估]")
    
    # 解析推荐内容
    if prediction:
        # 从预测中解析推荐
        pred_recommendation = parse_recommendation(prediction)
    else:
        # 格式错误时从整个响应中尝试解析
        pred_recommendation = parse_recommendation(processed_str)
    
    
    #########省略思考过程
    #pred_recommendation = prediction
    ####################################

    # 解析用户偏好（可选，用于理解模型思路）
    pred_preferences = parse_preferences(prediction if prediction else processed_str)
    print(pred_preferences)
    # 内容评估 - 关键部分：计算模型选择的正确性
    content_score = 0.0
    
    # 找出模型选择的视频索引
    predicted_index = find_matching_video(pred_recommendation, candidate_videos)
    
    # 判断是否选择正确
    if predicted_index == correct_index:
        print(f"  模型正确选择了视频! 索引: {predicted_index}")
        content_score = content_reward  # 正确选择，给予满分奖励
    elif predicted_index >= 0:
        print(f"  模型选择了错误的视频。选择: {predicted_index}, 正确: {correct_index}")
        content_score = -content_reward * 0.75  # 错误选择，给予负分惩罚
    else:
        print(f"  模型没有明确选择任何视频")
        content_score = -content_reward  # 未选择，同样给予负分
    
    print(f"内容得分: {content_score:.2f}/{content_reward}")
    
    # 计算总分
    total_score = format_score + content_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score:.2f}")
    print(f"  Content: {content_score:.2f}")
    print(f"  Total: {total_score:.2f}")
    print("="*80 + "\n")
    
    return total_score
    
#