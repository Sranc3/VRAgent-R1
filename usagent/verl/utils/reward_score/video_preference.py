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
    

    all_patterns = english_patterns 
    
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
        r'ideal next (?:video|content) (?:is |would be |)([\w\s&,:\-\'".]+)'
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

def validate_response_structure(processed_str: str) -> bool:
    """进行响应结构的全面验证。
    
    Args:
        processed_str: 模型的处理后响应字符串
        
    Returns:
        表示是否满足所有格式要求的布尔值
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
    if (positions['think_start'] > positions['think_end'] or
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
        if not has_user_preference:
            format_quality_score *= 0.7
            print("  [警告] 缺少User_preference标记")
        if not has_next_video:
            format_quality_score *= 0.7
            print("  [警告] 缺少Next_video标记")
        if not has_numbering:
            format_quality_score *= 0.9
            print("  [警告] 缺少编号格式")
        
    print(f"  格式质量得分: {format_quality_score:.2f}")
    return validation_passed, format_quality_score

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
    

def calculate_embedding_similarity(pred_txt, gt_txt, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """使用预训练模型计算文本嵌入相似度。"""
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    
    # 加载模型
    model = SentenceTransformer(model_name)
    
    # 计算嵌入向量
    pred_emb = model.encode(pred_txt, convert_to_tensor=True)
    gt_emb = model.encode(gt_txt, convert_to_tensor=True)
    
    # 余弦相似度
    similarity = F.cosine_similarity(pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
    
    return similarity

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
    # 错误处理：确保函数始终返回有效分数，即使处理过程中出现异常
    
    print("\n" + "="*80)
    print(" 处理新样本 ".center(80, '='))
        
        # 确保ground_truth是字典类型
        # if not isinstance(ground_truth, dict):
        #     print(f"[严重错误] ground_truth 不是字典类型: {type(ground_truth)}")
        #     # 尝试转换为字典，如果是字符串可能是JSON
        #     if isinstance(ground_truth, str):
        #         try:
        #             import json
        #             ground_truth = json.loads(ground_truth)
        #         except:
        #             print("[严重错误] 无法将ground_truth字符串转换为字典")
        #             return 0.0
        #     else:
        #         # 无法处理，返回默认分数
        #         return 0.0
        
        # 解析真实数据，使用安全的get方法并设置默认值
    #gt_preferences = ground_truth['user_preferences'].tolist()
    gt_recommendation = ground_truth.get('next_recommendation', '')
        
        # 确保gt_preferences是列表类型
    # if not isinstance(gt_preferences, list):
    #     print(f"[错误] user_preferences 不是列表类型: {type(gt_preferences)}")
    #     if isinstance(gt_preferences, str):
    #         gt_preferences = [gt_preferences]
    #     else:
    #         gt_preferences = []
        
        # 如果缺少训练所需的数据，提前返回
    if  not gt_recommendation:
        print("[错误] 真实数据中缺少必要的字段 (next_recommendation)")
        return 0.0
        
    #print(f"真实偏好: {gt_preferences}")
    print(f"真实推荐: {gt_recommendation}")
    
    # 提取预测，如果extract_prediction函数失败，prediction将为None
    prediction, processed_str = extract_prediction(solution_str)
    
    # 如果没有提取到预测，给予最低分
    if prediction is None:
        print("[错误] 无法提取预测内容")
        return 0.0
        
    # 格式验证 - 使用增强的格式验证函数
    format_valid, format_quality_score = validate_response_structure(processed_str)
    
    # 计算格式分数 - 结合基础验证结果和质量分数
    if format_valid:
        format_score = format_reward * format_quality_score
    else:
        # 即使基本格式不通过，我们仍然给予一定的分数基于格式质量
        format_score = format_reward * format_quality_score * 0.5
    
    print(f"\n格式得分: {format_score:.2f}/{format_reward}")
    
    # 内容评估 (即使格式不正确，我们也尝试评估内容)
    # 使用predict标签或整个响应进行偏好分析
    print("\n[内容评估]")
    if prediction:
        # 从预测中解析偏好和推荐
        #pred_preferences = parse_preferences(prediction)
        pred_recommendation = parse_recommendation(prediction)
    else:
        # 格式错误时从整个响应中尝试解析
        #pred_preferences = parse_preferences(processed_str)
        pred_recommendation = parse_recommendation(processed_str)
    
    # 计算偏好匹配度
    pref_score = 0.0
    # if gt_preferences:
    #     try:
    #         pref_score = calculate_preference_match(pred_preferences, gt_preferences)
    #         print(f"偏好匹配得分: {pref_score:.2f}")
    #     except Exception as e:
    #         print(f"[错误] 计算偏好匹配时出错: {str(e)}")
    
    # 计算推荐匹配度
    rec_score = 0.0
    if gt_recommendation:
        # try:
        #     rec_score = calculate_recommendation_match(pred_recommendation, gt_recommendation)
        #     print(f"推荐匹配得分: {rec_score:.2f}")
        # except Exception as e:
        #     print(f"[错误] 计算推荐匹配时出错: {str(e)}")
        sim_score = calculate_embedding_similarity(pred_recommendation, gt_recommendation)
        if sim_score>0.5:
            rec_score = sim_score
        else:
            rec_score = -0.75
    
    # 计算内容总分 - 如果使用了规范格式，奖励更高
    # content_multiplier = 1.0
    # if bool(re.search(r'user_preference(?:s)?:', prediction.lower())) and \
    #     bool(re.search(r'next_video:', prediction.lower())):
    #     # 使用了标准英文标记，给予额外奖励
    #     content_multiplier = 1.2
    #     print("内容格式加成: 使用了标准英文标记 (+20%)")
    
    # if gt_preferences and gt_recommendation:
    #     # 两者都有时，取平均
    #     content_score = content_multiplier * (pref_score + rec_score) / 2 * content_reward
    # elif gt_preferences:
    #     # 只有偏好时
    #     content_score = content_multiplier * pref_score * content_reward
    # elif gt_recommendation:
    #     # 只有推荐时
    #     content_score = content_multiplier * rec_score * content_reward
    # else:
    #     content_score = 0
    content_score = rec_score * content_reward
        
    print(f"内容得分: {content_score:.2f}/{content_reward}")
    
    # 计算总分
    total_score = format_score + content_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {content_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    #print(f"\n总分: {total_score:.2f}/{format_reward + content_reward}")
    
    return total_score
    
# except Exception as e:
#     # 捕获所有可能的异常，确保函数不会崩溃
#     import traceback
#     print(f"[严重错误] 计算分数时出现未处理的异常: {str(e)}")
#     print(traceback.format_exc())
#     # 出现异常时返回0分
#     return 0.0 