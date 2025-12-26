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
    # TODO: 上面很多没有用的格式要不要扣掉？
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
        print("predict_content {}".format(predict_content))
        like_video = bool(re.search(r'(Yes?|yes?|like?|No?|no?)', predict_content))
        
        # 检查是否有编号格式 (1) 和 (2)
        
        print(f"  结构化格式检查: like_video标记: {'存在' if like_video else '缺失'}")
        
        # 根据结构化元素的存在情况调整质量分数

        if not like_video:
            format_quality_score *= 0.7
            print("  [警告] 缺少Next_video标记")

    print(f"  格式质量得分: {format_quality_score:.2f}")
    return validation_passed, format_quality_score






def compute_score(solution_str: str,
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 content_reward: float = 2.0) -> float:
    """计算模型响应的综合分数。
    
    Args:
        solution_str: 原始模型响应字符串
        ground_truth: 包含真实数据的字典
        format_reward: 格式正确性的奖励/扣分点数
        content_reward: 内容正确性的奖励/扣分点数（用户喜欢为正分，不喜欢为负分）
        
    Returns:
        总分 (格式和内容奖励的总和)
    """
    # 错误处理：确保函数始终返回有效分数，即使处理过程中出现异常
    
    print("\n" + "="*80)
    print(" 处理新样本 ".center(80, '='))
    # 获取真实标签（用户是否喜欢：1表示喜欢，0表示不喜欢）
    gt_user_like = ground_truth.get('label', None)
    
    # 如果缺少训练所需的数据，提前返回
    if gt_user_like is None:
        print("[错误] 真实数据中缺少必要的字段 (user_like)")
        return 0.0
        
    print(f"真实标签 (用户喜欢): {gt_user_like}")
    
    # 提取预测，如果extract_prediction函数失败，prediction将为None
    # solution_str <|im_start|>system
    prediction, processed_str = extract_prediction(solution_str)
    # No </answer><|endoftext|>
    print('prediction {}'.format(prediction))
    
    
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
    print("\n[内容评估]")
    
    # 从预测中解析用户喜好标签（1表示喜欢，0表示不喜欢）
    try:
        # 尝试从预测中提取用户喜好标签
        pred_user_like = None
        
        # 首先尝试查找明确的标签
        like_pattern = r'user_like:?\s*(\d+)'
        like_match = re.search(like_pattern, prediction.lower())
        
        if like_match:
            pred_user_like = int(like_match.group(1))
        else:
            # 尝试从文本中推断
            if re.search(r'enjoy|yes', prediction.lower()):
                pred_user_like = 1
            elif re.search(r'dislike|no', prediction.lower()):
                pred_user_like = 0
        
        print(f"预测标签 (用户喜欢): {pred_user_like}")
        
        # 计算内容得分
        if pred_user_like is not None and pred_user_like == gt_user_like:
            # 预测正确，获得正的content_reward
            content_score = content_reward
            print(f"预测正确！获得正分: +{content_reward}")
        else:
            # 预测错误，获得负的content_reward
            content_score = -content_reward
            print(f"预测错误！获得负分: -{content_reward}")
    except Exception as e:
        print(f"[错误] 解析预测标签时出错: {str(e)}")
        content_score = -content_reward  # 出错时给予负分
    
    print(f"内容得分: {content_score:.2f}/{content_reward}")
    
    # 计算总分
    total_score = format_score + content_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {content_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")
    
    return total_score
    
# except Exception as e:
#     # 捕获所有可能的异常，确保函数不会崩溃
#     import traceback
#     print(f"[严重错误] 计算分数时出现未处理的异常: {str(e)}")
#     print(traceback.format_exc())
#     # 出现异常时返回0分
#     return 0.0 