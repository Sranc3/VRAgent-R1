from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import pandas as pd
# min_pixels = 256 * 28 * 28
# max_pixels = 1280 * 28 * 28

# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
import time
model_path = '/data2/zhangxuan/qwen2-vl/qwen2_vl_checkpoint'
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

# result_path = '/data2/zhangxuan/qwen2-vl/qwen2_egoschema_sub7b.json'
# egoschema_path = '/data0/egoschema/EgoSchema-main/subset_anno.json'
# all_anno = json.load(open(egoschema_path))


# result_anno = json.load(open(result_path))
# result_anno_key = {anno['q_uid']: [anno['output'][0].split('Answer: ')[-1][0], anno['qa']['truth']] for anno in result_anno} 
# print(result_anno_key)





# prompt = "[INST] <image>\nThis is the cover image of a video, the title of the video is {}." \

#          #"Based on the information, please concisely summarize the content, including the interest area, key character and the main event." \
#         "If you want to recommend the video, what kind of information" \
#          "Just give the key information, do not translate the chinese words [/INST]"



#result_path = '/data2/tencent/video_sum.json'
#// ... existing code ...

def multi_modal_info_advanced(video_path, image_path, text, title):
    """
    Advanced multi-modal processing function with two stages:
    1. First use title and cover image to determine if information is sufficient
    2. If insufficient and video exists, use video frames
    3. Ensure output content aligns with the title
    
    Args:
    video_path: path to the video file
    image_path: path to the cover image
    text: recommendation request prompt
    title: video title
    
    Returns:
    Dictionary containing judgment result and final output
    """
    # Stage 1: Using only title and cover image
    stage1_prompt = f"This is a cover image of a video, and the title is: {title}. Based only on the cover image and title, determine if there is sufficient information for recommendation. Answer 'Sufficient' or 'Insufficient' and briefly explain why."
    
    # Use cover image for first stage judgment
    messages = [
    [{"role": "user", 
      "content": [
        {"type": "image", "image": image_path}, 
        {"type": "text", "text": stage1_prompt}
        ]
    }]]

    text_s1 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text_s1, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    stage1_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Determine next steps based on first stage result
    if "Sufficient" in stage1_output and os.path.exists(image_path):
        # If first stage determines information is sufficient, use cover image
        second_prompt = f"This is a cover image of a video, and the title is: {title}. {text}"
        messages = [
        [{"role": "user", 
          "content": [
            {"type": "image", "image": image_path}, 
            {"type": "text", "text": second_prompt}
            ]
        }]]
    else:
        # If first stage determines information is insufficient and video exists, use video frames
        if os.path.exists(video_path):
            second_prompt = f"These are key frames from a video, and the title is: {title}. Pay special attention to content related to the title. {text}"
            messages = [
            [{"role": "user", 
              "content": [
                {"type": "video", 
                 "video": video_path,
                 "nframes": 5,  # Only take 5 frames
                 "max_pixels": 360 * 420
                }, 
                {"type": "text", "text": second_prompt}
                ]
            }]]
        else:
            # If video doesn't exist, still use cover image
            second_prompt = f"This is a cover image of a video, and the title is: {title}. Despite limited information, try to extract content related to the title as much as possible. {text}"
            messages = [
            [{"role": "user", 
              "content": [
                {"type": "image", "image": image_path}, 
                {"type": "text", "text": second_prompt}
                ]
            }]]
    
    # Stage 2: Final generation based on judgment result
    text_s2 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text_s2, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    final_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Return dictionary with both stage results
    return {
        "stage1_judgment": stage1_output,
        "final_output": final_output[0]
    }








def multi_modal_info(video_path, image_path, text):
    # 检查视频文件是否存在
    if os.path.exists(video_path):
        # 使用视频
        messages = [
        [{"role": "user", 
          "content": [
            {"type": "video", 
             "video": video_path,
             "nframes": 5,  # 总共只取5帧
             "max_pixels": 360 * 420
            }, 
            {"type": "text", "text": text}
            ]
        }]]
    else:
        # 如果视频不存在，使用图像
        messages = [
        [{"role": "user", 
          "content": [
            {"type": "image", "image": image_path}, 
            {"type": "text", "text": text}
            ]
        }]]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs ,padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    # print(inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
  

image_root = "/data2/tencent/MicroLens-100k/MicroLens-100k_covers/"
video_root = "/data2/tencent/MLLM-MSR_v1/MLLM-MSR/data/microlens/"
title_path = "/data2/tencent/MicroLens-50k_titles.json"
used_video_title = '/data2/RL/data/raw/used_video_titles_part2.json'
extra_title_path = '/data2/tencent/extra.json'
title_anno = json.load(open(used_video_title))   

#result_path = '/data2/tencent/MLLM_sum_v2_extra.json'
result_path = '/data2/tencent/MLLM_sum_used_video_part2.json'



request = "If you want to recommend the video, based on the visual and text information, summarize the information as the fomat: Interest area: XXX; key characters: XXX; main event: XXX "\
          "Do not translate the Chinese words "

request2 = "You should first understand the title with visual information, and refine the text title, use brief words "\
            "here are some good examples: "\
            "example 1: 'the girl just drove the wrong way, but did not expect to encounter terrible things # thriller movie # movie commentary' "\
            "example 2: 'a bear is chasing a little sheep, indicating the problem of child safety  #science # Child safety' " \
            "limit the words in 35"

request3 = "Analyze the video content through these three steps:\n"\
          "1. Content Understanding:\n"\
          "   - Identify what visual content aligns with or extends beyond the title's description\n"\
          "   - Note any discrepancies or additional context provided by the visuals\n"\
          "2. Key Information Extraction, including: Main Characters, Core Event and Emotional Appeal\n"\
          "3. Tag Analysis and Refinement:\n"\
          "   - Review existing tags from the title\n"\
          "   - Verify if current tags accurately represent the content\n"\
          "   - Suggest additional or alternative tags if needed\n\n"\
          "Create a concise summary (within 35 words) following this format:\n"\
          "'[Core Content Description]  + [Refined Tags]'\n\n"\
          "Examples:\n"\
          "1. 'Young girl's wrong turn leads to suspenseful encounter. Gripping cautionary tale for drivers. #thriller #roadsafety #suspense'\n"\
          "2. 'Heartwarming parent-child cooking adventure. Fun family bonding with educational value. #familytime #kidscooking #parenting'\n\n"\
          "Focus on accuracy and engagement while maintaining brevity."

#'这里设计一下多模态思维链'
#第一步，看标题和封面图，能够得到充足的信息
#足够的话，请总结，限制词数30左右




result_anno = []
for i, data_dict in enumerate(title_anno):
    id = data_dict['id']
    title = data_dict['title']
    image_path = image_root + id +  '.jpg'
    video_path =  video_root + id +  '.mp4'
    video_content = "Given the visual frames of a video, and the title of the video is :{}".format(title)
    #video_title = "the title of the video is :{}".format(title)
    question = f"{video_content}\n"  + f"{request3}"
    result = multi_modal_info(video_path, image_path ,question)
    data_dict['MLLM_sum'] = result
    result_anno.append(data_dict)
    print(i,  data_dict)

    with open(result_path, 'w') as f:
        json.dump(result_anno, f)


# for i, data_dict in enumerate(result_anno):
#     pre_res = result_anno_key[data_dict['q_uid']]
#     if ord(pre_res[0]) - ord('A') == pre_res[1]:
#         continue
#     begin = time.time()
#     time_instruciton = f"The video lasts for 180 seconds, and 180 frames are uniformly sampled from it. Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 5, where 1 indicates low confidence and 5 signifies high confidence. \n The Answer format is:\n Answer: xx\n The confidence score is: xx\n The reason is: xx\n"
#     option = "Option:\nA: {}\nB: {}\nC: {}\nD: {}\nE: {}".format(data_dict['qa']["option 0"], data_dict['qa']["option 1"],data_dict['qa']["option 2"],data_dict['qa']["option 3"],data_dict['qa']["option 4"])
#     question = f"{time_instruciton}\n" + f"Question: {data_dict['qa']['question']}\n" + f"{option}"
#     video_path = os.path.join('/data0/egoschema/videos', data_dict['q_uid'] + '.mp4')
#     data_dict['output'] = get_answer(video_path, question)
#     question = 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Describe this video in detail.'
#     data_dict['caption'] = get_answer(video_path, question)
#     # result_anno.append(result)
#     end = time.time()
#     print(i,  data_dict)
#     print("need {}s".format(end - begin))
    
    