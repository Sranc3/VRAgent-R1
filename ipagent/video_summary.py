import torch
from multiprocess import set_start_method
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
from torchvision import transforms
from PIL import ImageOps
from torch.cuda.amp import autocast
import os
import pandas as pd
from PIL import Image

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#model_id  = "lmms-lab/llama3-llava-next-8b"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(model_id,cache_dir = '/data1/share/.HF_cache/',attn_implementation="flash_attention_2", torch_dtype=torch.float16,
                                                        #   device_map="auto"
                                                          ).eval()

prompt = "[INST] <image>\nThis is the cover image of a video, the title of the video is {}." \
         " Based on the information, please concisely summarize the content, including the interest area, key character and the main event." \
         "Just give the key information, do not contain detailed words [/INST]"


df = pd.read_csv('/data2/tencent/MLLM-MSR/MLLM-MSR/data/microlens/MicroLens-50k_titles.csv')
# 通过ID获取title (注意ID从1开始)
df.set_index('item', inplace=True)

# 通过ID获取title
def get_title_by_id(id):
    # 直接使用索引查询
    return df.loc[id]['title']

# 使用示例
title = get_title_by_id(19221)

def add_image_file_path(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example['item_id'] = filename
    print(filename)
    example['item_title'] = get_title_by_id(int(filename))
    #print(example)
    return example

# img_dir = "../../inference_playground/microlens/microlens_50k_subset" #Change this to the real path of the image folder
result_path = '/data2/tencent/MLLM_sum.json'
img_dir = "/data2/tencent/MLLM-MSR/MLLM-MSR/data/microlens/MicroLens-50k/MicroLens-50k_covers"
dataset = load_dataset("imagefolder", data_dir=img_dir)
dataset = dataset.map(lambda x: add_image_file_path(x))
print(dataset)

processor = AutoProcessor.from_pretrained(model_id, return_tensors=torch.float16)


def gpu_computation(batch, rank):
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
            delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    batch['image'] = padded_images

    # Your big GPU call goes here, for example:
    model_inputs = processor([prompt for i in range(len(batch['image']))], batch['image'], batch['item_title'], return_tensors="pt",padding=True).to(device)

    with torch.no_grad() and autocast():
        outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)
    ans = [a.split("[/INST]")[1] for a in ans]
    print(ans)
    return {"summary": ans}

#f.close()

# def analyze_video_content(image_path, title):
#     """
#     分析视频封面和标题，预测视频内容
    
#     参数:
#         image_path (str): 视频封面图片路径
#         title (str): 视频标题
#     返回:
#         str: 视频内容预测
#     """
#     # 设置设备
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 加载模型和处理器
#     model_id = "llava-hf/llava-1.5-7b-hf"
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     processor = AutoProcessor.from_pretrained(model_id)

#     # 加载和处理图像
#     image = Image.open(image_path)
    
#     # 构建提示词
#     prompt = f"""[INST] <image>
#     这是一个视频的封面图片，视频标题是："{title}"
#     请根据封面图片和标题，详细描述这个视频可能的内容。
#     分析要点：
#     1. 视频主题和类型
#     2. 可能包含的具体内容
#     3. 目标受众
#     请用一段连贯的文字描述。[/INST]"""

#     # 处理输入
#     inputs = processor(
#         prompt,
#         image,
#         return_tensors="pt"
#     ).to(device, torch.float16)

#     # 生成预测
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=500,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#         )
    
#     # 解码输出
#     response = processor.decode(outputs[0], skip_special_tokens=True)
#     # 提取回答部分（去除提示词）
#     response = response.split("[/INST]")[1].strip()
    
#     return response


if __name__ == "__main__":
    set_start_method("spawn")
    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=8,
        with_rank=True,
        # num_proc=torch.cuda.device_count(),  # one process per GPU
        num_proc = 4
    )

    train_dataset = updated_dataset['train']
    item_id = train_dataset['item_id']
    summary = train_dataset['summary']
    df = pd.DataFrame({'item_id': item_id, 'summary': summary})
    df.to_csv('image_summary.csv', index=False)

