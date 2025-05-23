import math
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from datasets import load_dataset
import ast
import json
import os
from  carbontracker.tracker import CarbonTracker
import time



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODE = 'direct'
SETTING = 'standard (4 options)'

import yaml
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1,8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def import_dataset():
    # Load the dataset (replace 'MMMU-Pro' with the correct name if available)
    dataset = load_dataset("MMMU/MMMU_Pro", SETTING, split="test")

    # Take a small sample of 5 examples (adjust as necessary)
    small_sample = dataset.select(range(2))

    # Inspect the sample
    print(small_sample)
    for sample in small_sample:
        print(sample['image_2'])
    return small_sample

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
def process_prompt(data):
    if SETTING == 'standard':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)
   
    return images, prompt
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def convert_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


data_models = ["microsoft/kosmos-2-patch14-224"]
sample = import_dataset()
print(sample['question'], sample['image_1'])

# Iterate over model names and print the number of parameters
for idx_model,m in enumerate(data_models[:]):
    model_name = m


    
    print(model_name)
    # try:
    #     model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #         model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    #     )
    #     model.to(device)

    #     processor = AutoProcessor.from_pretrained(model_name)
    # except:
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True).to('cuda')
        #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        # For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_name)
        # For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
        #model = model.to(device='cuda', dtype=torch.float16)
        # For Mac with MPS (Apple silicon or AMD GPUs).
        # Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
        #model = model.to(device='mps', dtype=torch.float16)

        # tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left',trust_remote_code=True)
    except:
        raise(0)
    num_params = sum(p.numel() for p in model.parameters())
    # Dynamically go through components of the pipeline
    energies = []
    
    prompt = [s['question'] for s in sample]

    image = [s['image_1'] for s in sample]
    print(image)

    
    print(image, prompt)
    # inputs = processor(text=prompt, return_tensors="pt", padding=True)
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    #image_tensor = model.process_images(image, model.config).to(dtype=model.dtype, device='cuda')

    # inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda')
    print(image, prompt)
    inputs = processor(image, prompt,padding=True, truncation=True, return_tensors="pt").to("cuda", torch.float16)

    # Tokenize the input prompt
    for _ in range(10):
        
        torch.cuda.empty_cache()
        time.sleep(5)
        st = time.time()
        tracker = CarbonTracker(epochs=1, update_interval=1, verbose=2, components="all")
        tracker.epoch_start()
        #print(data)
        try:
            for i in range(5):
                with torch.no_grad():
                    
                    generated_ids = model.generate(
                        **inputs,
                        #images=image_tensor,
                        max_new_tokens=2000,  # Generate a long response
                        num_beams=5,  # Force beam search (slower)
                        repetition_penalty=2.0,  # Encourage varied text generation
                        length_penalty=2.0,  # Encourage longer output
                    )
            # msgs = [{'role': 'user', 'content': prompt[0]}]
            # # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print(image[0])
            # res, context, _ = model.chat(
            # image=image[0],
            # msgs=msgs,
            # context=None,
            # tokenizer=tokenizer,
            # sampling=True,
            # temperature=0.7
            # )
            # # Specify `cleanup_and_extract=False` in order to see the raw model generation.
            # processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

        except:
            raise(0)
        ed = time.time()
        timing, energy, divided = tracker.epoch_end()
        print
        divided = [float(d) for d in divided]
        energies.append({"tim": timing, "energy":energy, "divided":divided, "timestamp_start":st, "timestamp_end":ed})
    info = {'num_params':num_params,'energies':energies}
#top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
print(f" Number of parameters: {num_params}")