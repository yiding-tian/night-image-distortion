import cv2
import numpy as np
import os
import time
import concurrent.futures
import argparse
import threading
import json
import random
import albumentations as A  # 用于添加真实的 ISO 噪点
from tqdm import tqdm

# ========================================================
# Parameters Configuration (Moderate Underexposure)
# ========================================================

# 调整倍率 (Slight 更亮, Severe 更暗)
SLIGHT_FACTOR = 0.5  
SEVERE_FACTOR = 1.5

# 基准参数 (Medium)
MEDIUM_PARAMS = {
    "brightness_factor": (0.4, 0.7),      # 核心修改：保留 40% 到 70% 的亮度 (中等欠曝)
    "noise_probability": (0.3, 0.6),      # 30%-60% 概率出现噪点
    "iso_noise_color_shift": (0.05, 0.2), # 噪点颜色偏移 (模拟彩噪)
    "iso_noise_intensity": (0.2, 0.4)     # 噪点强度
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            # 对于亮度因子，Severe 意味着更暗 (Factor 变小)
            # Slight 意味着更亮 (Factor 变大)
            if k == "brightness_factor":
                if factor > 1.0: # Severe
                    params[k] = (max(0.1, v[0] / factor), max(0.1, v[1] / factor))
                else:            # Slight
                    params[k] = (min(1.0, v[0] / factor), min(1.0, v[1] / factor))
            else:
                # 噪点参数正常缩放
                params[k] = (min(v[0] * factor, 1.0), min(v[1] * factor, 1.0))
        else:
            params[k] = v
    return params

UNDEREXPOSURE_LEVELS = {
    "slight": calculate_level_params(MEDIUM_PARAMS, SLIGHT_FACTOR),
    "medium": MEDIUM_PARAMS,
    "severe": calculate_level_params(MEDIUM_PARAMS, SEVERE_FACTOR)
}

# ========================================================
# System Prompt & SFT Templates (CoT Technical Version)
# ========================================================
SYSTEM_PROMPT_TEXT = (
    "You are a professional AI visual expert. Provide a structured Chain-of-Thought (CoT) diagnosis for nighttime images. "
    "Your analysis must follow this exact sequence: \n"
    "1. Scene Description: Describe the objects and environment.\n"
    "2. Quality Issue Detection: Determine if a distortion exists.\n"
    "3. Distribution Scope: Classify as Global or Local.\n"
    "4. Distortion Specifics: Identify the type and the specific affected target.\n"
    "5. Severity Level: Assess the intensity of the degradation."
)

QUESTION_TEMPLATES = [
    "Analyze this nighttime image using a step-by-step technical diagnosis. First, describe the scene contents. Then, detect any image quality issues, determine their scope (Global/Local), specify the distortion type and affected target, and finally assess the severity.",
    "Please perform a Chain-of-Thought assessment of this photo. 1. What are the main objects? 2. Is there a technical defect? 3. Is the defect global or local? 4. What is the specific distortion and where is it located? 5. What is the severity level?",
    "Examine the image quality. Start by describing the scene, then provide a structured report covering: Issue Detection, Spatial Scope, Type & Target identification, and Severity Level.",
    "Conduct a technical analysis: Scene Description -> Issue Detection -> Distribution Scope -> Distortion & Target Specifics -> Severity Level. Ensure each step is addressed in order.",
    "As a visual expert, identify the elements in this scene and diagnose its quality. Is there a problem? Is it Global or Local? Name the specific distortion/target and evaluate the severity level."
]

def get_clean_desc(visual_analysis):
    import re
    match = re.search(r'(?i)Defect', visual_analysis)
    if match:
        return visual_analysis[:match.start()].strip().rstrip('.,; ')
    return visual_analysis.strip()

# CoT 结构化回答模板
ANSWER_TEMPLATE = (
    "Technical Analysis (CoT):\n"
    "1. Scene Description: {scene_desc}.\n"
    "2. Quality Issue Detection: Yes, a technical degradation is identified.\n"
    "3. Distribution Scope: {scope}.\n"
    "4. Distortion Specifics: The {type} {target_info}.\n"
    "5. Severity Level: {level}."
)

# ========================================================
# Core Logic: Linear Dimming + ISO Noise
# ========================================================

def apply_moderate_underexposure(img, brightness_factor, noise_prob, iso_color, iso_intensity):
    """
    1. 注入噪点 (模拟高感光度)
    2. 线性降低亮度 (模拟快门/光圈进光量减少)
    """
    img_processed = img.copy()

    # --- Step 1: 概率性注入 ISO 噪点 ---
    if np.random.rand() < noise_prob:
        try:
            rgb_image = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            transform = A.ISONoise(
                color_shift=(iso_color, iso_color),
                intensity=(iso_intensity, iso_intensity),
                always_apply=True
            )
            result = transform(image=rgb_image)
            img_processed = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 如果 albumentations 报错，忽略噪点继续处理亮度
            pass

    # --- Step 2: 线性降低亮度 (Linear Dimming) ---
    img_float = img_processed.astype(np.float32) / 255.0
    
    # 直接乘以系数
    darkened_img = img_float * brightness_factor
    
    # 截断并转回 uint8
    final_img = np.clip(darkened_img * 255, 0, 255).astype(np.uint8)
    
    return final_img

def process_single_folder(folder_path, output_folder, level_name, params, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    folder_name = os.path.basename(folder_path)
    raw_img_path = os.path.join(folder_path, "raw_image.jpg")
    
    if not os.path.exists(raw_img_path):
        return

    img = cv2.imread(raw_img_path)
    if img is None: return

    output_filename = f"{folder_name}.jpg"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    # 随机取样参数
    b_factor = random.uniform(*params["brightness_factor"])
    n_prob = random.uniform(*params["noise_probability"])
    n_color = random.uniform(*params["iso_noise_color_shift"])
    n_intensity = random.uniform(*params["iso_noise_intensity"])

    # 处理图像
    processed_img = apply_moderate_underexposure(img, b_factor, n_prob, n_color, n_intensity)

    # 对比图处理
    final_image = processed_img
    if cmp_mode:
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = h / 800.0
        header_h = int(h * 0.12)
        header = np.full((header_h, w * 2, 3), 255, dtype=np.uint8)
        
        info_text = f"UNDEREXPOSED ({int(b_factor*100)}% Brightness)"
        if n_prob > np.random.rand(): 
            info_text += " + NOISE"

        cv2.putText(header, "ORIGINAL", (int(w * 0.3), int(header_h * 0.7)), font, font_scale, (0, 0, 0), 2)
        cv2.putText(header, info_text, (int(w * 1.1), int(header_h * 0.7)), font, font_scale, (0, 0, 255), 2)
        
        final_image = np.vstack([header, np.hstack([img, processed_img])])

    # 保存
    cv2.imwrite(output_path, final_image)

    # 生成符合 CoT 逻辑的对话内容
    scene_desc = get_clean_desc(visual_analysis)
    instruction_text = random.choice(QUESTION_TEMPLATES)
    
    answer_text = ANSWER_TEMPLATE.format(
        scene_desc=scene_desc,
        scope="Global",
        type="Underexposure",
        target_info="is uniformly distributed across the entire dark scene",
        level=level_name.capitalize()
    )

    entry = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEXT
            },
            {
                "role": "user",
                "content": f"{instruction_text}\n<image>"
            },
            {
                "role": "assistant",
                "content": answer_text
            }
        ],
        "images": [
            output_path 
        ]
    }

    with dataset_lock:
        dataset_results.append(entry)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate Moderate Underexposure with Noise")
    parser.add_argument('--base_folder', type=str, required=True, help='数据集根目录')
    parser.add_argument('--main_json', type=str, required=True, help='Path to dataset_with_prompt.json')
    parser.add_argument('--output_folder', type=str, default='underexposure_results')
    parser.add_argument('--dataset_json', type=str, default='underexposure_sft.json', help='结果清单文件名。')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--cmp', action='store_true', help='开启对比图模式')

    args = parser.parse_args()

    if not os.path.exists(args.base_folder):
        print(f"Error: Base directory not found at '{args.base_folder}'")
        return

    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' not found.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    if isinstance(main_data_raw, list):
        main_data = {item['filename']: item for item in main_data_raw}
    else:
        main_data = main_data_raw

    subfolders_info = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        if os.path.isdir(folder_path):
            subfolders_info.append((folder_path, info.get("visual_analysis", "")))

    if args.max_images:
        subfolders_info = subfolders_info[:args.max_images]
    
    print(f"🚀 任务启动 | 模式: 中度低曝 + 噪点 | 文件夹数量: {len(subfolders_info)}")

    available_levels = ['slight', 'medium', 'severe']
    dataset_results = []
    dataset_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for folder, visual_analysis in subfolders_info:
            current_level = args.level if args.level != 'random' else random.choice(available_levels)
            params = UNDEREXPOSURE_LEVELS[current_level]
            
            level_out_dir = os.path.join(args.output_folder, current_level)

            future = executor.submit(process_single_folder, folder, level_out_dir, current_level, params, dataset_results, dataset_lock, visual_analysis, args.cmp)
            futures[future] = folder

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(subfolders_info), desc="Processing"):
            pass

    if dataset_results:
        with open(args.dataset_json, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Dataset saved to: {args.dataset_json}")

if __name__ == "__main__":
    main()