import cv2
import numpy as np
import os
import time
import concurrent.futures
import albumentations as A
import argparse
import json
import threading
import random
from tqdm import tqdm
from datetime import datetime

# ========================================================
# Parameters Configuration (Noise Tuning)
# ========================================================
# 用户可直接调整的倍率参数
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

# 以 Medium 为基准
MEDIUM_PARAMS = {
    "color_shift": (0.1, 0.3),
    "intensity": (0.3, 0.5),
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            # 针对不同参数的特殊限制
            low = min(v[0] * factor, 1.0)
            high = min(v[1] * factor, 1.0)
            params[k] = (low, high)
        else:
            params[k] = min(v * factor, 1.0)
    return params

NOISE_LEVELS = {
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
# Core Logic
# ========================================================

def apply_high_iso_noise(folder_path, output_folder, level_name, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    """
    进入图片对应的文件夹，寻找 raw_image.jpg 并应用高 ISO 噪声 (ISONoise)。
    """
    folder_name = os.path.basename(folder_path)
    raw_image_path = os.path.join(folder_path, 'raw_image.jpg')
    
    if not os.path.exists(raw_image_path):
        return

    output_filename = f"{folder_name}.jpg"
    # Ensure the output folder for the current level exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    raw_image = cv2.imread(raw_image_path)
    if raw_image is None:
        return

    try:
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        level_params = NOISE_LEVELS[level_name]
        color_shift = random.uniform(*level_params["color_shift"])
        intensity = random.uniform(*level_params["intensity"])
        
        transform = A.ISONoise(
            color_shift=(color_shift, color_shift),
            intensity=(intensity, intensity),
            always_apply=True
        )
        
        result = transform(image=rgb_image)
        noisy_rgb_image = result["image"]

        final_image = cv2.cvtColor(noisy_rgb_image, cv2.COLOR_RGB2BGR)
        
        if cmp_mode:
            # Side-by-side comparison: [Original, Processed]
            h, w, c = raw_image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            header = np.full((header_h, w * 2, c), 255, dtype=np.uint8)
            cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MODIFIED", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            main_body = np.hstack([raw_image, final_image])
            final_image = np.vstack([header, main_body])
            
        cv2.imwrite(output_path, final_image)

        # [MODIFIED] 生成符合 CoT 逻辑的对话内容
        scene_desc = get_clean_desc(visual_analysis)
        instruction_text = random.choice(QUESTION_TEMPLATES)
        
        answer_text = ANSWER_TEMPLATE.format(
            scene_desc=scene_desc,
            scope="Global",
            type="ISO Noise",
            target_info="is uniformly distributed across the entire sensor output",
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
            
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime ISO Noise dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
    parser.add_argument('--main_json', type=str, required=True, help='Path to the main JSON config (e.g., dataset_with_prompt.json)')
    parser.add_argument('--output_folder', type=str, default='iso_noise_outputs', help='Folder to save noisy images')
    parser.add_argument('--max_images', type=int, default=None, help='本次任务处理的最大图片数量。')
    parser.add_argument('--num_threads', type=int, default=4, help='并行线程数。')
    parser.add_argument('--dataset_json', type=str, default='iso_noise_sft.json', help='结果清单文件名。')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium', 
                        help='Noise level. If "random", each image will be assigned a random level.')
    parser.add_argument('--cmp', action='store_true', help='If set, saves side-by-side comparison with original image.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_folder):
        print(f"Error: Base folder '{args.base_folder}' does not exist.")
        return

    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' does not exist.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    if isinstance(main_data_raw, list):
        main_data = {item['filename']: item for item in main_data_raw}
    else:
        main_data = main_data_raw

    all_subdirs_info = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        if os.path.isdir(folder_path):
            all_subdirs_info.append((folder_path, info.get("visual_analysis", "")))

    target_subdirs_info = all_subdirs_info[:args.max_images] if args.max_images else all_subdirs_info

    if not target_subdirs_info:
        print("Error: No subdirectories found to process.")
        return

    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()

    print(f"Task Started | Level Mode: {args.level.upper()} | Target Images: {len(target_subdirs_info)}")
    
    available_levels = ['slight', 'medium', 'severe']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for subdir, visual_analysis in target_subdirs_info:
            current_level = args.level if args.level != 'random' else random.choice(available_levels)
            level_output_folder = f"{args.output_folder}/{current_level}"
            
            future = executor.submit(
                apply_high_iso_noise, 
                subdir, 
                level_output_folder, 
                current_level, 
                dataset_results, 
                dataset_lock,
                visual_analysis,
                args.cmp
            )
            futures[future] = subdir
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_subdirs_info), desc="Processing images"):
            pass

    # [MODIFIED] Save dataset to JSON file (re-enabled)
    # [MODIFIED] Save dataset to JSON file (符合用户样例：扁平列表格式)
    if dataset_results:
        with open(args.dataset_json, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Dataset saved to: {args.dataset_json}")

if __name__ == '__main__':
    main()
