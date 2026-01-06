import cv2
import numpy as np
import json
import os
import glob
import concurrent.futures
import argparse
from tqdm import tqdm
import threading
import random
from datetime import datetime
import time

# ========================================================
# Parameters Configuration (Local Overexposure Tuning)
# ========================================================
# 用户可直接调整的倍率参数
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

# 以 Medium 为基准
MEDIUM_PARAMS = {
    "sigma": (30.0, 60.0),      # 减小扩散半径，让光晕更集中
    "intensity": (1.5, 1.8)     # 减小强度系数，让亮度增加更温和
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            params[k] = (v[0] * factor, v[1] * factor)
        else:
            params[k] = v * factor
    return params

OVEREXPOSURE_LEVELS = {
    "slight": calculate_level_params(MEDIUM_PARAMS, SLIGHT_FACTOR),
    "medium": MEDIUM_PARAMS,
    "severe": calculate_level_params(MEDIUM_PARAMS, SEVERE_FACTOR)
}

# ========================================================
# System Prompt (English Version) - 统一系统提示
# ========================================================
SYSTEM_PROMPT_TEXT = (
    "You are an AI visual expert specializing in nighttime image quality assessment and restoration. "
    "I will provide a nighttime image. Please professionally diagnose and analyze its image quality based on the following six core dimensions:\n\n"
    "1. **Noise**: Evaluate luminance and color noise caused by high ISO, and check for artifacts.\n"
    "2. **Underexposure**: Check if the image is too dark, with crushed shadows losing details.\n"
    "3. **Overexposure**: Check for clipped highlights (e.g., streetlights) and blooming effects.\n"
    "4. **Motion Blur**: Identify local trailing effects caused by fast-moving subjects.\n"
    "5. **Defocus Blur**: Determine if there is an out-of-focus issue affecting the whole subject.\n"
    "6. **Camera Shake**: Identify global, directional blur caused by unstable handheld shooting.\n\n"
    "Please output an objective, structured analysis report based on the user's specific instruction."
)

# ... (QUESTION/ANSWER TEMPLATES)

# ========================================================
# Core Logic
# ========================================================

def find_all_mask_files(mask_dir, label):
    """ 获取所有相关的 mask 文件 """
    search_pattern = os.path.join(mask_dir, f"*{label}*.png")
    matched_files = glob.glob(search_pattern)
    label_underscore = label.replace(' ', '_')
    search_pattern_underscore = os.path.join(mask_dir, f"*{label_underscore}*.png")
    matched_files_underscore = glob.glob(search_pattern_underscore)
    return list(set(matched_files + matched_files_underscore))

def apply_atmospheric_bloom(img, mask, sigma, intensity):
    """
    模拟真实的大气光晕效果
    """
    img_float = img.astype(np.float32) / 255.0
    
    mask_float = mask.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])

    # 提取光源层 (保留光源原始颜色)
    light_source = img_float * mask_float

    # 制造光晕 (Bloom Generation)
    bloom_layer = cv2.GaussianBlur(light_source, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 光学叠加 (Additive Blending)
    img_with_core = img_float + (light_source * 0.5) 
    final_img = img_with_core + (bloom_layer * intensity)

    # 色调映射 (Clipping)
    final_img = np.clip(final_img, 0, 1.0)
    
    return (final_img * 255).astype(np.uint8)

def process_image_overexposure(folder_path, target_label, output_folder, level_name, dataset_results, dataset_lock, sigma, intensity, cmp_mode=False):
    folder_name = os.path.basename(folder_path)
    raw_img_path = os.path.join(folder_path, "raw_image.jpg")
    mask_dir = os.path.join(folder_path, "mask")
    
    if not os.path.exists(raw_img_path) or not os.path.exists(mask_dir):
        return

    mask_files = find_all_mask_files(mask_dir, target_label)
    if not mask_files: return

    img = cv2.imread(raw_img_path)
    if img is None: return
    h, w = img.shape[:2]

    # 合并 Mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for m_path in mask_files:
        m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if m_img is None: continue
        if m_img.shape[:2] != (h, w):
            m_img = cv2.resize(m_img, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_or(combined_mask, m_img)

    if np.sum(combined_mask) == 0: return

    # --- 执行光晕算法 ---
    processed_img = apply_atmospheric_bloom(img, combined_mask, sigma=sigma, intensity=intensity)

    final_image = processed_img
    if cmp_mode:
        # Side-by-side comparison: [Original, Processed]
        h, w, c = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = h / 600.0
        thickness = max(2, int(font_scale * 3))
        header_h = int(h * 0.15)
        header = np.full((header_h, w * 2, c), 255, dtype=np.uint8)
        cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(header, "MODIFIED", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
        main_body = np.hstack([img, processed_img])
        final_image = np.vstack([header, main_body])

    # 保存
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{folder_name}.jpg"
    out_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(out_path, final_image)

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Local Overexposure dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
    parser.add_argument('--main_json', type=str, required=True, help='Path to the main JSON config')
    parser.add_argument('--output_folder', type=str, default='local_overexposure_outputs', help='Folder to save processed images')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of concurrent threads.')
    parser.add_argument('--max_images', type=int, default=None, help='本次任务处理的最大图片数量。')
    parser.add_argument('--dataset_json', type=str, default='local_overexposure_sft.json', help='结果清单文件名。')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium', 
                        help='Overexposure level. If "random", each image will be assigned a random level.')
    parser.add_argument('--cmp', action='store_true', help='If set, saves side-by-side comparison with original image.')

    args = parser.parse_args()

    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' does not exist.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data = json.load(f)

    tasks = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        
        motion = info.get("augmentation_assessment", {}).get("can_add_local_overexposure", {})
        if motion.get("feasible") is True:
            target = motion.get("target_source")
            if target and os.path.isdir(folder_path):
                tasks.append((folder_path, target))

    if args.max_images: tasks = tasks[:args.max_images]

    dataset_results, dataset_lock = [], threading.Lock()
    start_time = time.time()

    print(f"Task Started | Level Mode: {args.level.upper()} | Target Images: {len(tasks)}")
    
    available_levels = ['slight', 'medium', 'severe']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for p, t in tasks:
            current_level = args.level if args.level != 'random' else random.choice(available_levels)
            level_params = OVEREXPOSURE_LEVELS[current_level]
            sigma = random.uniform(*level_params["sigma"])
            intensity = random.uniform(*level_params["intensity"])
            level_output_folder = f"{args.output_folder}/{current_level}"
            
            future = executor.submit(process_image_overexposure, p, t, level_output_folder, current_level, dataset_results, dataset_lock, sigma, intensity, args.cmp)
            futures[future] = (p, t)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing images"):
            pass

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
