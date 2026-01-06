import cv2
import numpy as np
import os
import time
import concurrent.futures
import argparse
import json
import threading
import random
from datetime import datetime
from tqdm import tqdm

# ========================================================
# Parameters Configuration (Overexposure Tuning)
# ========================================================

# 用户可直接调整的倍率参数
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

# 以 Medium 为基准
MEDIUM_PARAMS = {
    "gamma": (0.6, 0.8),             # 提高 gamma 值，提亮效果更温和
    "brightness_add": (60, 80),      # 减小全局亮度加值
    "bloom_threshold": (180, 200),    # 提高阈值，只让极亮区域产生光晕
    "bloom_radius": (71, 121)        # 减小光晕半径，扩散更自然
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            if k == "gamma":
                params[k] = (max(0.1, v[0] / factor), max(0.1, v[1] / factor))
            elif k == "bloom_threshold":
                params[k] = (max(50, int(v[0] / factor)), max(50, int(v[1] / factor)))
            elif k == "brightness_add":
                params[k] = (min(255, int(v[0] * factor)), min(255, int(v[1] * factor)))
            elif k == "bloom_radius":
                params[k] = (max(3, int(v[0] * factor)), max(3, int(v[1] * factor)))
            else:
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
# Core Logic
# ========================================================

def apply_realistic_overexposure(folder_path, output_folder, level_name, dataset_results, dataset_lock, gamma, brightness_add, bloom_threshold, bloom_radius, cmp_mode=False):
    """
    进入图片对应的文件夹，寻找 raw_image.jpg 并应用基于 LAB 空间的曝光增强和光晕效果。
    """
    folder_name = os.path.basename(folder_path)
    raw_image_path = os.path.join(folder_path, 'raw_image.jpg')
    
    if not os.path.exists(raw_image_path):
        return

    output_filename = f"{folder_name}.jpg"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    raw_image = cv2.imread(raw_image_path)
    if raw_image is None:
        return

    # --- 步骤 1: 转换到 LAB 空间并分离通道 ---
    lab_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # --- 步骤 2: 创建光晕 (Bloom) 遮罩 ---
    _, bloom_mask_sharp = cv2.threshold(l_channel, bloom_threshold, 255, cv2.THRESH_BINARY)
    bloom_radius = bloom_radius if bloom_radius % 2 != 0 else bloom_radius + 1
    bloom_effect_mask = cv2.GaussianBlur(bloom_mask_sharp, (bloom_radius, bloom_radius), 0)

    # --- 步骤 3: 提亮 L 通道 ---
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_gamma_corrected = cv2.LUT(l_channel, gamma_table)
    
    l_brightened = cv2.add(l_gamma_corrected, np.array([brightness_add], dtype=np.uint8))
    l_final = cv2.add(l_brightened, bloom_effect_mask)

    # --- 步骤 4: 合并通道并转回 BGR ---
    final_lab_image = cv2.merge([l_final, a_channel, b_channel])
    final_image = cv2.cvtColor(final_lab_image, cv2.COLOR_LAB2BGR)

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

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Overexposure dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
    parser.add_argument('--output_folder', type=str, default='overexposure_outputs', help='Folder to save processed images')
    parser.add_argument('--num_threads', type=int, default=4, help='并行线程数。')
    parser.add_argument('--dataset_json', type=str, default='overexposure_sft.json', help='结果清单文件名。')
    parser.add_argument('--max_images', type=int, default=None, help='本次任务处理的最大图片数量。')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium', 
                        help='Overexposure level. If "random", each image will be assigned a random level.')
    parser.add_argument('--cmp', action='store_true', help='If set, saves side-by-side comparison with original image.')

    args = parser.parse_args()

    if not os.path.exists(args.base_folder):
        print(f"Error: Base folder '{args.base_folder}' does not exist.")
        return

    all_subdirs = [d.path for d in os.scandir(args.base_folder) if d.is_dir()]
    target_subdirs = all_subdirs[:args.max_images] if args.max_images else all_subdirs

    if not target_subdirs:
        print("Error: No subdirectories found to process.")
        return

    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()

    print(f"Task Started | Level Mode: {args.level.upper()} | Target Images: {len(target_subdirs)}")
    
    available_levels = ['slight', 'medium', 'severe']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for subdir in target_subdirs:
            current_level_name = args.level if args.level != 'random' else random.choice(available_levels)
            level_params = OVEREXPOSURE_LEVELS[current_level_name]

            gamma_val = random.uniform(*level_params["gamma"])
            brightness_add_val = random.randint(*level_params["brightness_add"])
            bloom_threshold_val = random.randint(*level_params["bloom_threshold"])
            bloom_radius_val = random.randint(*level_params["bloom_radius"])

            level_output_folder = f"{args.output_folder}/{current_level_name}"

            future = executor.submit(
                apply_realistic_overexposure, subdir, level_output_folder, current_level_name, 
                dataset_results, dataset_lock, gamma_val, brightness_add_val, 
                bloom_threshold_val, bloom_radius_val, args.cmp
            )
            futures[future] = subdir
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_subdirs), desc="Processing images"):
            pass

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
