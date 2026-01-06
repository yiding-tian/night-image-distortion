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

def apply_high_iso_noise(folder_path, output_folder, level_name, dataset_results, dataset_lock, cmp_mode=False):
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
            
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime ISO Noise dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
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
            current_level = args.level if args.level != 'random' else random.choice(available_levels)
            level_output_folder = f"{args.output_folder}/{current_level}"
            
            future = executor.submit(
                apply_high_iso_noise, 
                subdir, 
                level_output_folder, 
                current_level, 
                dataset_results, 
                dataset_lock,
                args.cmp
            )
            futures[future] = subdir
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_subdirs), desc="Processing images"):
            pass

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
