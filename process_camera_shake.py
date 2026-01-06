import cv2
import numpy as np
import os
import time
import concurrent.futures
import albumentations as A
import argparse
import json
import threading
from tqdm import tqdm
import random
from datetime import datetime

# ========================================================
# Parameters Configuration (Camera Shake Tuning)
# ========================================================
# 用户可直接调整的倍率参数
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

# 以 Medium 为基准
MEDIUM_PARAMS = {
    "blur_limit": (40, 60),
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            # 缩放范围，并进行合理的边界限制 (对于 MotionBlur, blur_limit 至少为 3)
            low = max(3, int(v[0] * factor))
            high = max(3, int(v[1] * factor))
            params[k] = (low, high)
        else:
            params[k] = v * factor
    return params

SHAKE_LEVELS = {
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

QUESTION_TEMPLATES = [
    "Analyze the image quality and describe any camera shake issues at a {level} level.",
    "Does this photo show signs of {level} camera shake? Please evaluate the blur.",
    "I suspect this image has {level} camera shake. Can you confirm and describe it?",
    "Conduct a technical assessment of the {level} camera shake degradation in this photo."
]

ANSWER_TEMPLATES = [
    "Upon inspection, this image is affected by {level} camera shake, resulting in a global directional blur.",
    "The analysis shows clear evidence of {level} camera shake, which has caused a uniform blur across the image.",
    "There is visible {level} camera shake in this picture, categorized as a global directional degradation.",
    "I detected {level} camera shake artifacts. The blur pattern suggests unstable handheld shooting."
]

# ========================================================
# Core Logic
# ========================================================

def apply_camera_shake(folder_path, output_folder, level_name, dataset_results, dataset_lock, cmp_mode=False):
    """
    进入图片对应的文件夹，寻找 raw_image.jpg 并应用 Camera Shake (MotionBlur)。
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
        blur_limit = SHAKE_LEVELS[level_name]["blur_limit"]
        transform = A.MotionBlur(
            blur_limit=blur_limit,
            angle_range=(0, 360), # 模拟任意方向的抖动
            allow_shifted=True,
            p=1.0
        )
        
        result = transform(image=rgb_image)
        shaken_rgb_image = result["image"]

        final_image = cv2.cvtColor(shaken_rgb_image, cv2.COLOR_RGB2BGR)
        
        if cmp_mode:
            # Side-by-side comparison: [Original, Processed]
            h, w, c = raw_image.shape
            
            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            
            # Create a white header
            header = np.full((header_h, w * 2, c), 255, dtype=np.uint8)
            
            # Labels
            cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MODIFIED", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            
            # Combine
            main_body = np.hstack([raw_image, final_image])
            final_image = np.vstack([header, main_body])
            
        cv2.imwrite(output_path, final_image)

        # 生成对话内容 (暂不生成SFT)
        # ... (rest of commented out code)
            
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Camera Shake dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
    parser.add_argument('--output_folder', type=str, default='camera_shake_outputs', help='Folder to save shaken images')
    parser.add_argument('--max_images', type=int, default=None, help='本次任务处理的最大图片数量。')
    parser.add_argument('--num_threads', type=int, default=4, help='并行线程数。')
    parser.add_argument('--dataset_json', type=str, default='camera_shake_sft.json', help='结果清单文件名。')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium', 
                        help='Shake level. If "random", each image will be assigned a random level.')
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
                apply_camera_shake, 
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
