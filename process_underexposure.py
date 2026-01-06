import cv2
import numpy as np
import os
import time
import concurrent.futures
import argparse
import threading
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
# 1. brightness_factor: 亮度保留比例 (0.5 表示亮度减半)
#    - 调高这个值 (如 0.6-0.8) 会让画面更亮
#    - 调低这个值 (如 0.3-0.5) 会让画面更暗
# 2. noise_params: 噪点相关设置 (保留原设)
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
# Core Logic: Linear Dimming + ISO Noise
# ========================================================

def apply_moderate_underexposure(img, brightness_factor, noise_prob, iso_color, iso_intensity):
    """
    1. 注入噪点 (模拟高感光度)
    2. 线性降低亮度 (模拟快门/光圈进光量减少)
    """
    img_processed = img.copy()

    # --- Step 1: 概率性注入 ISO 噪点 ---
    # 这会给暗部增加颗粒感，增加真实度
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
    # 使用浮点数运算，避免精度丢失
    img_float = img_processed.astype(np.float32) / 255.0
    
    # 直接乘以系数 (例如 * 0.6)
    # 这样暗部变暗，亮部也变暗，整体直方图左移，不会丢失对比度细节
    darkened_img = img_float * brightness_factor
    
    # 截断并转回 uint8
    final_img = np.clip(darkened_img * 255, 0, 255).astype(np.uint8)
    
    return final_img

def process_single_folder(folder_path, output_folder, level_name, params, cmp_mode=False):
    folder_name = os.path.basename(folder_path)
    raw_img_path = os.path.join(folder_path, "raw_image.jpg")
    
    if not os.path.exists(raw_img_path):
        return

    img = cv2.imread(raw_img_path)
    if img is None: return

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
        
        # 显示保留的亮度百分比
        info_text = f"UNDEREXPOSED ({int(b_factor*100)}% Brightness)"
        if n_prob > np.random.rand(): # 只是为了显示，不代表实际逻辑
            info_text += " + NOISE"

        cv2.putText(header, "ORIGINAL", (int(w * 0.3), int(header_h * 0.7)), font, font_scale, (0, 0, 0), 2)
        cv2.putText(header, info_text, (int(w * 1.1), int(header_h * 0.7)), font, font_scale, (0, 0, 255), 2)
        
        final_image = np.vstack([header, np.hstack([img, processed_img])])

    # 保存
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, f"{folder_name}.jpg"), final_image)

def main():
    parser = argparse.ArgumentParser(description="Generate Moderate Underexposure with Noise")
    parser.add_argument('--base_folder', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_folder', type=str, default='moderate_underexposure_outputs')
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--cmp', action='store_true', help='开启对比图模式')
    
    args = parser.parse_args()

    if not os.path.exists(args.base_folder):
        print(f"Error: Base directory not found at '{args.base_folder}'")
        return

    subfolders = [f.path for f in os.scandir(args.base_folder) if f.is_dir()]
    if args.max_images:
        subfolders = subfolders[:args.max_images]
    
    print(f"🚀 任务启动 | 模式: 中度低曝 + 噪点 | 文件夹数量: {len(subfolders)}")

    available_levels = ['slight', 'medium', 'severe']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for folder in subfolders:
            current_level = args.level if args.level != 'random' else random.choice(available_levels)
            params = UNDEREXPOSURE_LEVELS[current_level]
            
            level_out_dir = os.path.join(args.output_folder, current_level)
            
            future = executor.submit(process_single_folder, folder, level_out_dir, current_level, params, args.cmp)
            futures[future] = folder

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(subfolders), desc="Processing"):
            pass

    print(f"\n✨ 处理完成。结果保存在: {args.output_folder}")

if __name__ == "__main__":
    main()