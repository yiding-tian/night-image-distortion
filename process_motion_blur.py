import argparse
import concurrent.futures
import json
import os
import glob
import threading
import time
import random
import math
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ========================================================
# Parameters Configuration (Motion Blur Tuning)
# ========================================================
# 用户可直接调整的倍率参数
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

# --- Device Configuration (Auto-detect GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 以 Medium 为基准
MEDIUM_PARAMS = {
    # [优化] 增加采样步数，让模糊效果更稠密、更细腻，显著提升“糊”的质感
    "steps": (100, 150),
    # [优化] 降低基础系数，配合非线性计算逻辑，减小细小物体的拖影
    "intensity": (0.15, 0.35),
    # 垂直偏移范围锁定在极小值，保证高度一致
    "vertical_shift_factor": (-0.015, 0.015),
    # [优化] 大幅降低噪点强度，让模糊看起来更丝滑、更干净
    "noise_range": (0.01, 0.03),
    # 关闭波动逻辑
    "curve_freq": 0.0,
    "curve_amp": 0.0
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            if k == "steps":
                params[k] = (max(20, int(v[0] * factor)), max(20, int(v[1] * factor)))
            elif k == "intensity":
                params[k] = (v[0] * factor, v[1] * factor)
            elif k == "noise_range":
                params[k] = (v[0] * factor, v[1] * factor)
            else:
                params[k] = v
        else:
            params[k] = v * factor
    return params

MOTION_BLUR_LEVELS = {
    "slight": calculate_level_params(MEDIUM_PARAMS, SLIGHT_FACTOR),
    "medium": MEDIUM_PARAMS,
    "severe": calculate_level_params(MEDIUM_PARAMS, SEVERE_FACTOR)
}

# ========================================================
# Core Logic
# ========================================================

def find_motion_masks(mask_dir, target_label):
    if not os.path.exists(mask_dir): return []
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files: return []

    orig = target_label.lower()
    variants = [orig, orig.replace(" ", "_"), orig.replace(" ", "")]
    words = orig.split()
    if len(words) > 1:
        variants.append(words[-1])
        variants.append(words[0])

    matched_paths = []
    for variant in variants:
        for f in mask_files:
            if variant in f.lower():
                matched_paths.append(os.path.join(mask_dir, f))
    return list(set(matched_paths))

def calculate_adaptive_magnitude(w, h, intensity, level_name):
    """
    [核心修改] 自适应物体大小计算移动距离：
    1. 使用非线性缩放：细小物体的位移进一步压低。
    2. 针对大物体：稍微放宽上限，使其拖影可见但不过长。
    """
    diagonal = np.sqrt(w**2 + h**2)
    
    # 基础位移
    base_magnitude = diagonal * intensity
    
    # 根据等级设定动态上限
    if level_name == "slight":
        max_limit = 25.0
    elif level_name == "medium":
        max_limit = 55.0  # 稍微从 45 调高到 55
    else:
        max_limit = 95.0  # 稍微从 80 调高到 95
        
    MIN_SHIFT_PX = 1.5

    # 逻辑：对于小物体，应用一个衰减系数
    if diagonal < 200:
        # 物体越小，衰减越厉害
        damp = max(0.4, diagonal / 200.0)
        base_magnitude *= damp

    # 逻辑：对于大物体，使用软上限
    if base_magnitude > max_limit:
        final_magnitude = max_limit + (base_magnitude - max_limit) * 0.12
    else:
        final_magnitude = base_magnitude

    final_magnitude = max(MIN_SHIFT_PX, final_magnitude)
    
    # 限制最大不超过物体自身最小尺寸的 50%
    limit_ratio = 0.5
    final_magnitude = min(final_magnitude, min(w, h) * limit_ratio)

    return final_magnitude

def refine_mask_edges(binary_mask_np):
    mask_uint8 = binary_mask_np if binary_mask_np.dtype == np.uint8 else (binary_mask_np * 255).astype(np.uint8)
    mask_soft = cv2.GaussianBlur(mask_uint8, (21, 21), 0)
    return mask_soft.astype(np.float32) / 255.0

def add_matched_noise(tensor, intensity=0.05):
    noise = torch.randn_like(tensor) * intensity
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0.0, 1.0)

def apply_physical_motion_blur_torch(image_np, mask_np, steps, trans_x, trans_y, noise_level=0.05):
    """ 
    物理光学流运动模糊 (GPU 加速)
    """
    img_input = image_np[..., ::-1].copy()
    img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 线性空间转换 (Gamma 2.2)
    img_tensor = torch.pow(img_tensor + 1e-6, 2.2)

    B, C, H, W = img_tensor.shape
    y_base, x_base = torch.meshgrid(
        torch.linspace(-1, 1, H, device=DEVICE),
        torch.linspace(-1, 1, W, device=DEVICE),
        indexing='ij'
    )
    base_grid = torch.stack((x_base, y_base), dim=-1).unsqueeze(0)

    accumulated_img = torch.zeros_like(img_tensor)
    accumulated_mask = torch.zeros_like(mask_tensor)
    total_weight = 0.0

    # 计算每一步的位移量
    dx_step = trans_x / steps
    dy_step = trans_y / steps

    for i in range(steps):
        # 线性插值路径
        cur_pixel_dx = dx_step * i
        cur_pixel_dy = dy_step * i

        cur_tx = cur_pixel_dx * 2 / W
        cur_ty = cur_pixel_dy * 2 / H
        
        grid_t = base_grid.clone()
        grid_t[..., 0] -= cur_tx
        grid_t[..., 1] -= cur_ty

        # 采样图像和掩码
        warped_img = F.grid_sample(img_tensor, grid_t, align_corners=True, padding_mode="border")
        warped_mask = F.grid_sample(mask_tensor, grid_t, align_corners=True, mode="bilinear", padding_mode="zeros")

        # 模拟快门均匀开启过程，权重设为 1.0
        weight = 1.0 

        accumulated_img += warped_img * warped_mask * weight
        accumulated_mask += warped_mask * weight
        total_weight += weight

    accumulated_mask[accumulated_mask < 1e-3] = 1e-3
    blurred_object = accumulated_img / accumulated_mask
    
    # 转回 sRGB 空间
    blurred_object = torch.pow(blurred_object + 1e-6, 1/2.2)
    img_orig_gamma = torch.pow(img_tensor + 1e-6, 1/2.2)
    
    # 注入微量噪点（已调低）
    blurred_object = add_matched_noise(blurred_object, intensity=noise_level)

    # 混合原图与模糊层
    final_mask = torch.clamp(accumulated_mask / (total_weight + 1e-6), 0, 1)
    output_tensor = blurred_object * final_mask + img_orig_gamma * (1 - final_mask)
    
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[..., ::-1]

def process_single_directory(subdir, target_label, output_folder, level_name, current_level_params, dataset_results, dataset_lock, cmp_mode=False):
    folder_name = os.path.basename(subdir)
    image_path = os.path.join(subdir, "raw_image.jpg")
    mask_dir = os.path.join(subdir, "mask")
    
    if not os.path.exists(image_path): return

    matched_paths = find_motion_masks(mask_dir, target_label)
    if not matched_paths: return

    image = cv2.imread(image_path)
    if image is None: return
    h_img, w_img = image.shape[:2]
    
    mask_info = []
    for m_path in matched_paths:
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h_img, w_img):
                 m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            area = np.sum(m > 0)
            if area > 0:
                mask_info.append({"mask": m, "area": area, "path": m_path})
    
    if not mask_info: return
    mask_info.sort(key=lambda x: x["area"], reverse=True)
    selected_masks = mask_info[:2]
    
    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for info in selected_masks:
        combined_mask = cv2.bitwise_or(combined_mask, info["mask"])

    tqdm.write(f"[INFO] Processing '{folder_name}' | Target: {target_label} | Selected: {len(selected_masks)}")

    coords = cv2.findNonZero(combined_mask)
    if coords is None: return
    x, y, w, h = cv2.boundingRect(coords)
    
    # 1. 计算自适应位移距离 (应用新逻辑)
    magnitude = calculate_adaptive_magnitude(w, h, current_level_params["intensity"], level_name)

    # 2. 随机化运动参数
    noise_val = random.uniform(*current_level_params["noise_range"])
    direction_x = random.choice([-1, 1])
    direction_y = random.choice([-1, 1])
    
    trans_x = magnitude * direction_x
    trans_y = magnitude * current_level_params["vertical_shift_factor"][1] * direction_y

    try:
        # 3. 增强掩码边缘
        soft_mask = refine_mask_edges(combined_mask)
        
        # 4. 执行高步数物理模糊
        final_img = apply_physical_motion_blur_torch(
            image, 
            soft_mask, 
            steps=current_level_params["steps"], 
            trans_x=trans_x,
            trans_y=trans_y, 
            noise_level=noise_val
        )
        
        if cmp_mode:
            h_v, w_v, c_v = image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h_v / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h_v * 0.15)
            header = np.full((header_h, w_v * 2, c_v), 255, dtype=np.uint8)
            cv2.putText(header, f"ORIGINAL ({target_label})", (int(w_v * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MOTION BLUR", (int(w_v * 1.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            main_body = np.hstack([image, final_img])
            final_img = np.vstack([header, main_body])

        output_filename = f"{folder_name}.jpg"
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_path, final_img)
        
    except Exception as e:
        tqdm.write(f"[ERROR] {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Motion Blur dataset.")
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--main_json", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_folder", type=str, default="motion_blur_outputs")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium')
    parser.add_argument('--cmp', action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' not found.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data = json.load(f)

    tasks = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        aug = info.get("augmentation_assessment", {})
        motion = aug.get("can_add_motion_blur", {})
        if motion.get("feasible") is True and os.path.isdir(folder_path):
            target = motion.get("target_object")
            if target: tasks.append((folder_path, target))

    if args.max_images: tasks = tasks[:args.max_images]
    dataset_results, dataset_lock = [], threading.Lock()
    start_time = time.time()
    print(f"Task Started | Level Mode: {args.level.upper()} | Target Images: {len(tasks)}")
    available_levels = ['slight', 'medium', 'severe']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for p, t in tasks:
            current_level_name = args.level if args.level != 'random' else random.choice(available_levels)
            level_params = MOTION_BLUR_LEVELS[current_level_name]
            steps_val = random.randint(*level_params["steps"])
            intensity_val = random.uniform(*level_params["intensity"])
            current_params = {
                'steps': steps_val, 
                'intensity': intensity_val, 
                'vertical_shift_factor': level_params["vertical_shift_factor"],
                'noise_range': level_params['noise_range']
            }
            level_output_folder = f"{args.output_folder}/{current_level_name}"
            future = executor.submit(process_single_directory, p, t, level_output_folder, current_level_name, current_params, dataset_results, dataset_lock, args.cmp)
            futures[future] = (p, t)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing images"):
            pass
    print(f"\n--- Processing Finished --- | Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
