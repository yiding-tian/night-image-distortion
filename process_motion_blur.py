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
    "steps": (60, 90),
    "intensity": (0.7, 1.0),
    "vertical_shift_factor": (0.2, 0.3),
    "noise_range": (0.08, 0.12),      # 明显的夜景噪点
    "curve_freq": (1.5, 2.5),
    "curve_amp": (0.08, 0.15)
}

def calculate_level_params(base_params, factor):
    params = {}
    for k, v in base_params.items():
        if isinstance(v, tuple):
            if k == "steps":
                params[k] = (max(10, int(v[0] * factor)), max(10, int(v[1] * factor)))
            else:
                params[k] = (v[0] * factor, v[1] * factor)
        else:
            params[k] = v * factor
    return params

MOTION_BLUR_LEVELS = {
    "slight": calculate_level_params(MEDIUM_PARAMS, SLIGHT_FACTOR),
    "medium": MEDIUM_PARAMS,
    "severe": calculate_level_params(MEDIUM_PARAMS, SEVERE_FACTOR)
}

# ... (SYSTEM PROMPT, TEMPLATES)

# ========================================================
# Core Logic
# ========================================================

# ... (Helper functions: calculate_adaptive_magnitude, refine_mask_edges, add_matched_noise, apply_physical_motion_blur_torch)

def calculate_adaptive_magnitude(w, h, intensity, level_name):
    """
    自适应物体大小计算移动距离
    """
    diagonal = np.sqrt(w**2 + h**2)
    
    # 根据等级调整上限
    if level_name == "slight":
        max_limit = 40.0
    elif level_name == "medium":
        max_limit = 90.0
    else:
        max_limit = 180.0
        
    MIN_SHIFT_PX = 5.0

    # 1. 基础线性计算
    base_magnitude = diagonal * intensity

    # 3. 混合限制逻辑 (Soft Clamping)
    if base_magnitude > max_limit:
        final_magnitude = max_limit + (base_magnitude - max_limit) * 0.15
    else:
        final_magnitude = base_magnitude

    # 4. 安全边界检查
    final_magnitude = max(MIN_SHIFT_PX, final_magnitude)
    final_magnitude = min(final_magnitude, min(w, h) * 0.9)

    return final_magnitude

def refine_mask_edges(binary_mask_np):
    """ 
    增强边缘羽化
    """
    mask_uint8 = binary_mask_np if binary_mask_np.dtype == np.uint8 else (binary_mask_np * 255).astype(np.uint8)
    mask_soft = cv2.GaussianBlur(mask_uint8, (25, 25), 0)
    return mask_soft.astype(np.float32) / 255.0

def add_matched_noise(tensor, intensity=0.05):
    """
    噪点重注
    """
    noise = torch.randn_like(tensor) * intensity
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0.0, 1.0)

def apply_physical_motion_blur_torch(image_np, mask_np, steps, trans_x, trans_y, 
                                     noise_level=0.05, curve_freq=1.0, curve_amp=0.05):
    """ 
    物理光学流运动模糊 (GPU 加速) + 物理增强 (曲线/噪点)
    """
    img_input = image_np[..., ::-1].copy()
    img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Linear Space Conversion
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

    move_len = math.sqrt(trans_x**2 + trans_y**2) + 1e-6
    norm_x = trans_x / move_len
    norm_y = trans_y / move_len
    perp_x = -norm_y
    perp_y = norm_x

    # Simulate Integration
    for i in range(steps):
        t = (i + 1) / steps
        
        linear_dx = trans_x * t
        linear_dy = trans_y * t
        
        sine_offset = math.sin(t * math.pi * curve_freq * 2) * (move_len * curve_amp)
        
        cur_pixel_dx = linear_dx + perp_x * sine_offset
        cur_pixel_dy = linear_dy + perp_y * sine_offset

        cur_tx = cur_pixel_dx * 2 / W
        cur_ty = cur_pixel_dy * 2 / H
        
        grid_t = base_grid.clone()
        grid_t[..., 0] -= cur_tx
        grid_t[..., 1] -= cur_ty

        warped_img = F.grid_sample(img_tensor, grid_t, align_corners=True, padding_mode="border")
        warped_mask = F.grid_sample(mask_tensor, grid_t, align_corners=True, mode="bilinear", padding_mode="zeros")

        weight = 1.0 

        accumulated_img += warped_img * warped_mask * weight
        accumulated_mask += warped_mask * weight
        total_weight += weight

    accumulated_mask[accumulated_mask < 1e-3] = 1e-3
    blurred_object = accumulated_img / accumulated_mask
    
    # Gamma Encode back to sRGB
    blurred_object = torch.pow(blurred_object + 1e-6, 1/2.2)
    img_orig_gamma = torch.pow(img_tensor + 1e-6, 1/2.2)
    
    # 注入噪点
    blurred_object = add_matched_noise(blurred_object, intensity=noise_level)

    # Blend
    final_mask = torch.clamp(accumulated_mask / (total_weight + 1e-6), 0, 1)
    output_tensor = blurred_object * final_mask + img_orig_gamma * (1 - final_mask)
    
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[..., ::-1]

def process_single_directory(subdir, target_label, output_folder, level_name, current_level_params, dataset_results, dataset_lock, cmp_mode=False):
    """ Process a single folder task """
    folder_name = os.path.basename(subdir)
    image_path = os.path.join(subdir, "raw_image.jpg")
    mask_dir = os.path.join(subdir, "mask")
    
    if not os.path.exists(image_path): 
        return

    mask_pattern = os.path.join(mask_dir, f"*{target_label}*.png")
    matched_masks = glob.glob(mask_pattern)
    
    if not matched_masks: 
        return

    image = cv2.imread(image_path)
    if image is None: 
        return
        
    h_img, w_img = image.shape[:2]
    
    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    valid_masks_count = 0

    for m_path in matched_masks:
        m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h_img, w_img):
                 m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.bitwise_or(combined_mask, m)
            valid_masks_count += 1
            
    if valid_masks_count == 0:
        return

    coords = cv2.findNonZero(combined_mask)
    if coords is None: 
        return
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 自适应计算移动距离
    magnitude = calculate_adaptive_magnitude(
        w, h, 
        current_level_params["intensity"], 
        level_name
    )

    # 随机化参数
    noise_val = random.uniform(*current_level_params["noise_range"])
    curve_freq_val = random.uniform(*current_level_params["curve_freq"])
    curve_amp_val = random.uniform(*current_level_params["curve_amp"])

    try:
        soft_mask = refine_mask_edges(combined_mask)
        
        final_img = apply_physical_motion_blur_torch(
            image, 
            soft_mask, 
            steps=current_level_params["steps"], 
            trans_x=magnitude, 
            trans_y=magnitude * current_level_params["vertical_shift_factor"],
            noise_level=noise_val,
            curve_freq=curve_freq_val,
            curve_amp=curve_amp_val
        )
        
        if cmp_mode:
            # Side-by-side comparison: [Original, Processed]
            h, w, c = image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            header = np.full((header_h, w * 2, c), 255, dtype=np.uint8)
            cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MODIFIED", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            main_body = np.hstack([image, final_img])
            final_img = np.vstack([header, main_body])

        output_filename = f"{folder_name}.jpg"
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_path, final_img)
        
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Motion Blur dataset.")
    parser.add_argument("--base_folder", type=str, required=True, help="Path to image dataset")
    parser.add_argument("--main_json", type=str, required=True, help="Path to the main JSON config")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--output_folder", type=str, default="motion_blur_outputs", help="Output directory")
    parser.add_argument("--num_threads", type=int, default=4, help="Concurrent threads")
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium', help='Blur severity')
    parser.add_argument('--cmp', action='store_true', help='If set, saves side-by-side comparison with original image.')

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
        
        if motion.get("feasible") is True:
            target = motion.get("target_object")
            if target and os.path.isdir(folder_path):
                tasks.append((folder_path, target))

    if args.max_images: 
        tasks = tasks[:args.max_images]

    dataset_results = []
    dataset_lock = threading.Lock()
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
            vertical_shift_factor_val = random.uniform(*level_params["vertical_shift_factor"])
            
            current_params = {
                'steps': steps_val, 
                'intensity': intensity_val, 
                'vertical_shift_factor': vertical_shift_factor_val,
                'noise_range': level_params['noise_range'],
                'curve_freq': level_params['curve_freq'],
                'curve_amp': level_params['curve_amp']
            }

            level_output_folder = f"{args.output_folder}/{current_level_name}"

            future = executor.submit(process_single_directory, p, t, level_output_folder, current_level_name, current_params, dataset_results, dataset_lock, args.cmp)
            futures[future] = (p, t)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing images"):
            pass

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
