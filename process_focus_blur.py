import argparse
import concurrent.futures
import os
import threading
import time
import random
import math
import json
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ========================================================
# Device Configuration
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================================
# Parameters Configuration
# ========================================================
MEDIUM_MAX_RADIUS = 40
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.0

def get_max_radius(base_radius, factor):
    return max(3, int(base_radius * factor))

FOCUS_LEVELS = {
    "slight": get_max_radius(MEDIUM_MAX_RADIUS, SLIGHT_FACTOR),
    "medium": MEDIUM_MAX_RADIUS,
    "severe": get_max_radius(MEDIUM_MAX_RADIUS, SEVERE_FACTOR)
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
    "4. Distortion Specifics: The {type} is specifically localized at the {target}.\n"
    "5. Severity Level: {level}."
)

# ========================================================
# Visualization Helper: Small Dot Focus Indicator
# ========================================================
def draw_red_cross(img, cx, cy, img_h, img_w):
    scale = min(img_h, img_w) / 1000.0
    radius = max(2, int(4 * scale))
    color = (0, 0, 255) # BGR: Red
    cv2.circle(img, (cx, cy), radius + 1, (0, 0, 0), -1) 
    cv2.circle(img, (cx, cy), radius, color, -1)     
    return img

# ========================================================
# Core Logic: Depth-Aware Defocus Blur
# ========================================================

def create_disk_kernel(radius):
    if radius <= 0:
        return None
    kernel_size = int(radius * 2 + 1)
    y, x = torch.meshgrid(
        torch.linspace(-radius, radius, kernel_size),
        torch.linspace(-radius, radius, kernel_size),
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)
    kernel = torch.clamp(radius - dist + 0.5, 0, 1)
    kernel = kernel / kernel.sum()
    return kernel.to(DEVICE)

def apply_blur_layered(img_linear, depth_map, focal_depth, max_radius, num_layers=6):
    B, C, H, W = img_linear.shape
    depth_diff = torch.abs(depth_map - focal_depth)
    max_diff = torch.max(depth_diff) + 1e-6
    norm_diff = depth_diff / max_diff 
    target_radii = norm_diff * max_radius
    
    layers = [img_linear]
    step = max_radius / (num_layers - 1)
    radii_list = [0.0]
    
    for i in range(1, num_layers):
        r = i * step
        radii_list.append(r)
        kernel = create_disk_kernel(r)
        if kernel is not None:
            k_size = kernel.shape[0]
            pad = k_size // 2
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            img_padded = F.pad(img_linear, (pad, pad, pad, pad), mode='reflect')
            blurred = F.conv2d(img_padded, kernel_4d, groups=3)
            layers.append(blurred)
        else:
            layers.append(img_linear)
            
    output = torch.zeros_like(img_linear)
    for i in range(num_layers - 1):
        r_low = radii_list[i]
        r_high = radii_list[i+1]
        mask = (target_radii >= r_low) & (target_radii <= r_high)
        if not torch.any(mask):
            continue
        weight = (target_radii - r_low) / (r_high - r_low)
        blended = layers[i] * (1.0 - weight) + layers[i+1] * weight
        output[mask.repeat(1, 3, 1, 1)] = blended[mask.repeat(1, 3, 1, 1)]
        
    return output

def apply_depth_focus_blur_torch(image_np, depth_np, focal_point, max_radius):
    img_h, img_w = image_np.shape[:2]
    img_input = image_np[..., ::-1].copy() 
    img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    img_linear = torch.pow(img_tensor + 1e-6, 2.2)
    depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    fx, fy = focal_point
    focal_depth = depth_tensor[0, 0, fy, fx].item()
    blurred_linear = apply_blur_layered(img_linear, depth_tensor, focal_depth, max_radius)
    blurred_srgb = torch.pow(blurred_linear + 1e-6, 1/2.2)
    output_np = blurred_srgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[..., ::-1]

# ========================================================
# Mask Helper (With Enhanced Tolerance)
# ========================================================

def find_focal_point_from_mask(subdir, secondary_object):
    """
    寻找焦点掩码，具有高容错性。
    """
    mask_dir = os.path.join(subdir, "mask")
    if not os.path.exists(mask_dir):
        return None
    
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files:
        return None

    # 构造多种可能的搜索变体
    orig = secondary_object.lower()
    variants = [
        orig,                           # "potted plant"
        orig.replace(" ", "_"),         # "potted_plant"
        orig.replace(" ", ""),          # "pottedplant"
    ]
    
    # 如果是多词组，添加核心词（最后一个词，如 "statue"）和首词（如 "potted"）
    words = orig.split()
    if len(words) > 1:
        variants.append(words[-1])      # 核心词：statue
        variants.append(words[0])       # 首词：potted

    best_mask_path = None
    
    # 按照优先级尝试匹配变体
    for variant in variants:
        for f in mask_files:
            if variant in f.lower():
                best_mask_path = os.path.join(mask_dir, f)
                break
        if best_mask_path:
            break
            
    if best_mask_path is None:
        return None
        
    mask_img = cv2.imread(best_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
        
    ys, xs = np.where(mask_img > 0)
    if len(ys) == 0:
        return None
        
    # 选择掩码的几何中心点
    idx = len(ys) // 2
    return (int(xs[idx]), int(ys[idx]))

# ========================================================
# Workflow Logic
# ========================================================

def process_single_directory(subdir, output_folder, level_name, main_subject, secondary_object, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    folder_name = os.path.basename(subdir)
    image_path = os.path.join(subdir, "raw_image.jpg")
    depth_path = os.path.join(subdir, "raw_image_depth.png")
    
    if not os.path.exists(image_path) or not os.path.exists(depth_path): 
        return

    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if image is None or depth is None: 
        return
    
    h, w = image.shape[:2]

    try:
        # 焦点对准 secondary_object
        focal_point = find_focal_point_from_mask(subdir, secondary_object)
        
        is_random_focus = False
        if focal_point is None:
            print(f"\n[!] 匹配失败: '{folder_name}' 找不到 '{secondary_object}' 的Mask。使用随机对焦。")
            is_random_focus = True
            focal_point = (random.randint(int(w*0.3), int(w*0.7)), random.randint(int(h*0.3), int(h*0.7)))

        max_radius = FOCUS_LEVELS[level_name]
        blurred_image = apply_depth_focus_blur_torch(image, depth, focal_point, max_radius)
        final_vis_image = blurred_image.copy()
        
        draw_red_cross(final_vis_image, focal_point[0], focal_point[1], h, w)

        if cmp_mode:
            orig_vis = image.copy()
            draw_red_cross(orig_vis, focal_point[0], focal_point[1], h, w)
            combined = np.hstack([orig_vis, final_vis_image])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            header = np.full((header_h, combined.shape[1], 3), 255, dtype=np.uint8)
            
            target_text = f"ORIGINAL (Focus Target: {secondary_object})"
            if is_random_focus:
                target_text += " [MASK NOT FOUND - RANDOM]"
            
            cv2.putText(header, target_text, (int(w * 0.1), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, f"DEPTH DEFOCUS (Max R={max_radius})", (int(w * 1.1), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            final_vis_image = np.vstack([header, combined])
            
        output_filename = f"{folder_name}.jpg"
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_path, final_vis_image)

        # [MODIFIED] 生成符合 CoT 逻辑的对话内容 (system + user + assistant)
        scene_desc = get_clean_desc(visual_analysis)
        instruction_text = random.choice(QUESTION_TEMPLATES)
        
        answer_text = ANSWER_TEMPLATE.format(
            scene_desc=scene_desc,
            scope="Local",
            type="Defocus Blur",
            target=main_subject,
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
                out_path 
            ]
        }

        with dataset_lock:
            dataset_results.append(entry)
        
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Depth-Aware Defocus Blur with Enhanced Mask Search.")
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--main_json", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="depth_focus_blur_outputs")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--dataset_json", type=str, default="focus_blur_sft.json", help="SFT 数据集 JSON 文件名")
    parser.add_argument('--level', type=str, choices=['slight', 'medium', 'severe', 'random'], default='medium')
    parser.add_argument('--cmp', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.base_folder) or not os.path.exists(args.main_json):
        print("Path Error.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    if isinstance(main_data_raw, list):
        config_data = {item['filename']: item for item in main_data_raw}
    else:
        config_data = main_data_raw

    tasks = []
    for filename, info in config_data.items():
        folder_name = os.path.splitext(filename)[0]
        subdir = os.path.join(args.base_folder, folder_name)
        aug = info.get("augmentation_assessment", {})
        focus_cfg = aug.get("can_add_focus_blur", {})
        
        if focus_cfg.get("feasible") is True and os.path.isdir(subdir):
            main_subject = focus_cfg.get("main_subject")
            secondary_obj = focus_cfg.get("secondary_object")
            visual_analysis = info.get("visual_analysis", "")
            if secondary_obj and main_subject:
                tasks.append((subdir, main_subject, secondary_obj, visual_analysis))

    if args.max_images:
        tasks = tasks[:args.max_images]

    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()
    available_levels = ['slight', 'medium', 'severe']

    print(f"🚀 Started | Tasks: {len(tasks)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {executor.submit(
            process_single_directory, s, 
            os.path.join(args.output_folder, args.level if args.level != 'random' else random.choice(available_levels)), 
            args.level if args.level != 'random' else random.choice(available_levels), 
            m, o, dataset_results, dataset_lock, v, args.cmp
        ): s for s, m, o, v in tasks}
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing"):
            pass

    # [MODIFIED] Save dataset to JSON file (符合用户样例：扁平列表格式)
    if dataset_results:
        with open(args.dataset_json, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)

    print(f"\n--- Processing Finished ---")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Dataset saved to: {args.dataset_json}")

if __name__ == "__main__":
    main()