import argparse
import concurrent.futures
import os
import threading
import time
import random
import math
import json
import glob
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ========================================================
# Device Configuration
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "focus blur",
    "scope_type": "Local", 
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with edge softening and loss of fine textures",
        "severe": "with complete loss of structural information and shape definition"
    },
    
    # [基础QA] 用于描述位置的模板
    "location_desc": {
        "Global": "It is a global distortion affecting the entire image frame uniformly.",
        "Local": "It is localized, specifically affecting the {target}."
    }
}

# ========================================================
# Parameters
# ========================================================
MEDIUM_PARAMS = {
    "max_radius": (32, 38)
}

SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 2.3

# ========================================================
# System Prompts
# ========================================================
SYSTEM_PROMPT_COT = (
    "You are a professional AI visual expert. Provide a comprehensive technical diagnosis. "
    "Regardless of how the user asks, your response must strictly follow this structure:\n"
    "1. Image Content: Briefly describe the scene.\n"
    "2. Distortion Issue & Severity: Identify the distortion and assess its intensity.\n"
    "3. Distribution Scope: Determine if it is global or local."
)

SYSTEM_PROMPT_BASIC = "You are a professional AI visual expert. Answer questions about image quality accurately and concisely."

# ========================================================
# Helpers
# ========================================================
def get_clean_desc(visual_analysis):
    """
    清理 visual_analysis，移除所有与 defect/quality 相关的后缀片段
    """
    import re
    
    # 移除 "Defect" 之后的所有内容
    match = re.search(r'(?i)Defect', visual_analysis)
    if match:
        visual_analysis = visual_analysis[:match.start()]
    
    # 移除常见的质量评估短尾巴
    patterns_to_remove = [
        r'\s*No\s+(severe|major|significant|obvious)[\w\s]*[\.。]?\s*$',
        r'\s*Quality\s+is\s+[\w\s]*[\.。]?\s*$',
        r'\s*Overall[\w\s]*[\.。]?\s*$',
        r'\s*The\s+image\s+is[\w\s]*[\.。]?\s*$',
    ]
    
    for pattern in patterns_to_remove:
        visual_analysis = re.sub(pattern, '', visual_analysis, flags=re.IGNORECASE)
    
    return visual_analysis.strip().rstrip('.,;: ')

# ========================================================
# Logic: 文本生成器
# ========================================================
def generate_universal_qa(level, distortion_conf, target_object="N/A"):
    d_name = distortion_conf["name"]
    d_desc = distortion_conf["severity_desc"][level]
    scope = distortion_conf["scope_type"]
    
    # --- Q1: Type ---
    q_type_opts = [
        "Identify the specific **distortion** present in this image.",
        "What type of **distortion** can be observed in this photograph?",
        "Name the primary **distortion** affecting this shot."
    ]
    q1 = random.choice(q_type_opts)
    a1 = f"The image suffers from **{d_name}**."

    # --- Q2: Severity ---
    q_sev_opts = [
        "Assess the severity level of the **distortion** in this image.",
        "How would you rate the intensity of the **distortion** in this photo?",
        "What is the severity of the **distortion** found here?"
    ]
    q2 = random.choice(q_sev_opts)
    a2 = f"The distortion is **{level}**, {d_desc}."

    # --- Q3: Location ---
    q_loc_opts = [
        "Determine the spatial distribution of the **distortion** within this image frame.",
        "Is the **distortion** in this picture global or localized to a specific area?",
        "Locate the **distortion** in this specific image."
    ]
    q3 = random.choice(q_loc_opts)
    
    if scope == "Global":
        a3 = distortion_conf["location_desc"]["Global"]
    else:
        a3 = distortion_conf["location_desc"]["Local"].format(target=target_object)

    return [(q1, a1), (q2, a2), (q3, a3)]

def generate_narrative_cot(scene_desc, level, distortion_conf, target_object="N/A"):
    """
    【严谨逻辑版 CoT 生成 - 修正标号】
    强制使用 1. 2. 3. 标号，与 System Prompt 完美对齐。
    """
    d_name = distortion_conf["name"]
    scope = distortion_conf["scope_type"]
    scene_clean = scene_desc.strip().rstrip('.')
    
    # ----------------------------------------------------------------------
    # Block 1: Image Content (强制 1.)
    # ----------------------------------------------------------------------
    # 将首字母小写，避免 "Visual analysis shows that The..." 这种错误
    if scene_clean and scene_clean[0].isupper():
        scene_clean = scene_clean[0].lower() + scene_clean[1:]
    
    part1 = f"1. Image Content: Visual analysis shows that {scene_clean}."

    # ----------------------------------------------------------------------
    # Block 2: Distortion Issue & Severity (强制 2.)
    # ----------------------------------------------------------------------
    issue_templates = [
        f"2. Distortion Issue & Severity: A technical inspection reveals **{d_name}**. The degradation is of **{level}** severity.",
        f"2. Distortion Issue & Severity: The image suffers from **{level} {d_name}**.",
        f"2. Distortion Issue & Severity: **{d_name}** is identified as the primary defect at **{level}** level."
    ]
    part2 = random.choice(issue_templates)

    # ----------------------------------------------------------------------
    # Block 3: Distribution Scope (强制 3.)
    # ----------------------------------------------------------------------
    if scope == "Global":
        scope_templates = [
            "3. Distribution Scope: This is a global artifact affecting the entire frame uniformly.",
            "3. Distribution Scope: The distortion distributes globally across the whole image field."
        ]
    else: # Local
        scope_templates = [
            f"3. Distribution Scope: This defect is localized, specifically affecting the **{target_object}**.",
            f"3. Distribution Scope: The distortion is not global but concentrated on the **{target_object}**."
        ]
    part3 = random.choice(scope_templates)

    # ----------------------------------------------------------------------
    # 组装 (用换行符 \n 连接，结构更清晰)
    # ----------------------------------------------------------------------
    return f"{part1}\n{part2}\n{part3}"

# ========================================================
# Visualization Helper
# ========================================================
def draw_red_cross(img, cx, cy, img_h, img_w):
    scale = min(img_h, img_w) / 1000.0
    radius = max(2, int(4 * scale))
    # 避免修改原图，拷贝一份
    vis_img = img.copy()
    cv2.circle(vis_img, (cx, cy), radius + 1, (0, 0, 0), -1) 
    cv2.circle(vis_img, (cx, cy), radius, (0, 0, 255), -1)     
    return vis_img

# ========================================================
# Core Logic: Depth-Aware Defocus Blur
# ========================================================
def create_disk_kernel(radius):
    if radius <= 0: return None
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

def apply_blur_layered(img_linear, depth_map, focal_depth, max_radius, num_layers=5):
    B, C, H, W = img_linear.shape
    depth_map_smoothed = F.avg_pool2d(depth_map, kernel_size=5, stride=1, padding=2)
    depth_diff = torch.abs(depth_map_smoothed - focal_depth)
    max_diff = torch.max(depth_diff) + 1e-6
    norm_diff = depth_diff / max_diff 
    target_radii = norm_diff * max_radius
    
    step = max_radius / (num_layers - 1)
    radii_list = [i * step for i in range(num_layers)]
    output = img_linear.clone()
    
    for i in range(num_layers - 1):
        r_low, r_high = radii_list[i], radii_list[i+1]
        mask = (target_radii >= r_low) & (target_radii < r_high)
        if not torch.any(mask): continue
        
        r_curr = (r_low + r_high) / 2
        kernel = create_disk_kernel(r_curr)
        if kernel is not None:
            k_size = kernel.shape[0]
            pad = k_size // 2
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            img_padded = F.pad(img_linear, (pad, pad, pad, pad), mode='reflect')
            blurred = F.conv2d(img_padded, kernel_4d, groups=3)
            if blurred.shape != img_linear.shape:
                blurred = F.interpolate(blurred, size=(H, W), mode='bilinear', align_corners=False)
            
            mask_expanded = mask.repeat(1, 3, 1, 1)
            output[mask_expanded] = blurred[mask_expanded]
            del kernel, kernel_4d, img_padded, blurred
    
    del depth_map_smoothed, target_radii, mask
    torch.cuda.empty_cache()
    return output

def apply_depth_focus_blur_torch(image_np, depth_np, focal_point, max_radius):
    img_h, img_w = image_np.shape[:2]
    img_input = image_np[..., ::-1].copy() 
    img_tensor = torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    img_linear = torch.pow(img_tensor + 1e-6, 2.2)
    depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    fx, fy = focal_point
    y_s, y_e = max(0, fy-2), min(img_h, fy+3)
    x_s, x_e = max(0, fx-2), min(img_w, fx+3)
    focal_region = depth_tensor[0, 0, y_s:y_e, x_s:x_e]
    focal_depth = torch.median(focal_region).item()
    
    blurred_linear = apply_blur_layered(img_linear, depth_tensor, focal_depth, max_radius, num_layers=5)
    blurred_srgb = torch.pow(blurred_linear + 1e-6, 1/2.2)
    output_np = blurred_srgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    del img_tensor, img_linear, depth_tensor, blurred_linear, blurred_srgb
    torch.cuda.empty_cache()
    return output_np[..., ::-1]

# ========================================================
# Mask Helper
# ========================================================

def find_focal_point_from_mask(subdir, secondary_object):
    mask_dir = os.path.join(subdir, "mask")
    if not os.path.exists(mask_dir): return None
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files: return None

    orig = secondary_object.lower()
    variants = [orig, orig.replace(" ", "_"), orig.replace(" ", "")]
    words = orig.split()
    if len(words) > 1:
        variants.append(words[-1]) # core noun
        variants.append(words[0])  # adjective
        if orig.endswith('s'): variants.append(orig[:-1])

    best_mask_path = None
    for variant in variants:
        for f in mask_files:
            if variant in f.lower():
                best_mask_path = os.path.join(mask_dir, f)
                break
        if best_mask_path: break
            
    if best_mask_path is None: return None
    mask_img = cv2.imread(best_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None: return None
    ys, xs = np.where(mask_img > 0)
    if len(ys) == 0: return None
    idx = len(ys) // 2
    return (int(xs[idx]), int(ys[idx]))

# ========================================================
# Workflow Logic (Updated)
# ========================================================

def process_single_directory(subdir, output_folder, m_radius, s_radius, main_subject, secondary_object, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    """
    处理单个文件夹：强制生成 Medium 和 Severe 两个版本。
    构建 [基础QA -> CoT] 的多轮对话数据。
    """
    try:
        folder_name = os.path.basename(subdir)
        image_path = os.path.join(subdir, "raw_image.jpg")
        depth_path = os.path.join(subdir, "raw_image_depth.png")
        if not os.path.exists(image_path) or not os.path.exists(depth_path): return
        image = cv2.imread(image_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if image is None or depth is None: return
        h, w = image.shape[:2]

        # 1. 获取 Mask 焦点
        focal_point = find_focal_point_from_mask(subdir, secondary_object)
        
        # [Strict Filtering] 如果没有 Mask，直接跳过
        if focal_point is None:
            return

        # 2. 生成图像
        img_m = apply_depth_focus_blur_torch(image, depth, focal_point, m_radius)
        img_s = apply_depth_focus_blur_torch(image, depth, focal_point, s_radius)

        levels_to_generate = [("medium", img_m), ("severe", img_s)]

        # 3. 处理 CMP 可视化 (如果开启)
        if cmp_mode:
            cmp_dir = os.path.join(output_folder, "cmp_vis")
            os.makedirs(cmp_dir, exist_ok=True)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            header = np.full((header_h, w * 3, 3), 255, dtype=np.uint8)
            
            cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MEDIUM", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(header, "SEVERE", (int(w * 2.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            
            orig_vis = draw_red_cross(image, focal_point[0], focal_point[1], h, w)
            main_body = np.hstack([orig_vis, img_m, img_s])
            cmp_img = np.vstack([header, main_body])
            
            cv2.imwrite(os.path.join(cmp_dir, f"{folder_name}_cmp.jpg"), cmp_img)

        scene_desc = get_clean_desc(visual_analysis)
        new_entries = []
        
        # CoT 提问模板
        cot_user_prompts = [
            "Please perform a detailed image quality assessment for this nighttime photo.",
            "Evaluate the visual quality of this night scene and diagnose any technical defects.",
            "I need a professional analysis of this nighttime image's quality issues.",
            "Assess the clarity of this night shot and explain what specific distortion is present."
        ]

        # 4. 核心数据生成循环
        for level_name, final_img in levels_to_generate:
            
            # 保存训练图片
            lvl_dir = os.path.join(output_folder, level_name)
            os.makedirs(lvl_dir, exist_ok=True)
            out_path = os.path.join(lvl_dir, f"{folder_name}.jpg")
            cv2.imwrite(out_path, final_img)
            
            # 1. 生成 3 条基础 QA
            qa_list = generate_universal_qa(level_name, DISTORTION_CONFIG, main_subject)
            for q, a in qa_list:
                new_entries.append({
                    "_meta_level": level_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_BASIC},
                        {"role": "user", "content": f"{q}\n<image>"},
                        {"role": "assistant", "content": a}
                    ],
                    "images": [out_path]
                })
            
            # 2. 生成 1 条 CoT
            cot_paragraph = generate_narrative_cot(scene_desc, level_name, DISTORTION_CONFIG, main_subject)
            selected_prompt = random.choice(cot_user_prompts)
            
            new_entries.append({
                "_meta_level": level_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_COT},
                    {"role": "user", "content": f"{selected_prompt}\n<image>"},
                    {"role": "assistant", "content": cot_paragraph}
                ],
                "images": [out_path]
            })
        
        if new_entries:
            with dataset_lock:
                dataset_results.extend(new_entries)

    except Exception as e:
        print(f"[ERROR] Failed to process {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Depth-Aware Defocus Blur (Multi-turn CoT)")
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--main_json", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="depth_focus_blur_outputs")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--dataset_json", type=str, default="focus_blur_dataset.json")
    parser.add_argument('--cmp', action='store_true', help="Enable comparison mode (vis grid).")

    args = parser.parse_args()

    if not os.path.exists(args.base_folder) or not os.path.exists(args.main_json):
        print("Path Error.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)
    config_data = {item['filename']: item for item in main_data_raw} if isinstance(main_data_raw, list) else main_data_raw

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

    if args.max_images: tasks = tasks[:args.max_images]

    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()
    
    print(f"🚀 Started | Tasks: {len(tasks)} | Mode: Multi-turn CoT | Saving Medium & Severe")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for s, m, o, v in tasks:
            m_radius = random.randint(*MEDIUM_PARAMS["max_radius"])
            s_radius = int(m_radius * SEVERE_FACTOR)
            if s_radius <= m_radius: s_radius = m_radius + 10
            
            future = executor.submit(
                process_single_directory, s, args.output_folder, 
                m_radius, s_radius, m, o, 
                dataset_results, dataset_lock, v, args.cmp
            )
            futures[future] = s
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            pass

    # Save results - Separate into two JSONs
    if dataset_results:
        base_name = os.path.splitext(args.dataset_json)[0]
        for sfx in ["_slight", "_medium", "_severe", "_random"]:
            if base_name.endswith(sfx): base_name = base_name[:-len(sfx)]
        
        for lvl in ["medium", "severe"]:
            lvl_results = []
            for item in dataset_results:
                if item.get("_meta_level") == lvl:
                    clean_item = item.copy()
                    clean_item.pop("_meta_level", None)
                    lvl_results.append(clean_item)
            
            if lvl_results:
                out_name = f"{base_name}_{lvl}.json"
                with open(out_name, 'w', encoding='utf-8') as f:
                    json.dump(lvl_results, f, indent=2, ensure_ascii=False)
                print(f"Dataset saved to: {out_name} (Count: {len(lvl_results)})")

    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()