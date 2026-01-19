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

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "motion blur",
    "scope_type": "Local", 
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with noticeable trailing artifacts along the movement path",
        "severe": "with heavy directional smearing and loss of object definition"
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
SLIGHT_FACTOR = 0.5
SEVERE_FACTOR = 3.5  # [修改] 虽然这里只作参考，但在 main 逻辑中已大幅强化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEDIUM_PARAMS = {
    "steps": (30, 35),
    "intensity": (0.1, 0.15),
    "vertical_shift_factor": (0.008, 0.012),
    "noise_range": (0.01, 0.015),
    "curve_freq": 0.0,
    "curve_amp": 0.0
}

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
    """
    d_name = distortion_conf["name"]
    scope = distortion_conf["scope_type"]
    scene_clean = scene_desc.strip().rstrip('.')
    
    # ----------------------------------------------------------------------
    # Block 1: Image Content
    # ----------------------------------------------------------------------
    if scene_clean and scene_clean[0].isupper():
        scene_clean = scene_clean[0].lower() + scene_clean[1:]
    
    part1 = f"1. Image Content: Visual analysis shows that {scene_clean}."

    # ----------------------------------------------------------------------
    # Block 2: Distortion Issue & Severity
    # ----------------------------------------------------------------------
    issue_templates = [
        f"2. Distortion Issue & Severity: A technical inspection reveals **{d_name}**. The degradation is of **{level}** severity.",
        f"2. Distortion Issue & Severity: The image suffers from **{level} {d_name}**.",
        f"2. Distortion Issue & Severity: **{d_name}** is identified as the primary defect at **{level}** level."
    ]
    part2 = random.choice(issue_templates)

    # ----------------------------------------------------------------------
    # Block 3: Distribution Scope
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

    return f"{part1}\n{part2}\n{part3}"

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
    大幅提高了 Severe 级别的上限，允许产生更长的拖影。
    """
    diagonal = np.sqrt(w**2 + h**2)
    
    # 基础位移
    base_magnitude = diagonal * intensity
    
    # 根据等级设定动态上限
    if level_name == "slight":
        max_limit = 25.0
    elif level_name == "medium":
        max_limit = 55.0  
    else:
        # [修改] 从 95.0 大幅提升至 200.0，允许非常夸张的拖影
        max_limit = 200.0  
        
    MIN_SHIFT_PX = 1.5

    # 逻辑：对于小物体，应用一个衰减系数
    if diagonal < 200:
        damp = max(0.4, diagonal / 200.0)
        base_magnitude *= damp

    # 逻辑：对于大物体，使用软上限
    if base_magnitude > max_limit:
        # [修改] 增加软上限的斜率 (0.12 -> 0.25)，让超出 limit 的部分保留更多
        final_magnitude = max_limit + (base_magnitude - max_limit) * 0.25
    else:
        final_magnitude = base_magnitude

    final_magnitude = max(MIN_SHIFT_PX, final_magnitude)
    
    # 限制最大不超过物体自身最小尺寸的比例
    if level_name == "severe":
        limit_ratio = 0.75 # [修改] 严重模式下，允许拖影长度达到物体尺寸的 75%
    else:
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

        # 模拟快门均匀开启过程
        weight = 1.0 

        accumulated_img += warped_img * warped_mask * weight
        accumulated_mask += warped_mask * weight
        total_weight += weight

    accumulated_mask[accumulated_mask < 1e-3] = 1e-3
    blurred_object = accumulated_img / accumulated_mask
    
    # 转回 sRGB 空间
    blurred_object = torch.pow(blurred_object + 1e-6, 1/2.2)
    img_orig_gamma = torch.pow(img_tensor + 1e-6, 1/2.2)
    
    # 注入微量噪点
    blurred_object = add_matched_noise(blurred_object, intensity=noise_level)

    # 混合原图与模糊层
    final_mask = torch.clamp(accumulated_mask / (total_weight + 1e-6), 0, 1)
    output_tensor = blurred_object * final_mask + img_orig_gamma * (1 - final_mask)
    
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    return output_np[..., ::-1]

def process_single_directory(subdir, target_label, output_folder, m_int, m_steps, m_v_shift, m_noise, s_int, s_steps, s_v_shift, s_noise, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    try:
        folder_name = os.path.basename(subdir)
        image_path = os.path.join(subdir, "raw_image.jpg")
        mask_dir = os.path.join(subdir, "mask")
        if not os.path.exists(image_path): return
        matched_paths = find_motion_masks(mask_dir, target_label)
        if not matched_paths:
            print(f"[SKIP] {folder_name}: Target object '{target_label}' mask not found.")
            return
        image = cv2.imread(image_path)
        if image is None: return
        h_img, w_img = image.shape[:2]
        
        # 合并 Mask
        mask_info = []
        for m_path in matched_paths:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (h_img, w_img):
                    m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                area = np.sum(m > 0)
                if area > 0: mask_info.append({"mask": m, "area": area})
        if not mask_info:
            print(f"[SKIP] {folder_name}: No valid mask found for '{target_label}'.")
            return
        mask_info.sort(key=lambda x: x["area"], reverse=True)
        selected_masks = mask_info[:2]
        combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for info in selected_masks: combined_mask = cv2.bitwise_or(combined_mask, info["mask"])
        coords = cv2.findNonZero(combined_mask)
        if coords is None: 
            print(f"[SKIP] {folder_name}: Combined mask is empty for '{target_label}'.")
            return
        x, y, bw, bh = cv2.boundingRect(coords)
        soft_mask = refine_mask_edges(combined_mask)

        def process_core(intensity, steps, v_shift, noise_val, level_name):
            magnitude = calculate_adaptive_magnitude(bw, bh, intensity, level_name)
            return apply_physical_motion_blur_torch(image, soft_mask, steps=steps, trans_x=magnitude, trans_y=magnitude*v_shift, noise_level=noise_val)

        img_m = process_core(m_int, m_steps, m_v_shift, m_noise, "medium")
        img_s = process_core(s_int, s_steps, s_v_shift, s_noise, "severe")

        final_m, final_s = img_m, img_s
        if cmp_mode:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h_img / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h_img * 0.15)
            header = np.full((header_h, w_img * 3, 3), 255, dtype=np.uint8)
            cv2.putText(header, "ORIGINAL", (int(w_img * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MEDIUM", (int(w_img * 1.35), int(header_h * 0.7)), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(header, "SEVERE", (int(w_img * 2.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            main_body = np.hstack([image, img_m, img_s])
            cmp_img = np.vstack([header, main_body])
            final_m, final_s = cmp_img, cmp_img

        scene_desc = get_clean_desc(visual_analysis)
        new_entries = []
        
        # CoT 提问模板
        cot_user_prompts = [
            "Please perform a detailed image quality assessment for this nighttime photo.",
            "Evaluate the visual quality of this night scene and diagnose any technical defects.",
            "I need a professional analysis of this nighttime image's quality issues.",
            "Assess the clarity of this night shot and explain what specific distortion is present."
        ]
        
        for level_name, final_img in [("medium", final_m), ("severe", final_s)]:
            lvl_dir = os.path.join(output_folder, level_name)
            os.makedirs(lvl_dir, exist_ok=True)
            out_path = os.path.join(lvl_dir, f"{folder_name}.jpg")
            cv2.imwrite(out_path, final_img)
            
            # 1. 生成 3 条基础 QA
            qa_list = generate_universal_qa(level_name, DISTORTION_CONFIG, target_label)
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
            cot_paragraph = generate_narrative_cot(scene_desc, level_name, DISTORTION_CONFIG, target_label)
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
        tqdm.write(f"[ERROR] {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Motion Blur dataset.")
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--main_json", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_folder", type=str, default="motion_blur_outputs")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--dataset_json", type=str, default="motion_blur_dataset.json", help="SFT 数据集 JSON 文件名")
    parser.add_argument('--cmp', action='store_true')

    args = parser.parse_args()
    
    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' not found.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    # 兼容列表格式的 JSON
    if isinstance(main_data_raw, list):
        main_data = {item['filename']: item for item in main_data_raw}
    else:
        main_data = main_data_raw

    tasks = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        aug = info.get("augmentation_assessment", {})
        motion = aug.get("can_add_motion_blur", {})
        if motion.get("feasible") is True and os.path.isdir(folder_path):
            target = motion.get("target_object")
            visual_analysis = info.get("visual_analysis", "")
            if target: tasks.append((folder_path, target, visual_analysis))

    if args.max_images: tasks = tasks[:args.max_images]
    dataset_results, dataset_lock = [], threading.Lock()
    start_time = time.time()
    
    print(f"🚀 Started | Generating BOTH Medium & Severe | Tasks: {len(tasks)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for p, t, v in tasks:
            # 1. 采样一次 Medium 基准参数
            m_steps = random.randint(*MEDIUM_PARAMS["steps"])
            m_intensity = random.uniform(*MEDIUM_PARAMS["intensity"])
            m_v_shift = random.uniform(*MEDIUM_PARAMS["vertical_shift_factor"])
            m_noise = random.uniform(*MEDIUM_PARAMS["noise_range"])

            # 2. 计算 Severe 参数（大幅提高倍率）
            # [修改] 强度倍率从 4.2 提升至 6.0，让拖影更长
            s_intensity = m_intensity * 6.0 
            
            # [修改] 步数必须跟随变大，防止出现断裂，倍率从 2.5 提升至 5.0
            s_steps = int(m_steps * 5.0) 
            
            # [修改] 垂直偏移倍率提升，增加混乱感
            s_v_shift = m_v_shift * 3.0 
            
            s_noise = m_noise * 2.0 
            
            # 3. 提交统一任务
            future = executor.submit(
                process_single_directory, p, t, args.output_folder, 
                m_intensity, m_steps, m_v_shift, m_noise,
                s_intensity, s_steps, s_v_shift, s_noise,
                dataset_results, dataset_lock, v, args.cmp
            )
            futures[future] = (p, t)

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Levels"):
            pass

    # 4. 自动分发保存数据集
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