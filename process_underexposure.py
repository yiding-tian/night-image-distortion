import cv2
import numpy as np
import os
import time
import concurrent.futures
import argparse
import threading
import json
import random
import albumentations as A  # 用于添加真实的 ISO 噪点
from tqdm import tqdm

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "underexposure",
    "scope_type": "Global",
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with shadow detail loss and overall darkness",
        "severe": "with heavy loss of visibility and shadow crushing"
    },
    
    # [基础QA] 用于描述位置的模板
    "location_desc": {
        "Global": "It is a global distortion affecting the entire image frame uniformly.",
        "Local": "It is localized, specifically affecting the {target}."
    }
}

# ========================================================
# Parameters (已根据需求调整：更暗 + 噪点更少)
# ========================================================
SLIGHT_FACTOR = 0.5  
SEVERE_FACTOR = 2.5  # [修改] 提高倍率，让 Severe 亮度除以 2.5，画面会非常暗

MEDIUM_PARAMS = {
    # 基础亮度：0.35~0.55。Severe 除以 2.5 后约为 0.14~0.22 (极暗)
    "brightness_factor": (0.35, 0.55),
    
    # 噪点概率：稍微降低一点触发几率
    "noise_probability": (0.2, 0.4),
    
    # [修改] 大幅降低杂色 (Color Shift)，避免出现“彩虹噪点”
    # 0.01~0.03 非常微弱，接近黑白颗粒
    "iso_noise_color_shift": (0.01, 0.03),
    
    # [修改] 降低基础强度。
    # Severe 乘 2.5 后约为 0.12~0.25，属于“可见但不过分”的噪点
    "iso_noise_intensity": (0.05, 0.10)
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
    【严谨逻辑版 CoT 生成】
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
# Core Logic: Linear Dimming + ISO Noise
# ========================================================

def apply_moderate_underexposure(img, brightness_factor, noise_prob, iso_color, iso_intensity):
    """
    1. 注入噪点 (模拟高感光度)
    2. 线性降低亮度 (模拟快门/光圈进光量减少)
    """
    img_processed = img.copy()

    # --- Step 1: 概率性注入 ISO 噪点 ---
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
            pass

    # --- Step 2: 线性降低亮度 (Linear Dimming) ---
    img_float = img_processed.astype(np.float32) / 255.0
    
    # 直接乘以系数
    darkened_img = img_float * brightness_factor
    
    # 截断并转回 uint8
    final_img = np.clip(darkened_img * 255, 0, 255).astype(np.uint8)
    
    return final_img

def process_single_folder(folder_path, output_folder, m_bf, m_np, m_nc, m_ni, s_bf, s_np, s_nc, s_ni, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    try:
        folder_name = os.path.basename(folder_path)
        raw_img_path = os.path.join(folder_path, "raw_image.jpg")
        if not os.path.exists(raw_img_path): return
        img = cv2.imread(raw_img_path)
        if img is None: return
        h, w = img.shape[:2]

        img_m = apply_moderate_underexposure(img, m_bf, m_np, m_nc, m_ni)
        img_s = apply_moderate_underexposure(img, s_bf, s_np, s_nc, s_ni)

        final_m, final_s = img_m, img_s
        if cmp_mode:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 600.0
            thickness = max(2, int(font_scale * 3))
            header_h = int(h * 0.15)
            header = np.full((header_h, w * 3, 3), 255, dtype=np.uint8)
            cv2.putText(header, "ORIGINAL", (int(w * 0.35), int(header_h * 0.7)), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, "MEDIUM", (int(w * 1.35), int(header_h * 0.7)), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(header, "SEVERE", (int(w * 2.35), int(header_h * 0.7)), font, font_scale, (0, 0, 255), thickness)
            main_body = np.hstack([img, img_m, img_s])
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
        
        for level_name, final_image in [("medium", final_m), ("severe", final_s)]:
            lvl_dir = os.path.join(output_folder, level_name)
            os.makedirs(lvl_dir, exist_ok=True)
            out_path = os.path.join(lvl_dir, f"{folder_name}.jpg")
            cv2.imwrite(out_path, final_image)
            
            # 1. 生成 3 条基础 QA
            qa_list = generate_universal_qa(level_name, DISTORTION_CONFIG)
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
            cot_paragraph = generate_narrative_cot(scene_desc, level_name, DISTORTION_CONFIG)
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
        print(f"[ERROR] Failed to process {folder_path}: {e}")

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate Underexposure with Tuned Noise")
    parser.add_argument('--base_folder', type=str, required=True, help='数据集根目录')
    parser.add_argument('--main_json', type=str, required=True, help='Path to dataset_with_prompt.json')
    parser.add_argument('--output_folder', type=str, default='underexposure_results')
    parser.add_argument('--dataset_json', type=str, default='underexposure_dataset.json', help='结果清单文件名。')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--cmp', action='store_true', help='开启对比图模式')

    args = parser.parse_args()

    if not os.path.exists(args.base_folder):
        print(f"Error: Base directory not found at '{args.base_folder}'")
        return

    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' not found.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    if isinstance(main_data_raw, list):
        main_data = {item['filename']: item for item in main_data_raw}
    else:
        main_data = main_data_raw

    subfolders_info = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        if os.path.isdir(folder_path):
            subfolders_info.append((folder_path, info.get("visual_analysis", "")))

    if args.max_images:
        subfolders_info = subfolders_info[:args.max_images]
    
    print(f"🚀 任务启动 | 模式: 极暗 + 低噪点 | 文件夹数量: {len(subfolders_info)}")

    dataset_results = []
    dataset_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for folder, visual_analysis in subfolders_info:
            # 1. 采样 Medium 参数 (基础值已调低)
            m_bf = random.uniform(*MEDIUM_PARAMS["brightness_factor"])
            m_np = random.uniform(*MEDIUM_PARAMS["noise_probability"])
            m_nc = random.uniform(*MEDIUM_PARAMS["iso_noise_color_shift"])
            m_ni = random.uniform(*MEDIUM_PARAMS["iso_noise_intensity"])

            # 2. 计算 Severe 参数
            # 亮度：除以 2.5，大幅变暗
            s_bf = m_bf / SEVERE_FACTOR
            
            # 噪点：虽然乘以 SEVERE_FACTOR，但因为基础值 m_ni 很小，
            # 最终 s_ni 依然会很克制，且上限限制为 1.0 (Albumentations 要求)
            s_np = min(m_np * SEVERE_FACTOR, 1.0)
            s_nc = min(m_nc * SEVERE_FACTOR, 1.0)
            s_ni = min(m_ni * SEVERE_FACTOR, 1.0)
            
            # 3. 提交任务
            future = executor.submit(
                process_single_folder, folder, args.output_folder, 
                m_bf, m_np, m_nc, m_ni, s_bf, s_np, s_nc, s_ni,
                dataset_results, dataset_lock, visual_analysis, args.cmp
            )
            futures[future] = folder

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            pass

    # 4. 自动分发保存数据集
    if dataset_results:
        base_name = os.path.splitext(args.dataset_json)[0]
        # 清理文件名防止重复后缀
        for sfx in ["_slight", "_medium", "_severe", "_random"]:
            if base_name.endswith(sfx):
                base_name = base_name[:-len(sfx)]

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