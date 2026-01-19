import cv2
import numpy as np
import json
import os
import glob
import concurrent.futures
import argparse
from tqdm import tqdm
import threading
import random
from datetime import datetime
import time

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "overexposure",
    "scope_type": "Local", 
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with atmospheric bloom around the light source",
        "severe": "with heavy light bleeding and loss of surrounding detail"
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
SEVERE_FACTOR = 2.6
MEDIUM_PARAMS = {
    "sigma": (45.0, 50.0),
    "intensity": (1.6, 1.7)
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
# Core Logic
# ========================================================

def find_all_mask_files(mask_dir, label):
    """ 获取所有相关的 mask 文件 """
    search_pattern = os.path.join(mask_dir, f"*{label}*.png")
    matched_files = glob.glob(search_pattern)
    label_underscore = label.replace(' ', '_')
    search_pattern_underscore = os.path.join(mask_dir, f"*{label_underscore}*.png")
    matched_files_underscore = glob.glob(search_pattern_underscore)
    return list(set(matched_files + matched_files_underscore))

def apply_atmospheric_bloom(img, mask, sigma, intensity):
    """
    模拟真实的大气光晕效果
    """
    img_float = img.astype(np.float32) / 255.0
    
    mask_float = mask.astype(np.float32) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = cv2.merge([mask_float, mask_float, mask_float])

    # 提取光源层 (保留光源原始颜色)
    light_source = img_float * mask_float

    # 制造光晕 (Bloom Generation)
    bloom_layer = cv2.GaussianBlur(light_source, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 光学叠加 (Additive Blending)
    img_with_core = img_float + (light_source * 0.5) 
    final_img = img_with_core + (bloom_layer * intensity)

    # 色调映射 (Clipping)
    final_img = np.clip(final_img, 0, 1.0)
    
    return (final_img * 255).astype(np.uint8)

def process_image_overexposure(folder_path, target_label, output_folder, m_sigma, m_intensity, s_sigma, s_intensity, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    try:
        folder_name = os.path.basename(folder_path)
        raw_img_path = os.path.join(folder_path, "raw_image.jpg")
        mask_dir = os.path.join(folder_path, "mask")
        
        if not os.path.exists(raw_img_path) or not os.path.exists(mask_dir):
            return

        mask_files = find_all_mask_files(mask_dir, target_label)
        if not mask_files:
            print(f"[SKIP] {folder_name}: Target object '{target_label}' mask not found.")
            return

        img = cv2.imread(raw_img_path)
        if img is None: return
        h, w = img.shape[:2]

        # 合并 Mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for m_path in mask_files:
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m_img is None: continue
            if m_img.shape[:2] != (h, w):
                m_img = cv2.resize(m_img, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.bitwise_or(combined_mask, m_img)

        if np.sum(combined_mask) == 0:
            print(f"[SKIP] {folder_name}: Combined mask is empty for '{target_label}'.")
            return

        # --- 执行算法 ---
        img_m = apply_atmospheric_bloom(img, combined_mask, sigma=m_sigma, intensity=m_intensity)
        img_s = apply_atmospheric_bloom(img, combined_mask, sigma=s_sigma, intensity=s_intensity)

        final_m, final_s = img_m, img_s
        if cmp_mode:
            # 三图拼接: [Original, Medium, Severe]
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

        # 保存
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
        print(f"[ERROR] Failed to process {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Local Overexposure dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders with raw_image.jpg')
    parser.add_argument('--main_json', type=str, required=True, help='Path to the main JSON config')
    parser.add_argument('--output_folder', type=str, default='local_overexposure_outputs', help='Folder to save processed images')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of concurrent threads.')
    parser.add_argument('--max_images', type=int, default=None, help='本次任务处理的最大图片数量。')
    parser.add_argument('--dataset_json', type=str, default='local_overexposure_dataset.json', help='结果清单文件名。')
    parser.add_argument('--cmp', action='store_true', help='If set, saves side-by-side comparison with original image.')

    args = parser.parse_args()

    if not os.path.exists(args.main_json):
        print(f"Error: JSON file '{args.main_json}' does not exist.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    if isinstance(main_data_raw, list):
        main_data = {item['filename']: item for item in main_data_raw}
    else:
        main_data = main_data_raw

    tasks = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        
        motion = info.get("augmentation_assessment", {}).get("can_add_local_overexposure", {})
        if motion.get("feasible") is True:
            target = motion.get("target_source")
            visual_analysis = info.get("visual_analysis", "")
            if target and os.path.isdir(folder_path):
                tasks.append((folder_path, target, visual_analysis))

    if args.max_images: tasks = tasks[:args.max_images]

    dataset_results, dataset_lock = [], threading.Lock()
    start_time = time.time()

    print(f"Task Started | Generating BOTH Medium & Severe | Target Images: {len(tasks)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for p, t, v in tasks:
            # 1. 采样一次 Medium 基准参数
            m_sigma = random.uniform(*MEDIUM_PARAMS["sigma"])
            m_intensity = random.uniform(*MEDIUM_PARAMS["intensity"])
            
            # 2. 计算 Severe 参数
            s_sigma = m_sigma * SEVERE_FACTOR
            s_intensity = m_intensity * SEVERE_FACTOR

            # 3. 提交一个统一的任务处理两个级别
            future = executor.submit(
                process_image_overexposure, p, t, args.output_folder, 
                m_sigma, m_intensity, s_sigma, s_intensity, 
                dataset_results, dataset_lock, v, args.cmp
            )
            futures[future] = (p, t)

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Levels"):
            pass

    # 4. 自动分发保存 Medium 和 Severe 的 JSON 数据集
    if dataset_results:
        # 获取不带后缀的基础文件名 (例如 local_overexposure_dataset)
        base_name = os.path.splitext(args.dataset_json)[0]
        # 如果文件名里已经带了级别后缀，先去掉它，保证最后生成的是 _medium 和 _severe
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
