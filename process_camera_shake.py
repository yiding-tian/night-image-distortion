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

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "camera shake",
    "scope_type": "Global", 
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with noticeable directional smearing",
        "severe": "with heavy ghosting and loss of structural details"
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
SEVERE_FACTOR = 2.0
MEDIUM_PARAMS = {"blur_limit": (40, 60)}

# ========================================================
# System Prompts
# ========================================================

# [System Prompt] 强约束：CoT 必须按 1-2-3 格式回答
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
    # 例如: "No severe.", "No major issues.", "Quality is good." 等
    patterns_to_remove = [
        r'\s*No\s+(severe|major|significant|obvious)[\w\s]*[\.。]?\s*$',
        r'\s*Quality\s+is\s+[\w\s]*[\.。]?\s*$',
        r'\s*Overall[\w\s]*[\.。]?\s*$',
        r'\s*The\s+image\s+is[\w\s]*[\.。]?\s*$',
    ]
    
    for pattern in patterns_to_remove:
        visual_analysis = re.sub(pattern, '', visual_analysis, flags=re.IGNORECASE)
    
    # 清理末尾的标点和空白
    return visual_analysis.strip().rstrip('.,;: ')

# ========================================================
# Logic: 文本生成器 (核心修改区域)
# ========================================================

def generate_universal_qa(level, distortion_conf, target_object="N/A"):
    """
    生成 3 个基础维度的问答。
    【修改点】：
    1. 确保问题独立（包含 in this image）。
    2. 统一术语：只使用 "distortion"，不使用 defect/flaw/issue 等其他词。
    """
    d_name = distortion_conf["name"]
    d_desc = distortion_conf["severity_desc"][level]
    scope = distortion_conf["scope_type"]
    
    # --- Q1: Type (类型) - 强制使用 "distortion" ---
    q_type_opts = [
        "Identify the specific **distortion** present in this image.",
        "What type of **distortion** can be observed in this photograph?",
        "Name the primary **distortion** affecting this shot."
    ]
    q1 = random.choice(q_type_opts)
    a1 = f"The image suffers from **{d_name}**."

    # --- Q2: Severity (程度) - 强制使用 "distortion" ---
    q_sev_opts = [
        "Assess the severity level of the **distortion** in this image.",
        "How would you rate the intensity of the **distortion** in this photo?",
        "What is the severity of the **distortion** found here?"
    ]
    q2 = random.choice(q_sev_opts)
    a2 = f"The distortion is **{level}**, {d_desc}."

    # --- Q3: Location (位置) - 强制使用 "distortion" ---
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
# Core Processing Logic
# ========================================================

def apply_camera_shake(folder_path, output_folder, m_limit, s_limit, angle, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    try:
        folder_name = os.path.basename(folder_path)
        raw_image_path = os.path.join(folder_path, 'raw_image.jpg')
        if not os.path.exists(raw_image_path): return
        raw_image = cv2.imread(raw_image_path)
        if raw_image is None: return
        h, w = raw_image.shape[:2]

        # --- Image Processing ---
        def process_core(limit):
            rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            transform = A.MotionBlur(blur_limit=(limit, limit), angle_range=(angle, angle), allow_shifted=True, p=1.0)
            return cv2.cvtColor(transform(image=rgb)["image"], cv2.COLOR_RGB2BGR)

        img_m = process_core(m_limit)
        img_s = process_core(s_limit)

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
            main_body = np.hstack([raw_image, img_m, img_s])
            cmp_img = np.vstack([header, main_body])
            final_m, final_s = cmp_img, cmp_img

        scene_desc = get_clean_desc(visual_analysis)
        new_entries = [] 

        # -----------------------------------------------------
        #  CoT 提问模板 (中长型 - 明确夜景与评估)
        # -----------------------------------------------------
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
            
            # 1. 生成 3 条基础 QA (统一使用 distortion)
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
        print(f"Error processing {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nighttime Camera Shake dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder containing subfolders')
    parser.add_argument('--main_json', type=str, required=True, help='Path to dataset_with_prompt.json')
    parser.add_argument('--output_folder', type=str, default='camera_shake_outputs', help='Folder to save images')
    parser.add_argument('--max_images', type=int, default=None, help='Max images to process')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--dataset_json', type=str, default='camera_shake_dataset.json', help='Output JSON filename')
    parser.add_argument('--cmp', action='store_true', help='Comparison mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_folder) or not os.path.exists(args.main_json):
        print("Error: Invalid input paths.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)
    main_data = {item['filename']: item for item in main_data_raw} if isinstance(main_data_raw, list) else main_data_raw

    subdirs_info = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        if os.path.isdir(folder_path):
            subdirs_info.append((folder_path, info.get("visual_analysis", "")))

    target_subdirs_info = subdirs_info[:args.max_images] if args.max_images else subdirs_info
    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()

    print(f"🚀 Started | Images: {len(target_subdirs_info)} | Mode: Independent QA (distortion only) + CoT")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for subdir, visual_analysis in target_subdirs_info:
            m_limit = random.randint(*MEDIUM_PARAMS["blur_limit"])
            shake_angle = random.randint(0, 360) 
            s_limit = int(m_limit * SEVERE_FACTOR)
            if s_limit <= m_limit: s_limit = m_limit + 4

            future = executor.submit(
                apply_camera_shake, subdir, args.output_folder, 
                m_limit, s_limit, shake_angle,
                dataset_results, dataset_lock, visual_analysis, args.cmp
            )
            futures[future] = subdir
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            pass

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

if __name__ == '__main__':
    main()