import cv2
import numpy as np
import os
import time
import concurrent.futures
import albumentations as A
import argparse
import json
import threading
import random
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
#  1. 通用失真配置 (Universal Distortion Config)
# ==============================================================================
DISTORTION_CONFIG = {
    "name": "noise",
    "scope_type": "Global", 
    
    # [基础QA] 用于描述严重程度的短语
    "severity_desc": {
        "medium": "with color noise and grain patterns across the frame",
        "severe": "with heavy color speckles and significant loss of fine detail"
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
SEVERE_FACTOR = 1.4
MEDIUM_PARAMS = {
    "color_shift": (0.07, 0.09),
    "intensity": (0.32, 0.37),
    "gauss_sigma": (4.8, 5.3),
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

def apply_high_iso_noise(folder_path, output_folder, m_color, m_int, m_sigma, s_color, s_int, s_sigma, dataset_results, dataset_lock, visual_analysis, cmp_mode=False):
    try:
        folder_name = os.path.basename(folder_path)
        raw_image_path = os.path.join(folder_path, 'raw_image.jpg')
        if not os.path.exists(raw_image_path): return
        raw_image = cv2.imread(raw_image_path)
        if raw_image is None: return
        h, w = raw_image.shape[:2]

        def process_core(color, intensity, g_sigma):
            rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            # 添加黑色基底，确保暗部也有噪点
            rgb = cv2.add(rgb, np.array([2, 2, 2], dtype=np.uint8))
            
            # 应用 ISO 噪点（彩色噪点）
            iso_transform = A.ISONoise(color_shift=(color, color), intensity=(intensity, intensity), p=1.0)
            rgb = iso_transform(image=rgb)["image"]
            
            # 手动添加高斯噪点（避免 Albumentations 版本兼容问题）
            noise = np.random.normal(3, g_sigma, rgb.shape).astype(np.float32)
            rgb_noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            return cv2.cvtColor(rgb_noisy, cv2.COLOR_RGB2BGR)

        img_m = process_core(m_color, m_int, m_sigma)
        img_s = process_core(s_color, s_int, s_sigma)

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
    parser = argparse.ArgumentParser(description="Generate Nighttime ISO Noise dataset.")
    parser.add_argument('--base_folder', type=str, required=True, help='Source folder')
    parser.add_argument('--main_json', type=str, required=True, help='Path to main JSON')
    parser.add_argument('--output_folder', type=str, default='iso_noise_results')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--dataset_json', type=str, default='iso_noise_dataset.json')
    parser.add_argument('--cmp', action='store_true', help='Enable comparison mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_folder) or not os.path.exists(args.main_json):
        print("Error: Input paths not valid.")
        return

    with open(args.main_json, 'r', encoding='utf-8') as f:
        main_data_raw = json.load(f)

    main_data = {item['filename']: item for item in main_data_raw} if isinstance(main_data_raw, list) else main_data_raw
    all_subdirs_info = []
    for filename, info in main_data.items():
        folder_name = os.path.splitext(filename)[0]
        folder_path = os.path.join(args.base_folder, folder_name)
        if os.path.isdir(folder_path):
            all_subdirs_info.append((folder_path, info.get("visual_analysis", "")))

    target_subdirs_info = all_subdirs_info[:args.max_images] if args.max_images else all_subdirs_info
    dataset_results = []
    dataset_lock = threading.Lock()
    start_time = time.time()

    print(f"🚀 Started | Generating BOTH Medium & Severe | Images: {len(target_subdirs_info)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {}
        for subdir, visual_analysis in target_subdirs_info:
            # 1. 为当前图片采样 Medium 基准参数
            m_color = random.uniform(*MEDIUM_PARAMS["color_shift"])
            m_int = random.uniform(*MEDIUM_PARAMS["intensity"])
            m_sigma = random.uniform(*MEDIUM_PARAMS["gauss_sigma"])

            # 2. 计算 Severe 参数
            s_color = min(m_color * 1.4, 1.0)  # 从 1.5 降到 1.4
            s_int = min(m_int * 2.0, 1.0)      # 从 2.2 降到 2.0
            s_sigma = m_sigma * 2.3            # 从 2.5 降到 2.3
            
            # 打印参数
            folder_name = os.path.basename(subdir)
            print(f"[{folder_name}] Medium: color={m_color:.3f}, intensity={m_int:.3f}, sigma={m_sigma:.2f} | "
                  f"Severe: color={s_color:.3f}, intensity={s_int:.3f}, sigma={s_sigma:.2f}")

            # 3. 提交一个统一任务处理两个级别
            future = executor.submit(
                apply_high_iso_noise, subdir, args.output_folder, 
                m_color, m_int, m_sigma, s_color, s_int, s_sigma,
                dataset_results, dataset_lock, visual_analysis, args.cmp
            )
            futures[future] = subdir
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Levels"):
            pass

    # 4. 自动分发保存数据集
    if dataset_results:
        base_name = os.path.splitext(args.dataset_json)[0]
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

if __name__ == '__main__':
    main()