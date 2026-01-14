import os
import shutil
import argparse
import torch
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from collections import Counter

def organize_from_cache(feature_file, output_dir, threshold, max_process=0):
    print(f"正在读取特征文件: {feature_file} ...")
    if not os.path.exists(feature_file):
        print("错误：找不到特征文件")
        return

    # 1. 加载数据
    try:
        data = torch.load(feature_file, weights_only=False)
    except TypeError:
        data = torch.load(feature_file)

    features = data["features"]
    records = data["records"]
    
    # 2. 数量截断
    if max_process > 0 and max_process < len(records):
        print(f"仅使用前 {max_process} 条数据进行测试...")
        features = features[:max_process]
        records = records[:max_process]

    # 3. 执行聚类
    print(f"开始聚类 (数据量: {len(records)}, 阈值: {threshold})...")
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='complete', 
        distance_threshold=threshold
    )
    labels = clustering.fit_predict(features)
    
    clusters = {}
    for label, record in zip(labels, records):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(record)
        
    print(f"聚类完成，共 {len(clusters)} 个场景。")

    # 4. 清空旧目录
    if os.path.exists(output_dir):
        print(f"正在清空旧目录内容: {output_dir} ...")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"警告: 删除 {file_path} 失败. 原因: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # ================= 初始化统计容器 =================
    # 报告数据 (物理文件列表)
    report_data = {
        "summary": {
            "total_images": len(records),
            "total_clusters": len(clusters),
            "threshold": threshold
        },
        "clusters": {} 
    }

    # 统计数据 (用于分析分布)
    global_scene_counter = Counter()   # 全局场景计数
    global_object_counter = Counter()  # 全局对象计数
    detailed_stats_list = []           # 每个聚类的详细纯度分析
    # =================================================

    print("正在处理文件、读取标签并计算统计数据...")
    
    for cid, items in tqdm(clusters.items()):
        # 临时列表，用于存储该组内所有图片的标签
        votes_scene = []
        votes_object = []
        
        folder_name = f"Cluster_{cid:04d}_cnt{len(items)}"
        target_path = os.path.join(output_dir, folder_name)
        os.makedirs(target_path, exist_ok=True)
        
        image_list = [] 

        # --- A. 遍历图片，搬运并收集原始标签 ---
        for item in items:
            try:
                shutil.copy2(item['img_path'], target_path)
                if item.get('json_path'): 
                    shutil.copy2(item['json_path'], target_path)
                    
                    try:
                        with open(item['json_path'], 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            s_label = meta.get("scene_classification", {}).get("label", "Unknown")
                            votes_scene.append(s_label)
                            o_label = meta.get("object_classification", {}).get("label", "Unknown")
                            votes_object.append(o_label)
                    except Exception as json_err:
                        # 只有读取出错才计入 Error
                        votes_scene.append("Error")
                        votes_object.append("Error")
                        # print(f"读取JSON失败: {item['json_path']}")
                else:
                    # 没有 JSON 文件的情况
                    votes_scene.append("No_JSON")
                    votes_object.append("No_JSON")
                
                image_list.append(item['img_name'])

            except Exception as e:
                print(f"复制失败: {item.get('img_name', 'Unknown')} ({e})")
        
        # --- B. 投票决定 Consensus (最终标签) ---
        if votes_scene:
            consensus_scene = Counter(votes_scene).most_common(1)[0][0]
        else:
            consensus_scene = "No Data"
            
        if votes_object:
            consensus_object = Counter(votes_object).most_common(1)[0][0]
        else:
            consensus_object = "No Data"

        # --- C. 更新全局统计 (核心逻辑：每张图片的标签 = 它所属聚类的最终标签) ---
        # 如果这个组有 10 张图，全都被视为 consensus_scene
        global_scene_counter[consensus_scene] += len(items)
        global_object_counter[consensus_object] += len(items)

        # --- D. 计算聚类内部纯度 (百分比) ---
        def get_internal_dist(raw_votes):
            count = Counter(raw_votes)
            total = len(raw_votes)
            if total == 0: return {}
            # 返回格式: {"Urban": "9 (90.0%)", "Indoor": "1 (10.0%)"}
            return {k: f"{v} ({v/total*100:.1f}%)" for k, v in count.most_common()}

        cluster_stat_entry = {
            "cluster_name": folder_name,
            "final_labels": {
                "scene": consensus_scene,
                "object": consensus_object
            },
            "image_count": len(items),
            "ratio_global": f"{len(items)/len(records)*100:.2f}%", # 占总数据集的比例
            "purity_analysis": {
                "scene_raw_dist": get_internal_dist(votes_scene),
                "object_raw_dist": get_internal_dist(votes_object)
            },
            "image_list": image_list
        }
        detailed_stats_list.append(cluster_stat_entry)

        # --- E. 写入基础报告数据 ---
        report_data["clusters"][folder_name] = {
            "final_scene_label": consensus_scene,
            "final_object_label": consensus_object,
            "images": image_list
        }

    # 5. 保存基础报告 (clustering_report.json)
    report_file_path = os.path.join(output_dir, "clustering_report.json")
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"保存基础报告失败: {e}")

    # ================= 6. 保存统计分析报告 (clustering_statistics.json) =================
    print("正在生成统计分析报告...")
    
    total_imgs = len(records)
    
    # 格式化全局统计数据
    global_scene_stats = {k: f"{v} ({v/total_imgs*100:.1f}%)" for k, v in global_scene_counter.most_common()}
    global_object_stats = {k: f"{v} ({v/total_imgs*100:.1f}%)" for k, v in global_object_counter.most_common()}
    
    statistics_data = {
        "global_summary": {
            "total_images": total_imgs,
            "total_clusters": len(clusters),
            "scene_distribution": global_scene_stats,   # 全局场景百分比
            "object_distribution": global_object_stats  # 全局对象百分比
        },
        "cluster_details": detailed_stats_list          # 每个聚类的详细纯度
    }

    stats_file_path = os.path.join(output_dir, "clustering_statistics.json")
    try:
        with open(stats_file_path, 'w', encoding='utf-8') as f:
            json.dump(statistics_data, f, indent=4, ensure_ascii=False)
        print(f"统计报告已保存: {stats_file_path}")
    except Exception as e:
        print(f"保存统计报告失败: {e}")
    # ===================================================================================

    print(f"\n全部完成！结果已保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: 聚类、投票及生成统计报表")
    parser.add_argument('--feature_path', type=str, default='dinov2_feat.pt', help='Step 1 生成的特征文件路径')
    parser.add_argument('--output_dir', type=str, default='sorted_scenes_result', help='结果输出目录')
    parser.add_argument('--threshold', type=float, default=0.45, help='聚类阈值')
    parser.add_argument('--max_process', type=int, default=4001, help='仅处理前N条数据')
    
    args = parser.parse_args()
    organize_from_cache(args.feature_path, args.output_dir, args.threshold, args.max_process)