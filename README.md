# 图像失真处理脚本 - SFT 数据生成工具

本项目包含一系列用于模拟不同图像失真效果的 Python 脚本，专门用于生成高质量的 SFT（监督微调）训练数据集。每个脚本通过特定算法模拟逼真的图像缺陷，包括相机抖动、对焦模糊、ISO 噪声、过曝和欠曝等。

## 📊 数据生成策略

**所有脚本现在采用统一的数据生成策略：**

- ✅ **自动生成两个级别**：每次运行自动生成 `medium` 和 `severe` 两个严重程度的失真图片
- ✅ **标准化数据格式**：每张图片生成 4 条 SFT 数据（3 条基础问答 + 1 条 CoT 推理）
- ✅ **双 JSON 输出**：自动生成 `{distortion}_dataset_medium.json` 和 `{distortion}_dataset_severe.json`
- ✅ **统一失真术语**：
  - 失真类型：`camera shake`, `motion blur`, `focus blur`, `overexposure`, `underexposure`, `noise`
  - 严重程度：`medium`, `severe`

## 🔧 通用参数

所有脚本都支持以下参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base_folder` | 源图片文件夹路径（必需） | `6_night_sam3` |
| `--main_json` | 场景描述 JSON 文件（必需） | `dataset_with_prompt.json` |
| `--output_folder` | 输出文件夹（可选） | `camera_shake_results` |
| `--dataset_json` | 输出数据集 JSON 文件名（可选） | `camera_shake_dataset.json` |
| `--max_images N` | 限制处理图片数量（可选） | `50` |
| `--num_threads T` | 并行线程数（可选，默认4） | `8` |
| `--cmp` | 生成对比图（可选） | 添加此标志 |

> **注意**：
> - 脚本会自动在 `output_folder` 下创建 `medium/` 和 `severe/` 子文件夹
> - 如果启用 `--cmp`，会额外生成 `cmp_vis/` 文件夹存放对比图（Original | Medium | Severe）
> - 数据集 JSON 文件会自动添加 `_medium` 和 `_severe` 后缀

---

## 📝 脚本详细说明

### 1. `process_camera_shake.py` - 相机抖动

**失真类型**：`camera shake` (全局)  
**功能**：模拟相机抖动导致的全局方向性模糊效果

```bash
python process_camera_shake.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder camera_shake_results \
  --max_images 10 \
  --cmp

# 输出：
# - camera_shake_results/medium/*.jpg
# - camera_shake_results/severe/*.jpg
# - camera_shake_results/cmp_vis/*.jpg (如果使用 --cmp)
# - camera_shake_dataset_medium.json (包含所有 medium 级别的 SFT 数据)
# - camera_shake_dataset_severe.json (包含所有 severe 级别的 SFT 数据)
```

---

### 2. `process_focus_blur.py` - 对焦模糊

**失真类型**：`focus blur` (局部)  
**功能**：模拟焦点落在次要物体上导致主体模糊的效果

```bash
python process_focus_blur.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder focus_blur_results \
  --max_images 5 \
  --cmp

# 输出：
# - focus_blur_results/medium/*.jpg
# - focus_blur_results/severe/*.jpg
# - focus_blur_results/cmp_vis/*.jpg (如果使用 --cmp)
# - focus_blur_dataset_medium.json
# - focus_blur_dataset_severe.json
```

---

### 3. `process_iso_noise.py` - ISO 噪声

**失真类型**：`noise` (全局)  
**功能**：模拟高 ISO 感光度产生的颗粒感和色度噪声

```bash
python process_iso_noise.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder iso_noise_results \
  --max_images 10 \
  --cmp

# 输出：
# - iso_noise_results/medium/*.jpg
# - iso_noise_results/severe/*.jpg
# - iso_noise_results/cmp_vis/*.jpg (如果使用 --cmp)
# - iso_noise_dataset_medium.json
# - iso_noise_dataset_severe.json
```

---

### 4. `process_local_overexposure.py` - 局部过曝

**失真类型**：`overexposure` (局部)  
**功能**：模拟特定光源（如路灯、灯笼）的强泛光和过曝效果

```bash
python process_local_overexposure.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder local_overexposure_results \
  --max_images 10 \
  --cmp

# 输出：
# - local_overexposure_results/medium/*.jpg
# - local_overexposure_results/severe/*.jpg
# - local_overexposure_results/cmp_vis/*.jpg (如果使用 --cmp)
# - local_overexposure_dataset_medium.json
# - local_overexposure_dataset_severe.json
```

---

### 5. `process_motion_blur.py` - 运动模糊

**失真类型**：`motion blur` (局部)  
**功能**：模拟特定运动物体因快速移动产生的物理拖影模糊

```bash
python process_motion_blur.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder motion_blur_results \
  --max_images 50 \
  --cmp

# 输出：
# - motion_blur_results/medium/*.jpg
# - motion_blur_results/severe/*.jpg
# - motion_blur_results/cmp_vis/*.jpg (如果使用 --cmp)
# - motion_blur_dataset_medium.json
# - motion_blur_dataset_severe.json
```

---

### 6. `process_overexposure.py` - 全局过曝

**失真类型**：`overexposure` (全局)  
**功能**：模拟光圈或快门设置不当导致的整体画面过度曝光

```bash
python process_overexposure.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder overexposure_results \
  --max_images 50 \
  --cmp

# 输出：
# - overexposure_results/medium/*.jpg
# - overexposure_results/severe/*.jpg
# - overexposure_results/cmp_vis/*.jpg (如果使用 --cmp)
# - overexposure_dataset_medium.json
# - overexposure_dataset_severe.json
```

---

### 7. `process_underexposure.py` - 欠曝

**失真类型**：`underexposure` (全局)  
**功能**：模拟夜景中进光量不足导致的暗部细节丢失和噪声

```bash
python process_underexposure.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder underexposure_results \
  --max_images 50 \
  --cmp

# 输出：
# - underexposure_results/medium/*.jpg
# - underexposure_results/severe/*.jpg
# - underexposure_results/cmp_vis/*.jpg (如果使用 --cmp)
# - underexposure_dataset_medium.json
# - underexposure_dataset_severe.json
```

---

## 📦 SFT 数据格式

每个 JSON 文件包含多条 SFT 训练数据，每条数据格式如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional AI visual expert. Answer questions about image quality accurately and concisely."
    },
    {
      "role": "user",
      "content": "Identify the specific **distortion** present in this image.\n<image>"
    },
    {
      "role": "assistant",
      "content": "The image suffers from **camera shake**."
    }
  ],
  "images": ["camera_shake_results/medium/6p10_example.jpg"]
}
```

每张图片生成 **4 条数据**：
1. **类型识别** (Type) - 识别失真类型
2. **严重程度评估** (Severity) - 评估失真程度
3. **位置定位** (Location) - 判断全局/局部
4. **CoT 推理** (Reasoning) - 完整的技术分析

---

## 🎯 快速开始

```bash
# 1. 安装依赖
pip install opencv-python numpy torch albumentations tqdm

# 2. 准备数据
# 确保你有：
# - 6_night_sam3/ (图片文件夹)
# - dataset_with_prompt.json (场景描述文件)

# 3. 运行任意脚本（自动生成 medium 和 severe 两个级别）
python process_camera_shake.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --output_folder camera_shake_results \
  --max_images 100 \
  --cmp

# 4. 查看结果
# - camera_shake_results/medium/ - medium 级别图片
# - camera_shake_results/severe/ - severe 级别图片
# - camera_shake_results/cmp_vis/ - 对比图
# - camera_shake_dataset_medium.json - medium 级别 SFT 数据
# - camera_shake_dataset_severe.json - severe 级别 SFT 数据
```

---

## ⚙️ 高级配置

### 失真参数调整

每个脚本的 `MEDIUM_PARAMS` 定义了中等程度的失真参数，`SEVERE_FACTOR` 定义了严重程度的倍数。修改这些参数可以调整失真强度：

```python
# 例如在 process_camera_shake.py 中
MEDIUM_PARAMS = {"blur_limit": (40, 60)}  # medium 模糊范围
SEVERE_FACTOR = 2.0  # severe = medium * 2.0
```

### 并行处理加速

使用更多线程加速处理：

```bash
python process_iso_noise.py \
  --base_folder 6_night_sam3 \
  --main_json dataset_with_prompt.json \
  --num_threads 16 \
  --max_images 1000
```

---

## 📊 数据统计

每个脚本处理完成后会显示：
- 处理的图片数量
- 生成的 SFT 数据条数
- medium/severe 各自的数据量
- 总处理时间
