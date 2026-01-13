# 图像失真处理脚本

本项目包含一系列用于模拟不同图像失真效果的 Python 脚本。每个脚本都旨在通过特定的算法和参数来生成逼真的图像缺陷，例如相机抖动、对焦模糊、ISO 噪声、过曝和欠曝等。脚本现在支持从 JSON 文件中提取场景描述，并生成符合 SFT（监督微调）格式的数据集。

## 脚本列表及功能说明

所有脚本都支持以下通用参数：
*   `--max_images N`: 限制处理图片的数量。N 为整数，表示最大处理图片数。
*   `--num_threads T`: 并行线程数，默认为 4。
*   `--level LEVEL`: 失真等级，可选 `slight` (轻微), `medium` (中等), `severe` (严重) 或 `random` (随机)。
*   `--main_json FILE`: **(核心)** 指定包含场景描述和增强评估的 JSON 文件（如 `dataset_with_prompt.json`）。
*   `--dataset_json FILE`: **(新增)** 指定生成的 SFT 训练数据集文件名（JSON 格式）。
*   `--cmp`: **(新增)** 如果设置此参数，脚本会将处理后的图与原图横向拼接在一起保存，方便对比效果。

> 注意：运行时只需指定一次 `--output_folder some_dir` 和 `--level medium` 等级，脚本会自动将图片输出到 `some_dir/medium`（或 `some_dir/slight` 等）目录，并使用源文件夹名称作为图片文件名。

---

### 1. `process_camera_shake.py` - 相机抖动

*   **功能**: 模拟图像因相机抖动而产生的全局方向性模糊效果。
*   **运行命令行示例**:
    ```bash
    python process_camera_shake.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder camera_shake_results \
      --dataset_json camera_shake_dataset.json \
      --level medium \
      --cmp
    ```

### 2. `process_focus_blur.py` - 对焦模糊

*   **功能**: 模拟因焦点落在次要物体上导致的主体模糊效果。
*   **运行命令行示例**:
    ```bash
    python process_focus_blur.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder focus_blur_results \
      --dataset_json focus_blur_dataset.json \
      --level medium \
      --cmp
    ```

### 3. `process_iso_noise.py` - ISO 噪声

*   **功能**: 模拟高感光度产生的颗粒感和色度噪声。
*   **运行命令行示例**:
    ```bash
    python process_iso_noise.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder iso_noise_results \
      --dataset_json iso_noise_dataset.json \
      --level medium \
      --cmp
    ```

### 4. `process_local_overexposure.py` - 局部过曝

*   **功能**: 模拟特定光源（如路灯、灯笼）的强泛光和过曝效果。
*   **运行命令行示例**:
    ```bash
    python process_local_overexposure.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder local_overexposure_results \
      --dataset_json local_overexposure_dataset.json \
      --level medium \
      --cmp
    ```

### 5. `process_motion_blur.py` - 运动模糊

*   **功能**: 模拟特定运动物体因快速移动产生的物理拖影模糊。
*   **运行命令行示例**:
    ```bash
    python process_motion_blur.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder motion_blur_results \
      --dataset_json motion_blur_dataset.json \
      --level medium \
      --cmp
    ```

### 6. `process_overexposure.py` - 全局过曝

*   **功能**: 模拟由于光圈或快门设置不当导致的整体画面过度曝光。
*   **运行命令行示例**:
    ```bash
    python process_overexposure.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder overexposure_results \
      --dataset_json overexposure_dataset.json \
      --level medium \
      --cmp
    ```

### 7. `process_underexposure.py` - 欠曝

*   **功能**: 模拟夜景中常见的进光量不足导致的暗部细节丢失和噪声。
*   **运行命令行示例**:
    ```bash
    python process_underexposure.py \
      --base_folder 6_night_sam3 \
      --main_json dataset_with_prompt.json \
      --max_images 5 \
      --output_folder underexposure_results \
      --dataset_json underexposure_dataset.json \
      --level medium \
      --cmp
    ```
