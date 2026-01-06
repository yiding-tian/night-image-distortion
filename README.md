# 图像失真处理脚本

本项目包含一系列用于模拟不同图像失真效果的Python脚本。每个脚本都旨在通过特定的算法和参数来生成逼真的图像缺陷，例如相机抖动、对焦模糊、ISO噪声、过曝和欠曝等。

## 脚本列表及功能说明

所有脚本都支持以下通用参数：
*   `--max_images N`: 限制处理图片的数量。N 为整数，表示最大处理图片数。
*   `--num_threads T`: 并行线程数，默认为 4。
*   `--level LEVEL`: 失真等级，可选 `slight` (轻微), `medium` (中等), `severe` (严重) 或 `random` (随机)。
*   `--cmp`: **(新增)** 如果设置此参数，脚本会将处理后的图与原图横向拼接在一起保存，方便对比效果。

> 注意：运行时只需指定一次 `--output_folder some_dir` 和 `--level medium` 等级，脚本会自动将图片输出到 `some_dir/medium`（或 `some_dir/slight` 等）目录，并使用源文件夹名称作为图片文件名。

### 1. `process_camera_shake.py` - 相机抖动

*   **功能**: 模拟图像因相机抖动而产生的模糊效果。
*   **运行命令行示例**:
    ```bash
    python process_camera_shake.py \
      --base_folder 6_night_sam3 \
      --max_images 25 \
      --output_folder camera_shake_results \
      --dataset_json camera_shake_dataset.json \
      --level medium \
      --cmp
    ```

### 2. `process_focus_blur.py` - 对焦模糊

*   **功能**: 模拟图像因对焦不准确而产生的模糊效果。
*   **运行命令行示例**:
    ```bash
    python process_focus_blur.py \
      --base_folder 6_night_sam3 \
      --main_json valid_focus_blur.json \
      --output_folder focus_blur_results \
      --level medium \
      --max_images 25 \
      --cmp
    ```

### 3. `process_iso_noise.py` - ISO噪声

*   **功能**: 模拟高ISO设置下图像产生的颗粒感和色度噪声。
*   **运行命令行示例**:
    ```bash
    python process_iso_noise.py \
      --base_folder 6_night_sam3 \
      --output_folder iso_noise_results \
      --num_threads 8 \
      --max_images 25 \
      --level medium
    ```
    *   **失真处理方式**:
        *   `slight` (轻微): 模拟低感光度（ISO 800-1600）下的轻微颗粒感。
        *   `medium` (中等): 模拟弱光环境（ISO 3200-6400）下明显的噪点。
        *   `severe` (严重): 模拟极端黑暗（ISO 12800+）下画质严重受损。

### 4. `process_local_overexposure.py` - 局部过曝

*   **功能**: 模拟图像中局部区域因光线过强而导致的过曝及泛光效果。
*   **运行命令行示例**:
    ```bash
    python process_local_overexposure.py \
      --base_folder 6_night_sam3 \
      --main_json valid_overexposure.json \
      --max_images 25 \
      --output_folder local_overexposure_results \
      --num_threads 4 \
      --level medium
    ```

### 5. `process_motion_blur.py` - 运动模糊

*   **功能**: 模拟图像中物体因快速移动而产生的拖影模糊效果。
*   **运行命令行示例**:
    ```bash
    python process_motion_blur.py \
      --base_folder 6_night_sam3 \
      --main_json valid_motion_blur.json \
      --num_threads 4 \
      --max_images 25 \
      --output_folder motion_blur_results \
      --level medium
    ```

### 6. `process_overexposure.py` - 全局过曝

*   **功能**: 模拟图像整体因曝光过度而导致的亮度过高和细节丢失。
*   **运行命令行示例**:
    ```bash
    python process_overexposure.py \
      --base_folder 6_night_sam3 \
      --max_images 25 \
      --output_folder overexposure_results \
      --level medium
    ```

### 7. `process_underexposure.py` - 欠曝

*   **功能**: 模拟图像整体因曝光不足而导致的亮度过低和细节丢失。
*   **运行命令行示例**:
    ```bash
    python process_underexposure.py \
      --base_folder 6_night_sam3 \
      --output_folder underexposure_results \
      --num_threads 4 \
      --max_images 25 \
      --level medium
    ```
    *   **失真处理方式**: 通过调整伽马值、亮度减法和ISO噪声，旨在创建接近全黑且细节丢失的图像。
