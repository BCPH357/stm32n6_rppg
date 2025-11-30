"""
Multi-ROI 臉部提取測試腳本

功能：
1. 從 UBFC 數據集影片中提取指定幀
2. 運行 detect_and_crop_face() 函數
3. 視覺化結果：
   - 左側：原始幀 + 檢測框和 ROI 標註
   - 右側：3 個提取的 36×36 ROI patches
4. 保存到 test_roi_extraction/ 目錄

使用方法：
    python test_roi_extraction.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# 本地定義 detect_and_crop_face 函數（避免依賴問題）
def detect_and_crop_face(frame, target_size=(36, 36)):
    """
    使用輕量級 Haar Cascade 檢測脸部並提取 3 個 ROI 區域

    Args:
        frame: BGR 圖像 (H, W, 3)
        target_size: 目標尺寸 (height, width)

    Returns:
        multi_roi_patches: numpy array (3, 36, 36, 3) - [forehead, left_cheek, right_cheek]
        或 None（如果檢測失敗）
    """
    # 轉換為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 Haar Cascade 檢測臉部
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # 使用第一個檢測到的臉部
    x, y, w, h = faces[0]

    # 確保邊界框在圖像範圍內
    img_h, img_w = frame.shape[:2]

    # 定義 3 個 ROI 區域
    # Forehead ROI
    fx1 = max(0, int(x + 0.20 * w))
    fx2 = min(img_w, int(x + 0.80 * w))
    fy1 = max(0, int(y + 0.05 * h))
    fy2 = min(img_h, int(y + 0.25 * h))

    # Left Cheek ROI
    lx1 = max(0, int(x + 0.05 * w))
    lx2 = min(img_w, int(x + 0.30 * w))
    ly1 = max(0, int(y + 0.35 * h))
    ly2 = min(img_h, int(y + 0.65 * h))

    # Right Cheek ROI
    rx1 = max(0, int(x + 0.70 * w))
    rx2 = min(img_w, int(x + 0.95 * w))
    ry1 = max(0, int(y + 0.35 * h))
    ry2 = min(img_h, int(y + 0.65 * h))

    # 提取並調整 3 個 ROI patches
    multi_roi_patches = []

    for (x1, x2, y1, y2) in [(fx1, fx2, fy1, fy2), (lx1, lx2, ly1, ly2), (rx1, rx2, ry1, ry2)]:
        if x2 <= x1 or y2 <= y1:
            # 無效 ROI，使用零填充
            roi_patch = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        else:
            # 裁切 ROI
            roi = rgb_frame[y1:y2, x1:x2]

            # 調整大小到 36×36
            roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)

            # 歸一化到 [0, 1]
            roi_normalized = roi_resized.astype(np.float32) / 255.0

            roi_patch = roi_normalized

        multi_roi_patches.append(roi_patch)

    # 返回形狀 (3, 36, 36, 3)
    return np.array(multi_roi_patches)


def extract_frame(video_path, frame_number):
    """
    從影片中提取指定幀

    Args:
        video_path: 影片路徑
        frame_number: 幀號（從 0 開始）

    Returns:
        frame: BGR 圖像，或 None（如果失敗）
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ 無法打開影片: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames:
        print(f"⚠️ 幀號 {frame_number} 超出範圍（總幀數: {total_frames}）")
        cap.release()
        return None

    # 設置到目標幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    cap.release()

    if not ret:
        print(f"❌ 無法讀取幀 {frame_number}")
        return None

    return frame


def detect_face_and_rois(frame):
    """
    檢測臉部和 ROI 區域（用於繪圖）

    Returns:
        face_bbox: (x, y, w, h) 或 None
        roi_coords: [(fx1,fx2,fy1,fy2), (lx1,lx2,ly1,ly2), (rx1,rx2,ry1,ry2)] 或 None
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade 檢測
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None

    # 使用第一個檢測到的臉部
    x, y, w, h = faces[0]
    face_bbox = (x, y, w, h)

    img_h, img_w = frame.shape[:2]

    # 計算 3 個 ROI 區域（與 preprocess_data.py 相同）
    # Forehead ROI
    fx1 = max(0, int(x + 0.20 * w))
    fx2 = min(img_w, int(x + 0.80 * w))
    fy1 = max(0, int(y + 0.05 * h))
    fy2 = min(img_h, int(y + 0.25 * h))

    # Left Cheek ROI
    lx1 = max(0, int(x + 0.05 * w))
    lx2 = min(img_w, int(x + 0.30 * w))
    ly1 = max(0, int(y + 0.35 * h))
    ly2 = min(img_h, int(y + 0.65 * h))

    # Right Cheek ROI
    rx1 = max(0, int(x + 0.70 * w))
    rx2 = min(img_w, int(x + 0.95 * w))
    ry1 = max(0, int(y + 0.35 * h))
    ry2 = min(img_h, int(y + 0.65 * h))

    roi_coords = [
        (fx1, fx2, fy1, fy2),  # Forehead
        (lx1, lx2, ly1, ly2),  # Left Cheek
        (rx1, rx2, ry1, ry2),  # Right Cheek
    ]

    return face_bbox, roi_coords


def visualize_roi_extraction(frame, frame_number, output_dir):
    """
    視覺化 ROI 提取過程

    Args:
        frame: BGR 圖像
        frame_number: 幀號（用於檔名）
        output_dir: 輸出目錄
    """
    # 1. 檢測臉部和 ROI 區域
    face_bbox, roi_coords = detect_face_and_rois(frame)

    if face_bbox is None:
        print(f"  [WARN] Frame {frame_number}: No face detected")
        return False

    # 2. 提取 ROI patches（使用原始函數）
    roi_patches = detect_and_crop_face(frame, target_size=(36, 36))

    if roi_patches is None:
        print(f"  [WARN] Frame {frame_number}: ROI extraction failed")
        return False

    # 3. 創建視覺化圖像
    fig = plt.figure(figsize=(16, 7))

    # === 左側：原始幀 + 標註 ===
    ax_left = plt.subplot(1, 2, 1)

    # 轉換 BGR → RGB 用於顯示
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax_left.imshow(frame_rgb)

    # 繪製臉部檢測框（綠色虛線）
    x, y, w, h = face_bbox
    face_rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime',
                                   facecolor='none', linestyle='--', label='Face BBox')
    ax_left.add_patch(face_rect)

    # 繪製 3 個 ROI 區域
    roi_colors = ['red', 'blue', 'orange']
    roi_labels = ['Forehead', 'Left Cheek', 'Right Cheek']

    for (x1, x2, y1, y2), color, label in zip(roi_coords, roi_colors, roi_labels):
        roi_w = x2 - x1
        roi_h = y2 - y1

        roi_rect = patches.Rectangle((x1, y1), roi_w, roi_h, linewidth=2,
                                       edgecolor=color, facecolor='none', label=label)
        ax_left.add_patch(roi_rect)

        # 添加文字標籤
        ax_left.text(x1, y1 - 5, label, color=color, fontsize=10,
                     fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=2))

    ax_left.set_title(f'Frame {frame_number} - Face Detection & ROI Regions', fontsize=12, fontweight='bold')
    ax_left.axis('off')
    ax_left.legend(loc='upper right', fontsize=8)

    # === 右側：3 個 ROI Patches ===
    # 創建 3×1 子圖
    for i, (label, color) in enumerate(zip(roi_labels, roi_colors)):
        ax = plt.subplot(3, 2, 2 + i * 2)

        # roi_patches[i] 是 (36, 36, 3)，已歸一化到 [0, 1]
        patch_img = roi_patches[i]

        ax.imshow(patch_img)
        ax.set_title(f'{label} ROI (36×36)', fontsize=10, fontweight='bold', color=color)
        ax.axis('off')

    # 調整布局
    plt.tight_layout()

    # 4. 保存圖片
    output_path = Path(output_dir) / f"frame_{frame_number:04d}_roi_extraction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: {output_path.name}")
    return True


def main():
    """主函數"""
    print("\n" + "="*70)
    print("Multi-ROI Face Extraction Test Tool")
    print("="*70)

    # 配置
    VIDEO_PATH = Path(r"D:\UBFC_dataset\DATASET2\subject26\vid.avi")
    OUTPUT_DIR = Path(r"D:\MIAT\rppg\test_roi_extraction")
    TEST_FRAMES = [10, 100, 200, 300, 400]  # 測試的幀號

    # 檢查影片是否存在
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found")
        print(f"   Path: {VIDEO_PATH}")
        print(f"\nPlease ensure UBFC dataset is downloaded to D:\\UBFC_dataset\\")
        return

    # 創建輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Video Path: {VIDEO_PATH}")

    # 獲取影片總幀數
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video Info:")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {total_frames / fps:.2f} sec")
    print(f"\nTest Frames: {TEST_FRAMES}")
    print(f"\nProcessing...\n")

    # 處理每一幀
    success_count = 0

    for frame_num in TEST_FRAMES:
        print(f"[{frame_num}] Extracting frame {frame_num}...", end=" ")

        # 檢查幀號是否有效
        if frame_num >= total_frames:
            print(f"[WARN] Out of range (max: {total_frames-1})")
            continue

        # 提取幀
        frame = extract_frame(VIDEO_PATH, frame_num)

        if frame is None:
            print("[ERROR] Extraction failed")
            continue

        print(f"OK ({frame.shape[1]}x{frame.shape[0]})")

        # 視覺化並保存
        if visualize_roi_extraction(frame, frame_num, OUTPUT_DIR):
            success_count += 1

    # 完成總結
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Success: {success_count}/{len(TEST_FRAMES)} frames")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nView results:")
    print(f"  explorer {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
