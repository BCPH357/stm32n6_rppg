"""
rPPG 数据预处理脚本 - UBFC 专用版 (Multi-ROI)
仅支持 UBFC-rPPG 数据集
使用 3 个 ROI 区域：前额 + 左脸颊 + 右脸颊
在服务器 CPU 上运行（无需 CUDA）

修正版 v2：使用健壮的 PPG → HR 流程
- Bandpass filter (0.7-3.0 Hz)
- 改良 peak detection (prominence + width)
- 三层 HR 清洗机制
- 严格范围控制 (40-160 BPM)
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.signal import find_peaks, butter, sosfiltfilt
from scipy.interpolate import interp1d

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Butterworth bandpass filter - 滤除 PPG 噪声

    Args:
        data: PPG 信号
        lowcut: 低频截止 (Hz) - 例如 0.7 Hz (42 BPM)
        highcut: 高频截止 (Hz) - 例如 3.0 Hz (180 BPM)
        fs: 采样率 (Hz)
        order: 滤波器阶数

    Returns:
        filtered_data: 滤波后的 PPG 信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 使用 second-order sections (sos) 以提高数值稳定性
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data

def detect_and_crop_face(frame, target_size=(36, 36)):
    """
    使用轻量级 Haar Cascade 检测脸部并提取 3 个 ROI 区域

    Args:
        frame: BGR 图像 (H, W, 3)
        target_size: 目标尺寸 (height, width)

    Returns:
        multi_roi_patches: numpy array (3, 36, 36, 3) - [forehead, left_cheek, right_cheek]
        或 None（如果检测失败）
    """
    # 转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 Haar Cascade 检测脸部
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # 使用第一个检测到的脸部
    x, y, w, h = faces[0]

    # 确保边界框在图像范围内
    img_h, img_w = frame.shape[:2]

    # 定义 3 个 ROI 区域
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

    # 提取并处理每个 ROI
    roi_patches = []

    for (x1, x2, y1, y2) in [(fx1, fx2, fy1, fy2), (lx1, lx2, ly1, ly2), (rx1, rx2, ry1, ry2)]:
        # 边界检查
        if x2 <= x1 or y2 <= y1:
            # 如果 ROI 无效，使用零填充
            roi_patch = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        else:
            # 裁切 ROI
            roi = rgb_frame[y1:y2, x1:x2]

            if roi.size == 0:
                roi_patch = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
            else:
                # 调整大小到 36×36
                roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

                # 归一化到 [0, 1]
                roi_patch = roi_resized.astype(np.float32) / 255.0

        roi_patches.append(roi_patch)

    # 堆叠成 (3, 36, 36, 3)
    multi_roi_patches = np.stack(roi_patches, axis=0)

    return multi_roi_patches

def calculate_hr_from_ppg(ppg_signal, timestamps, fps, video_frame_count):
    """
    从 PPG 信号使用健壮的峰值检测计算逐帧心率（改进版）

    改进点：
    1. Bandpass filter (0.7-3.0 Hz) 去除噪声
    2. 改良 peak detection (prominence + width)
    3. 严格的 RR interval 过滤 (0.3-1.5 秒)
    4. 三层 HR 清洗机制
    5. 强制范围控制 (40-160 BPM)

    Args:
        ppg_signal: numpy array - PPG/BVP waveform (Line 1)
        timestamps: numpy array - 时间戳 (Line 3, seconds)
        fps: float - 视频帧率
        video_frame_count: int - 视频总帧数

    Returns:
        hr_per_frame: numpy array (video_frame_count,) - 每帧的心率 BPM
    """
    # 检查输入
    if len(ppg_signal) == 0 or len(timestamps) == 0:
        raise ValueError("Empty PPG signal or timestamps")

    if len(ppg_signal) != len(timestamps):
        raise ValueError(f"PPG length ({len(ppg_signal)}) != timestamps length ({len(timestamps)})")

    # 计算 PPG 采样率
    ppg_fs = len(ppg_signal) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 30.0

    # Step 1: Bandpass filter (0.7-3.0 Hz)
    # 0.7 Hz = 42 BPM, 3.0 Hz = 180 BPM
    try:
        filtered_ppg = butter_bandpass_filter(ppg_signal, lowcut=0.7, highcut=3.0, fs=ppg_fs, order=3)
    except Exception as e:
        print(f"  [Warning] Bandpass filter failed: {e}, using raw PPG")
        filtered_ppg = ppg_signal

    # Step 2: 改良 peak detection
    # distance: 至少 0.35 秒间隔（防止误检，对应最大 ~170 BPM）
    # prominence: 峰值显著性（避免小波动被误认为峰值）
    # width: 峰值宽度（确保峰值形状合理）
    min_peak_distance = int(0.35 * ppg_fs)

    peaks, properties = find_peaks(
        filtered_ppg,
        distance=min_peak_distance,
        prominence=0.1,  # 峰值显著性
        width=3          # 峰值宽度
    )

    if len(peaks) < 2:
        # 如果找不到足够的峰值，返回默认心率
        print(f"  [Warning] Only {len(peaks)} peaks found, using default HR=75 BPM")
        return np.full(video_frame_count, 75.0, dtype=np.float32)

    # Step 3: 计算 RR intervals（秒）
    peak_times = timestamps[peaks]
    rr_intervals = np.diff(peak_times)  # 相邻峰值的时间差

    # Step 4: RR interval 过滤（第一层清洗）
    # 只保留 0.3 < RR < 1.5 秒（对应 40-200 BPM）
    valid_rr_mask = (rr_intervals >= 0.3) & (rr_intervals <= 1.5)

    if np.sum(valid_rr_mask) < 2:
        print(f"  [Warning] Too few valid RR intervals, using default HR=75 BPM")
        return np.full(video_frame_count, 75.0, dtype=np.float32)

    valid_rr = rr_intervals[valid_rr_mask]

    # Step 5: 转换为 HR (BPM)
    hr_at_peaks = 60.0 / valid_rr  # HR = 60 / RR

    # Step 6: HR 计算后过滤（第二层清洗）
    # 只保留 40-160 BPM
    valid_hr_mask = (hr_at_peaks >= 40) & (hr_at_peaks <= 160)

    if np.sum(valid_hr_mask) < 2:
        print(f"  [Warning] Too few valid HR values, using default HR=75 BPM")
        return np.full(video_frame_count, 75.0, dtype=np.float32)

    valid_hr = hr_at_peaks[valid_hr_mask]

    # 对应的时间点（两个峰值的中点）
    # 需要重新计算，因为经过了两次过滤
    valid_peak_times = peak_times[:-1][valid_rr_mask][valid_hr_mask]
    valid_peak_times_next = peak_times[1:][valid_rr_mask][valid_hr_mask]
    valid_times = (valid_peak_times + valid_peak_times_next) / 2.0

    # Step 7: 创建视频帧的时间轴
    frame_times = np.linspace(timestamps[0], timestamps[-1], video_frame_count)

    # Step 8: 插值到每一帧
    if len(valid_hr) < 2:
        # 如果只有 1 个有效 HR，使用常数
        hr_per_frame = np.full(video_frame_count, valid_hr[0], dtype=np.float32)
    else:
        # 使用线性插值（外推使用边界值）
        interp_func = interp1d(valid_times, valid_hr, kind='linear',
                               bounds_error=False, fill_value=(valid_hr[0], valid_hr[-1]))
        hr_per_frame = interp_func(frame_times).astype(np.float32)

    # Step 9: 插值后强制清洗（第三层清洗）
    # 确保所有值都在合理范围，处理 NaN/inf
    hr_per_frame = np.nan_to_num(hr_per_frame, nan=75.0, posinf=160.0, neginf=40.0)
    hr_per_frame = np.clip(hr_per_frame, 40, 160)

    return hr_per_frame

def process_ubfc_video(video_path, ground_truth_path):
    """
    处理 UBFC 视频文件 (Multi-ROI) - 使用健壮的 PPG peak-based HR 计算

    Returns:
        frames: numpy array (T, 3, 36, 36, 3) - T frames × 3 ROIs × 36×36×3
        hr_labels: numpy array (T,) - 每帧的心率 BPM（从 PPG 峰值计算，带 bandpass filter）
        fps: float - 视频帧率
    """
    # 读取视频
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取 ground truth
    # UBFC DATASET_2 格式:
    #   Line 1: PPG signal (BVP waveform) ← 使用这个！
    #   Line 2: Heart rate (低采样率，不用)
    #   Line 3: Timestep (seconds) ← 用于时间对齐
    with open(ground_truth_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"Invalid ground_truth.txt format (need 3 lines) in {ground_truth_path}")

    # 读取 Line 1（PPG signal）
    ppg_line = lines[0].strip()
    ppg_values = [float(x) for x in ppg_line.split()]
    ppg_signal = np.array(ppg_values)

    # 读取 Line 3（Timestamps）
    ts_line = lines[2].strip()
    ts_values = [float(x) for x in ts_line.split()]
    timestamps = np.array(ts_values)

    # 检查数据有效性
    if len(ppg_signal) == 0:
        raise ValueError(f"No valid PPG data found in {ground_truth_path}")

    if len(timestamps) == 0:
        raise ValueError(f"No valid timestamp data found in {ground_truth_path}")

    # 处理每一帧
    frames = []
    frame_idx = 0

    with tqdm(total=total_frames, desc=f"  处理视频") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测并提取 3 个 ROI 区域
            multi_roi = detect_and_crop_face(frame)

            if multi_roi is not None:
                frames.append(multi_roi)  # Shape: (3, 36, 36, 3)
            else:
                # 如果检测失败，使用全黑帧
                frames.append(np.zeros((3, 36, 36, 3), dtype=np.float32))

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # 使用健壮的 PPG 峰值计算逐帧心率
    print(f"  计算 robust peak-based HR (PPG samples: {len(ppg_signal)}, frames: {len(frames)})...")
    hr_labels = calculate_hr_from_ppg(ppg_signal, timestamps, fps, len(frames))

    return np.array(frames), hr_labels, fps

def create_temporal_samples(frames, hr_labels, window_size=8, stride=1, hr_min=40, hr_max=160):
    """
    创建时间窗口样本 (Multi-ROI) - 增强版：严格过滤异常 HR

    Args:
        frames: (T, 3, 36, 36, 3) - T frames × 3 ROIs × 36×36×3
        hr_labels: (T,) - 每帧的心率标签（从 PPG 峰值计算，带 bandpass filter）
        window_size: 时间窗口大小
        stride: 滑动步长
        hr_min: 最小合理心率 (BPM) - 改为 40
        hr_max: 最大合理心率 (BPM) - 改为 160

    Returns:
        samples: list of (window_size, 3, 36, 36, 3)
        labels: list of float (心率 BPM)
        stats: dict - 统计信息
    """
    samples = []
    labels = []
    filtered_count = 0

    T = len(frames)
    for i in range(0, T - window_size + 1, stride):
        # 提取窗口
        window = frames[i:i+window_size]  # Shape: (8, 3, 36, 36, 3)

        # 使用窗口中间帧的心率作为标签
        mid_idx = i + window_size // 2
        label = hr_labels[mid_idx]

        # 检查窗口内的所有 HR 是否合理
        window_hrs = hr_labels[i:i+window_size]

        # 过滤条件（更严格）：
        # 1. 中间帧 HR 在合理范围 (40-160 BPM)
        # 2. 窗口内所有 HR 都在合理范围
        # 3. 窗口内 HR 变化不要太大（标准差 < 15）
        if (hr_min <= label <= hr_max and
            np.all((window_hrs >= hr_min) & (window_hrs <= hr_max)) and
            np.std(window_hrs) < 15):
            samples.append(window)
            labels.append(label)
        else:
            filtered_count += 1

    stats = {
        'total_windows': T - window_size + 1,
        'valid_samples': len(samples),
        'filtered_samples': filtered_count,
        'filter_ratio': filtered_count / max(1, T - window_size + 1)
    }

    return samples, labels, stats

def process_ubfc_dataset(raw_data_dir, output_dir, window_size=8, stride=1):
    """处理完整的 UBFC 数据集 - 使用健壮的 PPG peak-based HR"""
    print("\n" + "="*60)
    print("📊 处理 UBFC-rPPG 数据集 (Robust Peak-based HR)")
    print("="*60)

    ubfc_dir = Path(raw_data_dir) / "UBFC-rPPG" / "UBFC_DATASET" / "DATASET_2"

    if not ubfc_dir.exists():
        print(f"❌ Error: UBFC DATASET_2 directory not found at {ubfc_dir}")
        print(f"\nPlease ensure the dataset is downloaded to:")
        print(f"  {ubfc_dir}")
        print(f"\nExpected structure:")
        print(f"  raw_data/UBFC-rPPG/UBFC_DATASET/DATASET_2/subject1/")
        print(f"  raw_data/UBFC-rPPG/UBFC_DATASET/DATASET_2/subject3/")
        print(f"  ...")
        print(f"\nRun: bash download_ubfc.sh")
        sys.exit(1)

    subjects = sorted(list(ubfc_dir.glob("subject*")))

    print(f"发现 {len(subjects)} 个受试者")

    if len(subjects) == 0:
        print(f"❌ Error: No subjects found in {ubfc_dir}")
        sys.exit(1)

    all_samples = []
    all_labels = []
    total_filtered = 0
    total_windows = 0

    for subject_dir in subjects:
        print(f"\n处理 {subject_dir.name}...")

        # 找到视频文件和 ground truth 文件
        video_file = subject_dir / "vid.avi"
        gt_file = subject_dir / "ground_truth.txt"

        if not video_file.exists() or not gt_file.exists():
            print(f"  ⚠️  跳过（缺少文件）")
            continue

        try:
            # 处理视频（使用健壮的 PPG peak-based HR）
            frames, hr_labels, fps = process_ubfc_video(video_file, gt_file)

            # 创建时间窗口样本（带严格 HR 过滤）
            samples, labels, stats = create_temporal_samples(
                frames, hr_labels, window_size, stride
            )

            all_samples.extend(samples)
            all_labels.extend(labels)
            total_filtered += stats['filtered_samples']
            total_windows += stats['total_windows']

            print(f"  ✅ 生成 {len(samples)} 个样本 (过滤 {stats['filtered_samples']} 个异常窗口)")

        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            continue

    # 转换为 numpy 数组
    all_samples = np.array(all_samples)  # (N, 8, 3, 36, 36, 3)
    all_labels = np.array(all_labels)    # (N,)

    # 转换为 PyTorch tensor
    samples_tensor = torch.from_numpy(all_samples).float()
    labels_tensor = torch.from_numpy(all_labels).float()

    # 维度顺序：保持 (N, T, ROI, H, W, C) -> (N, 8, 3, 36, 36, 3)
    # 不需要 permute，因为模型会处理这个形状

    # 保存
    output_path = Path(output_dir) / "ubfc_processed.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 打印标签统计
    print(f"\n📊 标签统计（Robust Peak-based HR）：")
    print(f"   最小值：{labels_tensor.min():.2f} BPM")
    print(f"   最大值：{labels_tensor.max():.2f} BPM")
    print(f"   平均值：{labels_tensor.mean():.2f} BPM")
    print(f"   标准差：{labels_tensor.std():.2f} BPM")
    print(f"\n过滤统计：")
    print(f"   总窗口数：{total_windows}")
    print(f"   有效样本：{len(all_samples)} ({100*(1-total_filtered/max(1,total_windows)):.1f}%)")
    print(f"   过滤样本：{total_filtered} ({100*total_filtered/max(1,total_windows):.1f}%)")

    torch.save({
        'samples': samples_tensor,  # (N, 8, 3, 36, 36, 3)
        'labels': labels_tensor,    # (N,) - 心率 BPM (robust peak-based)
        'num_samples': len(all_samples),
        'window_size': window_size,
        'stride': stride,
        'num_rois': 3,  # Multi-ROI marker
        'label_type': 'heart_rate_bpm_robust_peak_based',  # 标记标签类型
        'hr_calculation_method': 'ppg_bandpass_peak_detection_triple_clean'  # 计算方法
    }, output_path)

    print(f"\n✅ UBFC 数据保存到：{output_path}")
    print(f"   样本数：{len(all_samples)}")
    print(f"   形状：{samples_tensor.shape}")
    print(f"   标签：心率 BPM (robust peak-based, bandpass + triple clean)")
    print(f"   大小：{output_path.stat().st_size / (1024**3):.2f} GB")

    return len(all_samples)

def main():
    parser = argparse.ArgumentParser(description='rPPG 数据预处理 (UBFC 专用) - Robust Peak-based HR')
    parser.add_argument('--dataset', type=str, choices=['ubfc'], default='ubfc',
                      help='数据集 (仅支持 UBFC)')
    parser.add_argument('--raw_data', type=str, default='raw_data',
                      help='原始数据目录')
    parser.add_argument('--output', type=str, default='data',
                      help='输出目录')
    parser.add_argument('--window_size', type=int, default=8,
                      help='时间窗口大小')
    parser.add_argument('--stride', type=int, default=1,
                      help='滑动步长')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔧 rPPG 数据预处理工具 (UBFC - Robust Peak-based HR)")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"原始数据: {args.raw_data}")
    print(f"输出目录: {args.output}")
    print(f"时间窗口: {args.window_size} 帧")
    print(f"滑动步长: {args.stride} 帧")
    print(f"HR 计算方法: PPG bandpass + robust peak detection + triple clean")
    print(f"HR 范围限制: 40-160 BPM (严格)")

    # 处理 UBFC
    total_samples = process_ubfc_dataset(
        args.raw_data, args.output, args.window_size, args.stride
    )

    # 保存数据集信息
    info = {
        'dataset': 'ubfc',
        'total_samples': total_samples,
        'window_size': args.window_size,
        'stride': args.stride,
        'image_size': [36, 36],
        'channels': 3,
        'num_rois': 3,
        'hr_method': 'ppg_robust_peak_based',
        'hr_range': [40, 160]
    }

    info_path = Path(args.output) / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("🎉 预处理完成！")
    print("="*60)
    print(f"总样本数: {total_samples}")
    print(f"数据信息: {info_path}")
    print(f"\n下一步：运行 bash run_training.sh 开始训练")

if __name__ == "__main__":
    main()
