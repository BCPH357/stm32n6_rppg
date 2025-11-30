"""
rPPG Heart Rate Detector
- 臉部檢測（Haar Cascade）
- 3 個 ROI 提取（前額、左臉頰、右臉頰）
- Multi-ROI 模型推論
- BVP 累積和心率計算（Welch PSD）
"""

import cv2
import torch
import numpy as np
from collections import deque
from model import UltraLightRPPG


class HeartRateDetector:
    """即時心率檢測器"""

    def __init__(self, model_path='models/best_model.pth'):
        """
        初始化檢測器

        Args:
            model_path: 模型權重路徑
        """
        print("="*60)
        print("Initializing Heart Rate Detector")
        print("="*60)

        # 載入模型
        self.model = self._load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

        # 載入臉部檢測器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
        print(f"Haar Cascade loaded: {cascade_path}")

        # 狀態管理
        self.window_size = 8  # 時間窗口大小
        self.frame_buffer = deque(maxlen=self.window_size)  # 影格緩衝
        self.hr_buffer = deque(maxlen=30)  # HR 預測緩衝（用於平滑，保留最近 30 個預測）
        self.hr_history = []  # 心率歷史（用於圖表顯示）

        # 臉部檢測緩存（減少計算）
        self.last_face_bbox = None
        self.face_detect_interval = 3  # 每 3 幀重新檢測（更頻繁，提升響應性）
        self.frame_count = 0

        # ROI 比例（相對於臉部 bbox）
        self.roi_ratios = {
            'forehead': {'x': (0.20, 0.80), 'y': (0.05, 0.25)},
            'left_cheek': {'x': (0.05, 0.30), 'y': (0.35, 0.65)},
            'right_cheek': {'x': (0.70, 0.95), 'y': (0.35, 0.65)}
        }

        print("[OK] Detector initialized successfully")
        print("="*60)

    def _load_model(self, model_path):
        """載入訓練好的模型"""
        model = UltraLightRPPG(window_size=8, num_rois=3)

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Model loaded: {model_path}")
            print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"   MAE: {checkpoint.get('mae', 'unknown'):.4f} BPM")
        except Exception as e:
            print(f"[Warning] Failed to load model ({e})")
            print("   Using randomly initialized weights")

        model.eval()
        return model

    def detect_face(self, frame):
        """
        檢測臉部區域

        Args:
            frame: BGR 影像

        Returns:
            (x, y, w, h) 或 None
        """
        # 每 N 幀才重新檢測，其他時候使用緩存
        if self.frame_count % self.face_detect_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) > 0:
                # 選擇最大的臉
                face = max(faces, key=lambda f: f[2] * f[3])
                self.last_face_bbox = tuple(face)
                return self.last_face_bbox
            else:
                return self.last_face_bbox
        else:
            return self.last_face_bbox

    def extract_rois(self, frame, face_bbox):
        """
        從臉部區域提取 3 個 ROI

        Args:
            frame: BGR 影像
            face_bbox: (x, y, w, h)

        Returns:
            roi_patches: (3, 36, 36, 3) numpy array
            roi_coords: list of (x1, y1, x2, y2)
        """
        if face_bbox is None:
            # 沒有臉部，返回零
            if self.frame_count % 100 == 0:
                print(f"[Warning] Frame {self.frame_count}: No face detected! Returning zeros.")
            return np.zeros((3, 36, 36, 3), dtype=np.float32), []

        x, y, w, h = face_bbox
        img_h, img_w = frame.shape[:2]

        roi_patches = []
        roi_coords = []

        for roi_name in ['forehead', 'left_cheek', 'right_cheek']:
            ratios = self.roi_ratios[roi_name]

            # 計算 ROI 座標（相對於臉部 bbox）
            x1 = max(0, int(x + ratios['x'][0] * w))
            x2 = min(img_w, int(x + ratios['x'][1] * w))
            y1 = max(0, int(y + ratios['y'][0] * h))
            y2 = min(img_h, int(y + ratios['y'][1] * h))

            roi_coords.append((x1, y1, x2, y2))

            # 裁切 ROI
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                # 調整大小到 36×36
                roi = cv2.resize(roi, (36, 36))
            else:
                # 無效 ROI，使用零
                roi = np.zeros((36, 36, 3), dtype=np.uint8)

            # 歸一化到 [0, 1]
            roi = roi.astype(np.float32) / 255.0
            roi_patches.append(roi)

        # 堆疊為 (3, 36, 36, 3)
        roi_patches = np.stack(roi_patches, axis=0)

        return roi_patches, roi_coords

    def process_frame(self, frame):
        """
        處理單幀影像

        Args:
            frame: BGR 影像

        Returns:
            result: 字典包含 {
                'face_bbox': (x, y, w, h) 或 None,
                'roi_coords': [(x1, y1, x2, y2), ...],
                'hr': 心率 (BPM) 或 None,
                'hr_raw': 當前原始 HR 預測值,
                'frame_count': 影格計數器,
                'hr_history': 心率歷史（用於圖表），
                'status': 狀態訊息
            }
        """
        self.frame_count += 1

        # 1. 檢測臉部
        face_bbox = self.detect_face(frame)

        # 2. 提取 3 個 ROI
        roi_patches, roi_coords = self.extract_rois(frame, face_bbox)

        # 3. 累積到影格緩衝
        self.frame_buffer.append(roi_patches)

        # 4. 如果緩衝區滿了（8 幀），進行推論
        hr_raw = None
        hr_smoothed = None
        status = f"正在收集影格... ({len(self.frame_buffer)}/{self.window_size})"

        if len(self.frame_buffer) == self.window_size:
            # 堆疊為 (8, 3, 36, 36, 3)
            window = np.stack(list(self.frame_buffer), axis=0)

            # 轉換為 tensor: (1, 8, 3, 36, 36, 3)
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)

            # 調試：檢查輸入數據
            if self.frame_count % 50 == 0:  # 每 50 幀打印一次
                print(f"[Debug] Frame {self.frame_count}: input min={window.min():.3f}, max={window.max():.3f}, mean={window.mean():.3f}")
                print(f"[Debug] Face detected: {face_bbox is not None}")
                if face_bbox is not None:
                    print(f"[Debug] Face bbox: {face_bbox}")

            # 推論 - 模型直接輸出 HR (BPM)
            with torch.no_grad():
                hr_raw = self.model(window_tensor).item()

            # 調試：打印模型輸出
            if self.frame_count % 50 == 0:
                print(f"[Debug] Model output (hr_raw): {hr_raw:.2f} BPM")

            # 將原始 HR 加入緩衝區（用於平滑）
            self.hr_buffer.append(hr_raw)

            # 使用移動平均平滑 HR（減少抖動）
            if len(self.hr_buffer) >= 5:  # 至少 5 個樣本才開始平滑
                hr_smoothed = np.median(list(self.hr_buffer)[-10:])  # 使用最近 10 個的中位數
                self.hr_history.append(hr_smoothed)

                # 限制歷史長度（最多 300 個，約 30 秒 @ 10 fps）
                if len(self.hr_history) > 300:
                    self.hr_history.pop(0)

                status = f"心率檢測中... (HR: {hr_smoothed:.1f} BPM)"
            else:
                status = f"正在收集 HR 數據... ({len(self.hr_buffer)}/5)"

        # 返回結果（轉換為原生 Python 類型以支援 JSON 序列化）
        return {
            'face_bbox': [int(x) for x in face_bbox] if face_bbox is not None else None,
            'roi_coords': [[int(x) for x in roi] for roi in roi_coords] if roi_coords else [],
            'hr': float(hr_smoothed) if hr_smoothed is not None else None,
            'hr_raw': float(hr_raw) if hr_raw is not None else None,
            'frame_count': int(self.frame_count),
            'hr_history': [float(h) for h in self.hr_history[-30:]] if len(self.hr_history) > 0 else [],  # 最近 30 個
            'status': status
        }

    def reset(self):
        """重置檢測器狀態"""
        self.frame_buffer.clear()
        self.hr_buffer.clear()
        self.hr_history.clear()
        self.last_face_bbox = None
        self.frame_count = 0
        print("[OK] Detector reset")


def test_detector():
    """測試檢測器（使用虛擬影格）"""
    print("\n" + "="*60)
    print("Testing Heart Rate Detector")
    print("="*60)

    # 創建檢測器
    detector = HeartRateDetector()

    # 測試虛擬影格
    print("\nProcessing 10 dummy frames...")
    for i in range(10):
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(dummy_frame)
        print(f"Frame {i+1}: {result['status']}")

    print("\n[OK] Test completed!")


if __name__ == "__main__":
    test_detector()
