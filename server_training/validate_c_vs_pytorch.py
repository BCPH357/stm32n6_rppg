"""
驗證 C 代碼 Temporal Fusion vs PyTorch 模型

對比兩者的輸出差異，確保權重導出正確

執行方法:
    python validate_c_vs_pytorch.py
"""

import torch
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path

# 導入模型定義
from model_split import TemporalFusion


def generate_test_data(num_samples=10, seed=42):
    """
    生成測試數據

    Args:
        num_samples: 測試樣本數
        seed: 隨機種子

    Returns:
        test_data: (num_samples, 24, 16) numpy array
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 生成隨機特徵（模擬 Spatial CNN 輸出）
    test_data = np.random.randn(num_samples, 24, 16).astype(np.float32)

    return test_data


def run_pytorch_inference(test_data):
    """
    運行 PyTorch 推論

    Args:
        test_data: (N, 24, 16) numpy array

    Returns:
        predictions: (N,) numpy array
    """
    print(f"\n[PyTorch 推論]")

    # 載入模型
    checkpoint = torch.load("checkpoints/temporal_fusion.pth", map_location='cpu')
    model = TemporalFusion(window_size=8, num_rois=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 推論
    predictions = []
    with torch.no_grad():
        for data in test_data:
            data_tensor = torch.from_numpy(data).unsqueeze(0)  # (1, 24, 16)
            hr = model(data_tensor)
            predictions.append(hr.item())

    predictions = np.array(predictions)

    print(f"  樣本數: {len(predictions)}")
    print(f"  平均 HR: {predictions.mean():.2f} BPM")
    print(f"  範圍: [{predictions.min():.2f}, {predictions.max():.2f}] BPM")

    return predictions


def create_c_test_program(test_data, output_file="test_c_output.c"):
    """
    創建 C 測試程序，包含測試數據

    Args:
        test_data: (N, 24, 16) numpy array
        output_file: 輸出文件名

    Returns:
        C 程序文件路徑
    """
    num_samples = len(test_data)

    c_code = f"""
#include <stdio.h>
#include "temporal_fusion.h"

// 測試數據 ({num_samples} 個樣本)
const float test_data[{num_samples}][24][16] = {{
"""

    # 寫入測試數據
    for i, sample in enumerate(test_data):
        c_code += "    {\n"
        for t in range(24):
            c_code += "        {"
            for f in range(16):
                c_code += f"{sample[t, f]:.8f}f"
                if f < 15:
                    c_code += ", "
            c_code += "}"
            if t < 23:
                c_code += ","
            c_code += "\n"
        c_code += "    }"
        if i < num_samples - 1:
            c_code += ","
        c_code += "\n"

    c_code += """
};

int main(void) {
    printf("C Temporal Fusion Inference\\n");
    printf("============================\\n");

"""

    # 運行推論並打印結果
    for i in range(num_samples):
        c_code += f"""
    float hr_{i} = temporal_fusion_infer(test_data[{i}]);
    printf("%.8f\\n", hr_{i});
"""

    c_code += """
    return 0;
}
"""

    # 寫入文件
    with open(output_file, 'w') as f:
        f.write(c_code)

    return output_file


def run_c_inference(test_data):
    """
    編譯並運行 C 代碼推論

    Args:
        test_data: (N, 24, 16) numpy array

    Returns:
        predictions: (N,) numpy array
    """
    print(f"\n[C 代碼推論]")

    # 創建 C 測試程序
    c_file = create_c_test_program(test_data, "test_c_validation.c")
    print(f"  已創建 C 測試程序: {c_file}")

    # 編譯
    print(f"  編譯中...")
    compile_cmd = [
        "gcc", "-o", "test_c_validation",
        c_file,
        "temporal_fusion.c",
        "temporal_fusion_weights_exported.c",
        "-lm", "-O2"
    ]

    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] 編譯失敗:")
        print(result.stderr)
        return None

    print(f"  [OK] 編譯成功")

    # 運行
    print(f"  運行推論...")
    result = subprocess.run(["./test_c_validation"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] 運行失敗:")
        print(result.stderr)
        return None

    # 解析輸出
    lines = result.stdout.strip().split('\n')
    predictions = []

    for line in lines:
        line = line.strip()
        if line and not line.startswith('C ') and not line.startswith('='):
            try:
                predictions.append(float(line))
            except ValueError:
                continue

    predictions = np.array(predictions)

    print(f"  [OK] 推論完成")
    print(f"  樣本數: {len(predictions)}")
    print(f"  平均 HR: {predictions.mean():.2f} BPM")
    print(f"  範圍: [{predictions.min():.2f}, {predictions.max():.2f}] BPM")

    return predictions


def compare_results(pytorch_preds, c_preds):
    """
    對比 PyTorch 和 C 代碼的結果

    Args:
        pytorch_preds: PyTorch 預測結果
        c_preds: C 代碼預測結果
    """
    print(f"\n{'='*70}")
    print(f"結果對比")
    print(f"{'='*70}")

    if c_preds is None or len(c_preds) != len(pytorch_preds):
        print(f"[ERROR] C 代碼推論失敗或樣本數不匹配")
        return

    # 計算差異
    diff = np.abs(pytorch_preds - c_preds)
    max_diff = diff.max()
    mean_diff = diff.mean()
    std_diff = diff.std()

    print(f"\n[差異統計]")
    print(f"  最大差異: {max_diff:.8f} BPM")
    print(f"  平均差異: {mean_diff:.8f} BPM")
    print(f"  標準差:   {std_diff:.8f} BPM")

    # 質量評級
    print(f"\n[等價性評級]")
    if max_diff < 1e-5:
        quality = "PERFECT"
        print(f"  質量: {quality} (max_diff < 1e-5)")
        print(f"  ✅ C 代碼與 PyTorch 完全等價！")
    elif max_diff < 1e-3:
        quality = "EXCELLENT"
        print(f"  質量: {quality} (max_diff < 1e-3)")
        print(f"  ✅ C 代碼實現正確")
    elif max_diff < 0.1:
        quality = "GOOD"
        print(f"  質量: {quality} (max_diff < 0.1 BPM)")
        print(f"  ⚠️  有微小差異，但可接受")
    else:
        quality = "POOR"
        print(f"  質量: {quality} (max_diff >= 0.1 BPM)")
        print(f"  ❌ 差異過大，請檢查權重導出！")

    # 樣本對比
    print(f"\n[樣本對比（前 10 個）]")
    print(f"  {'Index':>6} | {'PyTorch':>12} | {'C Code':>12} | {'Diff':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for i in range(min(10, len(pytorch_preds))):
        pt = pytorch_preds[i]
        c = c_preds[i]
        d = abs(pt - c)
        print(f"  {i:6d} | {pt:12.6f} | {c:12.6f} | {d:12.8f}")

    return quality


def main():
    print("="*70)
    print("驗證 C 代碼 vs PyTorch Temporal Fusion")
    print("="*70)

    # 檢查必要文件
    required_files = [
        "checkpoints/temporal_fusion.pth",
        "temporal_fusion.h",
        "temporal_fusion.c",
        "temporal_fusion_weights_exported.c"
    ]

    for f in required_files:
        if not Path(f).exists():
            print(f"[ERROR] 找不到文件: {f}")
            if f == "temporal_fusion_weights_exported.c":
                print("請先運行: python export_temporal_fusion_weights.py")
            return

    # 生成測試數據
    print(f"\n[生成測試數據]")
    num_samples = 20
    test_data = generate_test_data(num_samples=num_samples, seed=42)
    print(f"  生成 {num_samples} 個測試樣本 (24×16)")

    # PyTorch 推論
    pytorch_preds = run_pytorch_inference(test_data)

    # C 代碼推論
    c_preds = run_c_inference(test_data)

    # 對比結果
    quality = compare_results(pytorch_preds, c_preds)

    # 總結
    print(f"\n{'='*70}")
    print(f"[驗證完成]")
    print(f"{'='*70}")

    if quality in ["PERFECT", "EXCELLENT"]:
        print(f"\n✅ C 代碼實現正確，可以部署到 STM32N6")
    elif quality == "GOOD":
        print(f"\n⚠️  有微小差異，建議再次檢查但通常可接受")
    else:
        print(f"\n❌ 差異過大，請檢查：")
        print(f"   1. 權重導出是否正確")
        print(f"   2. C 代碼實現是否有 bug")
        print(f"   3. PyTorch 模型版本是否匹配")

    print(f"\n下一步:")
    print(f"  1. 如果驗證通過，下載文件到本地")
    print(f"  2. 整合到 STM32N6 項目")
    print(f"  3. 在硬件上測試")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
