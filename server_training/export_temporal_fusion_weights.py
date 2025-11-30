"""
導出 Temporal Fusion 權重為 C 語言陣列

從 PyTorch 模型導出權重到 C 格式，用於 STM32 CPU 推論

執行方法:
    python export_temporal_fusion_weights.py

輸出:
    temporal_fusion_weights_exported.c
"""

import torch
import numpy as np
from pathlib import Path

# 導入模型定義
from model_split import TemporalFusion


def export_conv1d_weights(state_dict, layer_prefix, out_ch, in_ch, kernel_size, var_prefix):
    """
    導出 Conv1D 權重為 C 陣列

    Args:
        state_dict: PyTorch state_dict
        layer_prefix: 層名前綴（如 'temporal.0'）
        out_ch, in_ch, kernel_size: 維度
        var_prefix: C 變數名前綴（如 'g_conv1'）

    Returns:
        C 代碼字符串
    """
    weight_key = f'{layer_prefix}.weight'
    bias_key = f'{layer_prefix}.bias'

    # PyTorch Conv1D weight shape: (out_ch, in_ch, kernel)
    weight = state_dict[weight_key].numpy()
    bias = state_dict[bias_key].numpy()

    print(f"  [{layer_prefix}]")
    print(f"    Weight shape: {weight.shape}")
    print(f"    Bias shape:   {bias.shape}")

    # 生成 C 陣列
    c_code = f"// {layer_prefix}: ({out_ch}, {in_ch}, {kernel_size})\n"
    c_code += f"const float {var_prefix}_weight[{out_ch}][{in_ch}][{kernel_size}] = {{\n"

    for oc in range(out_ch):
        c_code += "    {\n"
        for ic in range(in_ch):
            c_code += "        {"
            for k in range(kernel_size):
                c_code += f"{weight[oc, ic, k]:.8f}f"
                if k < kernel_size - 1:
                    c_code += ", "
            c_code += "}"
            if ic < in_ch - 1:
                c_code += ","
            c_code += "\n"
        c_code += "    }"
        if oc < out_ch - 1:
            c_code += ","
        c_code += "\n"

    c_code += "};\n\n"

    # Bias
    c_code += f"const float {var_prefix}_bias[{out_ch}] = {{\n    "
    for i, b in enumerate(bias):
        c_code += f"{b:.8f}f"
        if i < len(bias) - 1:
            c_code += ", "
        if (i + 1) % 8 == 0 and i < len(bias) - 1:
            c_code += "\n    "
    c_code += "\n};\n\n"

    return c_code


def export_fc_weights(state_dict, layer_prefix, out_dim, in_dim, var_prefix):
    """
    導出 FC 權重為 C 陣列

    Args:
        state_dict: PyTorch state_dict
        layer_prefix: 層名前綴（如 'fc.0'）
        out_dim, in_dim: 維度
        var_prefix: C 變數名前綴（如 'g_fc1'）

    Returns:
        C 代碼字符串
    """
    weight_key = f'{layer_prefix}.weight'
    bias_key = f'{layer_prefix}.bias'

    # PyTorch Linear weight shape: (out_dim, in_dim)
    weight = state_dict[weight_key].numpy()
    bias = state_dict[bias_key].numpy()

    print(f"  [{layer_prefix}]")
    print(f"    Weight shape: {weight.shape}")
    print(f"    Bias shape:   {bias.shape}")

    # 生成 C 陣列
    c_code = f"// {layer_prefix}: ({out_dim}, {in_dim})\n"
    c_code += f"const float {var_prefix}_weight[{out_dim}][{in_dim}] = {{\n"

    for o in range(out_dim):
        c_code += "    {"
        for i in range(in_dim):
            c_code += f"{weight[o, i]:.8f}f"
            if i < in_dim - 1:
                c_code += ", "
            if (i + 1) % 8 == 0 and i < in_dim - 1:
                c_code += "\n     "
        c_code += "}"
        if o < out_dim - 1:
            c_code += ","
        c_code += "\n"

    c_code += "};\n\n"

    # Bias
    c_code += f"const float {var_prefix}_bias[{out_dim}] = {{\n    "
    for i, b in enumerate(bias):
        c_code += f"{b:.8f}f"
        if i < len(bias) - 1:
            c_code += ", "
    c_code += "\n};\n\n"

    return c_code


def main():
    print("="*70)
    print("導出 Temporal Fusion 權重為 C 語言陣列")
    print("="*70)

    # 載入 PyTorch 模型
    checkpoint_path = "checkpoints/temporal_fusion.pth"
    print(f"\n[載入 PyTorch 模型]")
    print(f"  檢查點: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"[ERROR] 找不到檢查點: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    print(f"  [OK] 模型已載入")
    print(f"  參數量: {checkpoint['num_params']:,}")

    # 生成 C 代碼
    c_code = """/**
 * @file temporal_fusion_weights_exported.c
 * @brief Temporal Fusion 權重（從 PyTorch 導出）
 *
 * 自動生成，請勿手動編輯！
 * 生成腳本: export_temporal_fusion_weights.py
 */

#include "temporal_fusion.h"

/* ============================================================================
 * Conv1D Layer 1 權重: (48 → 32, kernel=3)
 * ========================================================================== */

"""

    print(f"\n[導出權重]")

    # Conv1D Layer 1
    c_code += export_conv1d_weights(
        state_dict,
        'temporal.0',
        out_ch=32, in_ch=48, kernel_size=3,
        var_prefix='g_conv1'
    )

    # Conv1D Layer 2
    c_code += "/* ============================================================================\n"
    c_code += " * Conv1D Layer 2 權重: (32 → 16, kernel=3)\n"
    c_code += " * ========================================================================== */\n\n"

    c_code += export_conv1d_weights(
        state_dict,
        'temporal.2',
        out_ch=16, in_ch=32, kernel_size=3,
        var_prefix='g_conv2'
    )

    # FC Layer 1
    c_code += "/* ============================================================================\n"
    c_code += " * FC Layer 1 權重: (128 → 32)\n"
    c_code += " * ========================================================================== */\n\n"

    c_code += export_fc_weights(
        state_dict,
        'fc.0',
        out_dim=32, in_dim=128,
        var_prefix='g_fc1'
    )

    # FC Layer 2
    c_code += "/* ============================================================================\n"
    c_code += " * FC Layer 2 權重: (32 → 1)\n"
    c_code += " * ========================================================================== */\n\n"

    c_code += export_fc_weights(
        state_dict,
        'fc.2',
        out_dim=1, in_dim=32,
        var_prefix='g_fc2'
    )

    # 寫入檔案
    output_path = "temporal_fusion_weights_exported.c"
    with open(output_path, 'w') as f:
        f.write(c_code)

    print(f"\n[成功]")
    print(f"  已保存: {output_path}")
    print(f"  文件大小: {Path(output_path).stat().st_size / 1024:.2f} KB")

    print(f"\n{'='*70}")
    print(f"導出完成")
    print(f"{'='*70}")
    print(f"\n下一步:")
    print(f"  1. 將 {output_path} 替換到 STM32 項目中")
    print(f"  2. 重新編譯測試程序: gcc -o test test_temporal_fusion.c temporal_fusion.c {output_path} -lm")
    print(f"  3. 運行測試: ./test")
    print(f"  4. 驗證輸出與 PyTorch 一致")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
