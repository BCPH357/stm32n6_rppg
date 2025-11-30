"""
檢查 ubfc_processed.pt 的數據結構

用法:
    python check_data_structure.py
"""

import torch
from pathlib import Path

def main():
    data_path = Path("data/ubfc_processed.pt")

    if not data_path.exists():
        print(f"[ERROR] File not found: {data_path}")
        return

    print("="*70)
    print("Checking ubfc_processed.pt Structure")
    print("="*70)

    # 載入數據
    print(f"\nLoading: {data_path}")
    data = torch.load(data_path)

    # 顯示所有鍵
    print("\n[Keys in dataset]")
    print("-"*70)
    for i, key in enumerate(data.keys(), 1):
        print(f"{i}. {key}")

    # 顯示每個鍵的詳細信息
    print("\n[Detailed Information]")
    print("-"*70)

    for key, value in data.items():
        print(f"\nKey: '{key}'")
        print(f"  Type: {type(value)}")

        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")

            # 如果是張量，顯示統計信息
            if hasattr(value, 'min'):
                try:
                    print(f"  Min: {value.min().item():.4f}")
                    print(f"  Max: {value.max().item():.4f}")
                    print(f"  Mean: {value.mean().item():.4f}")
                    print(f"  Std: {value.std().item():.4f}")
                except:
                    pass
        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")

    print("\n" + "="*70)
    print("Check Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
