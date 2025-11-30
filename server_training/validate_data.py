"""
数据验证脚本
验证原始数据集和预处理数据的完整性
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from collections import defaultdict

def validate_raw_ubfc(raw_data_dir):
    """验证原始 UBFC 数据集"""
    print("\n" + "="*60)
    print("🔍 Validating Raw UBFC-rPPG Dataset")
    print("="*60)

    ubfc_dir = Path(raw_data_dir) / "UBFC-rPPG" / "UBFC_DATASET" / "DATASET_2"

    if not ubfc_dir.exists():
        print(f"❌ Error: UBFC DATASET_2 directory not found at {ubfc_dir}")
        print(f"\nExpected path:")
        print(f"  {ubfc_dir}")
        return False

    # Find all subjects
    subjects = sorted(list(ubfc_dir.glob("subject*")))

    if len(subjects) == 0:
        print(f"❌ Error: No subject folders found in {ubfc_dir}")
        return False

    print(f"\n📊 Found {len(subjects)} subjects")

    # Expected: 42-43 subjects
    if len(subjects) < 40:
        print(f"⚠️  Warning: Expected 42-43 subjects, found {len(subjects)}")
    else:
        print(f"✅ Subject count OK ({len(subjects)} subjects)")

    # Validate structure
    print("\n🔍 Validating structure...")
    issues = defaultdict(list)
    valid_count = 0

    for subject_dir in subjects:
        subject_name = subject_dir.name

        # Check for vid.avi
        video_file = subject_dir / "vid.avi"
        if not video_file.exists():
            issues[subject_name].append("Missing vid.avi")

        # Check for ground_truth.txt
        gt_file = subject_dir / "ground_truth.txt"
        if not gt_file.exists():
            issues[subject_name].append("Missing ground_truth.txt")

        # If both exist, count as valid
        if video_file.exists() and gt_file.exists():
            valid_count += 1

    print(f"✅ Valid subjects: {valid_count}/{len(subjects)}")

    # Report issues
    if issues:
        print(f"\n⚠️  Issues found in {len(issues)} subjects:")
        for subject, problems in list(issues.items())[:10]:  # Show first 10
            print(f"  - {subject}: {', '.join(problems)}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    # Show sample
    if valid_count > 0:
        print(f"\n📁 Sample subject structure:")
        sample_subject = next(s for s in subjects if s.name not in issues)
        print(f"  {sample_subject.name}/")
        for item in sample_subject.iterdir():
            size = item.stat().st_size / (1024**2)  # MB
            print(f"    ├── {item.name} ({size:.2f} MB)")

    # Summary
    print(f"\n" + "="*60)
    if valid_count >= 40 and len(issues) == 0:
        print("✅ Raw UBFC dataset validation PASSED")
        return True
    elif valid_count >= 40:
        print("⚠️  Raw UBFC dataset validation PASSED with warnings")
        return True
    else:
        print("❌ Raw UBFC dataset validation FAILED")
        return False

def validate_preprocessed_data(data_dir):
    """验证预处理数据"""
    print("\n" + "="*60)
    print("🔍 Validating Preprocessed Data")
    print("="*60)

    data_file = Path(data_dir) / "ubfc_processed.pt"

    if not data_file.exists():
        print(f"❌ Error: Preprocessed data not found at {data_file}")
        return False

    print(f"\n📁 Loading data from {data_file}")
    print(f"   File size: {data_file.stat().st_size / (1024**3):.2f} GB")

    try:
        data = torch.load(data_file, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

    print("✅ Data loaded successfully")

    # Check keys
    print("\n🔍 Checking data structure...")
    expected_keys = ['samples', 'labels', 'num_samples', 'window_size', 'stride', 'num_rois']
    missing_keys = [k for k in expected_keys if k not in data]

    if missing_keys:
        print(f"⚠️  Warning: Missing keys: {missing_keys}")

    print(f"\n📊 Data contents:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
            print(f"    Size: {value.element_size() * value.nelement() / (1024**2):.2f} MB")
        else:
            print(f"  {key}: {value}")

    # Validate samples shape
    samples = data.get('samples')
    labels = data.get('labels')
    num_rois = data.get('num_rois', 1)

    if samples is None:
        print("\n❌ Error: 'samples' not found in data")
        return False

    if labels is None:
        print("\n❌ Error: 'labels' not found in data")
        return False

    # Expected shape for Multi-ROI: (N, 8, 3, 36, 36, 3)
    # Or for Single-ROI: (N, 8, 3, 36, 36)
    print(f"\n🔍 Validating shapes...")
    print(f"  Samples: {samples.shape}")
    print(f"  Labels:  {labels.shape}")

    if num_rois == 3:
        # Multi-ROI expected shape
        if len(samples.shape) != 6:
            print(f"❌ Error: Expected 6D tensor for Multi-ROI, got {len(samples.shape)}D")
            return False

        N, T, ROI, H, W, C = samples.shape
        expected_shape = (T==8, ROI==3, H==36, W==36, C==3)

        print(f"\n  Breakdown:")
        print(f"    N (samples):     {N}")
        print(f"    T (time):        {T} {'✅' if T==8 else '❌ Expected 8'}")
        print(f"    ROI (regions):   {ROI} {'✅' if ROI==3 else '❌ Expected 3'}")
        print(f"    H (height):      {H} {'✅' if H==36 else '❌ Expected 36'}")
        print(f"    W (width):       {W} {'✅' if W==36 else '❌ Expected 36'}")
        print(f"    C (channels):    {C} {'✅' if C==3 else '❌ Expected 3'}")

        if not all(expected_shape):
            print(f"\n❌ Error: Shape validation failed")
            return False

    else:
        # Single-ROI expected shape: (N, 8, 3, 36, 36)
        if len(samples.shape) != 5:
            print(f"❌ Error: Expected 5D tensor for Single-ROI, got {len(samples.shape)}D")
            return False

    # Validate labels
    if len(labels.shape) != 1:
        print(f"❌ Error: Expected 1D labels, got {len(labels.shape)}D")
        return False

    if samples.shape[0] != labels.shape[0]:
        print(f"❌ Error: Sample count mismatch: {samples.shape[0]} vs {labels.shape[0]}")
        return False

    # Check data statistics
    print(f"\n📈 Data statistics:")
    print(f"  Samples:")
    print(f"    Min:  {samples.min().item():.4f}")
    print(f"    Max:  {samples.max().item():.4f}")
    print(f"    Mean: {samples.mean().item():.4f}")
    print(f"    Std:  {samples.std().item():.4f}")
    print(f"  Labels:")
    print(f"    Min:  {labels.min().item():.4f}")
    print(f"    Max:  {labels.max().item():.4f}")
    print(f"    Mean: {labels.mean().item():.4f}")
    print(f"    Std:  {labels.std().item():.4f}")

    # Warning if data range is unexpected
    if samples.min() < 0 or samples.max() > 1:
        print(f"\n⚠️  Warning: Sample values outside [0, 1] range")

    # Check for NaN/Inf
    if torch.isnan(samples).any():
        print(f"\n❌ Error: Samples contain NaN values")
        return False

    if torch.isinf(samples).any():
        print(f"\n❌ Error: Samples contain Inf values")
        return False

    if torch.isnan(labels).any():
        print(f"\n❌ Error: Labels contain NaN values")
        return False

    if torch.isinf(labels).any():
        print(f"\n❌ Error: Labels contain Inf values")
        return False

    # Summary
    print(f"\n" + "="*60)
    print("✅ Preprocessed data validation PASSED")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {samples.shape[0]}")
    print(f"  Window size: {data.get('window_size', 'N/A')}")
    print(f"  Stride: {data.get('stride', 'N/A')}")
    print(f"  ROIs: {num_rois}")
    print(f"  Data size: {data_file.stat().st_size / (1024**3):.2f} GB")

    return True

def main():
    parser = argparse.ArgumentParser(description='Validate rPPG dataset')
    parser.add_argument('--mode', type=str, choices=['raw', 'preprocessed', 'both'],
                      default='both', help='Validation mode')
    parser.add_argument('--raw_data', type=str, default='raw_data',
                      help='Raw data directory')
    parser.add_argument('--data', type=str, default='data',
                      help='Preprocessed data directory')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("📋 rPPG Data Validation Tool")
    print("="*60)
    print(f"Mode: {args.mode}")

    all_passed = True

    # Validate raw data
    if args.mode in ['raw', 'both']:
        passed = validate_raw_ubfc(args.raw_data)
        all_passed = all_passed and passed

    # Validate preprocessed data
    if args.mode in ['preprocessed', 'both']:
        passed = validate_preprocessed_data(args.data)
        all_passed = all_passed and passed

    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ All validations PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ Some validations FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
