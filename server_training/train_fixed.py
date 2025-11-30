"""
rPPG 模型訓練腳本 - 修復版
修復內容：
1. 添加標籤歸一化（Z-score normalization）
2. 修正 MAPE 計算（避免除以接近 0 的值）
3. 添加訓練/驗證統計信息
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import json

# 導入模型
from model import UltraLightRPPG


class rPPGDataset(Dataset):
    """rPPG 數據集類（帶歸一化）"""
    def __init__(self, data_path, normalize=True):
        """
        Args:
            data_path: .pt 文件路徑
            normalize: 是否對標籤進行歸一化
        """
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path)

        self.samples = data['samples']  # (N, 8, 3, 36, 36, 3) or (N, 8, 3, 36, 36)
        self.labels = data['labels']    # (N,)

        # 計算並保存歸一化參數
        self.label_mean = self.labels.mean().item()
        self.label_std = self.labels.std().item()

        print(f"  ✅ Loaded {len(self.samples)} samples")
        print(f"  Label statistics (raw):")
        print(f"    Min:  {self.labels.min():.4f}")
        print(f"    Max:  {self.labels.max():.4f}")
        print(f"    Mean: {self.label_mean:.4f}")
        print(f"    Std:  {self.label_std:.4f}")

        # 歸一化標籤（Z-score）
        if normalize:
            self.labels = (self.labels - self.label_mean) / (self.label_std + 1e-8)
            print(f"  ✅ Labels normalized (Z-score)")
            print(f"    Normalized mean: {self.labels.mean():.4f}")
            print(f"    Normalized std:  {self.labels.std():.4f}")

        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def denormalize(self, normalized_labels):
        """將歸一化的標籤轉換回原始尺度"""
        if self.normalize:
            return normalized_labels * self.label_std + self.label_mean
        return normalized_labels


def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for samples, labels in pbar:
        samples = samples.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # 混合精度訓練
        with autocast():
            outputs = model(samples)  # (B, 1)
            loss = criterion(outputs, labels)

        # 反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, dataset):
    """
    驗證

    Args:
        dataset: 用於反歸一化的數據集對象
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for samples, labels in tqdm(val_loader, desc="Validation"):
            samples = samples.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    # 收集所有預測和標籤
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # 反歸一化到原始 BVP 尺度（用於解釋性指標）
    all_preds_denorm = dataset.denormalize(all_preds)
    all_labels_denorm = dataset.denormalize(all_labels)

    # 計算歸一化空間的指標（用於訓練監控）
    mae_norm = np.mean(np.abs(all_preds - all_labels))
    rmse_norm = np.sqrt(np.mean((all_preds - all_labels)**2))

    # 計算原始空間的指標（用於解釋）
    mae_raw = np.mean(np.abs(all_preds_denorm - all_labels_denorm))
    rmse_raw = np.sqrt(np.mean((all_preds_denorm - all_labels_denorm)**2))

    # 修正的 MAPE（避免除以接近 0 的值）
    # 使用絕對值 + epsilon 作為分母
    epsilon = 0.1  # 對於 BVP 信號合理的 epsilon
    mape = np.mean(np.abs((all_preds_denorm - all_labels_denorm) / (np.abs(all_labels_denorm) + epsilon))) * 100

    return avg_loss, mae_norm, rmse_norm, mae_raw, rmse_raw, mape


class EarlyStopping:
    """Early stopping 類"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)
    print("="*60)
    print("Configuration")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False))

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 創建輸出目錄
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入數據集
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)

    datasets = []
    for data_path in config['data_paths']:
        if Path(data_path).exists():
            ds = rPPGDataset(data_path, normalize=True)  # 啟用歸一化
            datasets.append(ds)
        else:
            print(f"[WARN] {data_path} not found, skipping")

    if not datasets:
        raise ValueError("No datasets loaded!")

    # 使用第一個數據集的歸一化參數（假設所有數據集類似）
    ref_dataset = datasets[0]

    # 合併數據集
    full_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"\nTotal samples: {len(full_dataset)}")

    # 劃分訓練/驗證集
    train_size = int(config['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 創建模型
    print("\n" + "="*60)
    print("Model")
    print("="*60)

    model = UltraLightRPPG(window_size=config['window_size'])
    model = model.to(device)

    total_params = model.get_num_params()
    print(f"Total parameters: {total_params:,}")

    # 損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 混合精度
    scaler = GradScaler()

    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    # 訓練循環
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    best_val_loss = float('inf')
    train_history = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)

        start_time = time.time()

        # 訓練
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # 驗證（傳入 ref_dataset 用於反歸一化）
        val_loss, mae_norm, rmse_norm, mae_raw, rmse_raw, mape = validate(
            model, val_loader, criterion, device, ref_dataset
        )

        # 學習率調度
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # 打印結果
        print(f"\nResults:")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  MAE (norm):      {mae_norm:.4f}  <- normalized space")
        print(f"  RMSE (norm):     {rmse_norm:.4f}")
        print(f"  MAE (BVP):       {mae_raw:.4f}  <- raw BVP scale")
        print(f"  RMSE (BVP):      {rmse_raw:.4f}")
        print(f"  MAPE (fixed):    {mape:.2f}%")
        print(f"  Time:            {epoch_time:.1f}s")

        # 記錄歷史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mae_norm': mae_norm,
            'rmse_norm': rmse_norm,
            'mae_raw': mae_raw,
            'rmse_raw': rmse_raw,
            'mape': mape,
            'time': epoch_time
        })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mae_raw': mae_raw,
                'rmse_raw': rmse_raw,
                'label_mean': ref_dataset.label_mean,
                'label_std': ref_dataset.label_std,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  [OK] Best model saved!")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n[STOP] Early stopping triggered at epoch {epoch+1}")
            break

    # 保存訓練歷史
    with open(output_dir / 'train_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"\nNOTE: This model predicts BVP values, not heart rate!")
    print(f"To get HR, you need to apply FFT/Welch to BVP sequence.")


if __name__ == "__main__":
    main()
