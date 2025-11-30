# rPPG 超轻量模型训练 - 服务器端全流程

完整的 rPPG (Remote Photoplethysmography) 训练流程，针对 STM32N6 部署优化。

**所有步骤均在服务器端运行（无需本地预处理）**

---

## 📁 项目结构

```
D:\MIAT\rppg\
├── README.md                   # 本文件
├── CLAUDE.md                    # 项目文档与历史记录
├── model.py                     # 模型架构参考
│
└── server_training/             # ✅ 服务器端工作目录（上传此文件夹到 server）
    ├── download_ubfc.sh         # 下载 UBFC 数据集
    ├── preprocess_data.py       # 数据预处理
    ├── train.py                 # 训练主脚本
    ├── model.py                 # 模型定义
    ├── validate_data.py         # 数据验证工具
    ├── config.yaml              # 训练配置
    ├── requirements.txt         # Python 依赖
    ├── environment.yml          # Conda 环境配置
    ├── setup_env.sh             # 环境设置脚本
    ├── run_training.sh          # 训练启动脚本
    ├── run_all.sh               # 一键运行所有步骤
    │
    ├── raw_data/                # 原始数据集（下载后存放）
    │   └── UBFC-rPPG/
    │       └── subject*/
    │
    ├── data/                    # 预处理数据
    │   ├── ubfc_processed.pt
    │   └── dataset_info.json
    │
    ├── checkpoints/             # 训练输出
    │   ├── best_model.pth
    │   └── train_history.json
    │
    └── logs/                    # 训练日志
        └── train_*.log
```

---

## 🚀 快速开始

### 前置要求

- Linux 服务器（推荐 Ubuntu 20.04+）
- NVIDIA GPU（推荐 A6000，至少 RTX 3090）
- CUDA 12.1
- Conda / Miniconda
- 约 10 GB 磁盘空间

### 一键运行（推荐）

```bash
# 1. 拷贝项目到服务器
scp -r server_training username@server:/path/to/rppg_training/

# 2. SSH 到服务器
ssh username@server
cd /path/to/rppg_training/

# 3. 设置环境
bash setup_env.sh

# 4. 一键运行所有步骤（下载 → 预处理 → 训练）
bash run_all.sh
```

**预计总耗时**: 4-6 小时

---

## 📖 详细步骤

### Step 1: 上传项目到服务器

```bash
# 在本地 Windows 执行
cd D:\MIAT\rppg
scp -r server_training username@server_ip:/home/username/rppg_training/

# 或使用 rsync（支持断点续传）
rsync -avz --progress server_training/ username@server_ip:/home/username/rppg_training/
```

### Step 2: 环境设置（一次性）

```bash
# SSH 到服务器
ssh username@server_ip
cd /home/username/rppg_training/

# 运行设置脚本
bash setup_env.sh
```

**脚本将**：
- 创建目录结构（raw_data/, data/, checkpoints/, logs/）
- 创建名为 `rppg_training` 的 conda 环境
- 安装 PyTorch 2.1.0 + CUDA 12.1
- 安装所有依赖（包括 gdown 用于下载数据集）
- 验证安装

### Step 3: 下载数据集

```bash
conda activate rppg_training
bash download_ubfc.sh
```

**说明**：
- 使用 `gdown` 从 Google Drive 自动下载 UBFC-rPPG 数据集
- 预计时间：30-60 分钟（取决于网速）
- 输出：`raw_data/UBFC-rPPG/subject01-43/`
- 数据集大小：约 5 GB

**如果 gdown 下载失败**：
1. 访问：https://sites.google.com/view/ybenezeth/ubfcrppg
2. 手动下载数据集到本地
3. 使用 scp 上传到服务器：
   ```bash
   scp -r UBFC-rPPG username@server:/path/to/rppg_training/raw_data/
   ```

### Step 4: 数据预处理

```bash
python preprocess_data.py --dataset ubfc --raw_data raw_data --output data
```

**说明**：
- 使用 Haar Cascade 检测脸部
- 提取 3 个 ROI 区域（前额、左脸颊、右脸颊）
- 每个 ROI 调整为 36×36 像素
- 创建时间窗口样本（8 帧/窗口）
- 预计时间：2-3 小时（CPU 密集）
- 输出：`data/ubfc_processed.pt`（约 1.2 GB）

### Step 5: 验证数据（可选但推荐）

```bash
# 验证原始数据
python validate_data.py --mode raw

# 验证预处理数据
python validate_data.py --mode preprocessed

# 验证两者
python validate_data.py --mode both
```

### Step 6: 训练模型

```bash
bash run_training.sh
```

**说明**：
- 使用 A6000 GPU 训练
- Batch size: 128
- Epochs: 50 (with early stopping)
- 预计时间：1.5-2 小时
- 输出：
  - `checkpoints/best_model.pth` - 最佳模型
  - `checkpoints/train_history.json` - 训练历史
  - `logs/train_YYYYMMDD_HHMMSS.log` - 训练日志

---

## 📊 模型信息

### UltraLightRPPG (Multi-ROI 版本)

- **架构**: Shared 2D CNN (空间) + 1D Conv (时序) + ROI Fusion
- **参数量**: ~20K（比单 ROI 版本减少 60%）
- **输入**: (B, 8, 3, 36, 36, 3)
  - B: Batch size
  - 8: 时间窗口（8 帧）
  - 3: ROI 数量（前额、左脸颊、右脸颊）
  - 36×36: 图像尺寸
  - 3: RGB 通道
- **输出**: (B, 1) - BVP 值
- **记忆体需求**: ~80 KB（模型权重）
- **适合**: STM32N6 部署

### 网络结构

```
Input (B, 8, 3, 36, 36, 3)
    ↓
Reshape → (B*T*ROI, C, H, W) = (B*24, 3, 36, 36)
    ↓
Shared Spatial CNN (所有 ROI 共享权重)
  - Conv2D(3→16) + BN + ReLU + MaxPool   (36×36 → 18×18)
  - Conv2D(16→32) + BN + ReLU + MaxPool  (18×18 → 9×9)
  - Conv2D(32→16) + BN + ReLU
  - AdaptiveAvgPool2d(1)                  (9×9 → 1×1)
    ↓ (B*24, 16)
Reshape → (B, T, ROI, 16) = (B, 8, 3, 16)
    ↓
ROI Fusion (Concatenation) → (B, 8, 48)
    ↓
Transpose → (B, 48, 8)
    ↓
Temporal Conv1D
  - Conv1D(48→32, k=3) + ReLU
  - Conv1D(32→16, k=3) + ReLU
    ↓ (B, 16, 8)
Flatten → (B, 128)
    ↓
Fully Connected
  - Linear(128→32) + ReLU
  - Linear(32→1)
    ↓
Output (B, 1) - BVP value
```

---

## 🔧 自定义配置

### 修改训练参数

编辑 `config.yaml`:

```yaml
# 数据路径
data_paths:
  - 'data/ubfc_processed.pt'

# 训练参数
batch_size: 128         # 可调整 (64, 128, 256)
num_epochs: 50          # 可调整
learning_rate: 0.001    # 可调整
train_split: 0.8        # 训练/验证比例

# Early stopping
early_stopping_patience: 5

# 硬件
num_workers: 4
```

### 修改模型架构

编辑 `model.py` 中的 `UltraLightRPPG` 类。

---

## 📈 监控训练

### 方法 1: 查看日志文件

```bash
tail -f logs/train_YYYYMMDD_HHMMSS.log
```

### 方法 2: 使用 screen/tmux（推荐用于长时间训练）

```bash
# 创建新 session
screen -S rppg_training

# 运行训练
bash run_all.sh

# 分离 session: Ctrl+A, D

# 重新连接
screen -r rppg_training
```

---

## ✅ 验证结果

训练完成后，检查：

1. **最佳模型**: `checkpoints/best_model.pth`
2. **训练历史**: `checkpoints/train_history.json`
3. **日志文件**: `logs/train_YYYYMMDD_HHMMSS.log`

### 预期性能

基于 UBFC 数据集：
- **MAE**: 3-5 BPM（目标）
- **RMSE**: 4-6 BPM
- **MAPE**: 5-10%

### 下载模型到本地

```bash
# 在本地执行
scp username@server:/path/to/rppg_training/checkpoints/best_model.pth .
```

---

## 🐛 故障排除

### 问题 1: gdown 下载失败

```
Error: Cannot download from Google Drive
```

**解决**：
1. 检查网络连接
2. 使用备用下载方法（见 Step 3）
3. 或使用 `rclone`:
   ```bash
   rclone copy "drive:UBFC-rPPG" raw_data/UBFC-rPPG/ --progress
   ```

### 问题 2: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决**：降低 batch size
```yaml
# config.yaml
batch_size: 64  # 或更小
```

### 问题 3: Haar Cascade 文件缺失

```
Error: haarcascade_frontalface_default.xml not found
```

**解决**：
```bash
# 验证文件
python -c "import cv2; print(cv2.data.haarcascades)"

# 如果不存在，手动下载
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### 问题 4: 数据集未找到

```
Error: UBFC directory not found
```

**解决**：确认目录结构
```bash
ls raw_data/UBFC-rPPG/
# 应该看到 subject1, subject2, ... subject43
```

### 问题 5: 环境创建失败

```
Error: Could not create conda environment
```

**解决**：手动安装
```bash
conda create -n rppg_training python=3.12.3
conda activate rppg_training
pip install -r requirements.txt
```

---

## 📝 注意事项

### 资源需求
- **磁盘空间**: 约 10 GB
  - 原始数据：~5 GB
  - 预处理数据：~1.2 GB
  - 模型和日志：~100 MB
  - 剩余缓冲：~4 GB
- **内存**: 至少 16 GB RAM
- **GPU**: 至少 8 GB VRAM（推荐 24 GB）

### 时间成本
| 阶段 | 时间 | 硬件 |
|------|------|------|
| 下载数据 | 30-60 分钟 | 网络 |
| 预处理 | 2-3 小时 | CPU |
| 训练 | 1.5-2 小时 | GPU |
| **总计** | **4-6 小时** | |

### Multi-ROI 特性
- 使用 3 个 ROI 区域提升准确度
- 每个 ROI 独立处理后融合
- 参数量减少但准确度提升
- 更适合 STM32N6 部署

---

## 📞 下一步

训练完成后：
1. ✅ 下载 `checkpoints/best_model.pth` 回本地
2. 转换为 ONNX 格式
   ```python
   import torch
   model.eval()
   dummy_input = torch.randn(1, 8, 3, 36, 36, 3)
   torch.onnx.export(model, dummy_input, "rppg_model.onnx")
   ```
3. 使用 X-CUBE-AI 转换为 STM32 格式
4. 部署到 STM32N6

---

## 📚 参考资料

- **UBFC-rPPG**: https://sites.google.com/view/ybenezeth/ubfcrppg
- **PyTorch**: https://pytorch.org/
- **X-CUBE-AI**: https://www.st.com/en/embedded-software/x-cube-ai.html
- **项目文档**: 参见 `CLAUDE.md`

---

**版本**: 2.0 - 服务器端全流程
**日期**: 2025-11-18
**更新**: 从"本地预处理+服务器训练"迁移到"纯服务器端"架构
