# 修復 PyTorch DLL 載入錯誤

## 問題

```
OSError: [WinError 1114] 動態連結程式庫 (DLL) 初始化程序失敗
Error loading "C:\Users\user\anaconda3\Lib\site-packages\torch\lib\c10.dll"
```

## 原因

PyTorch CPU 版本需要 **Microsoft Visual C++ Redistributable**。

## 解決方案

### 方法 1: 安裝 Visual C++ Redistributable（推薦）

下載並安裝最新版本：

**下載連結**: https://aka.ms/vs/17/release/vc_redist.x64.exe

或訪問官方頁面：
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

**安裝步驟**：
1. 下載 `vc_redist.x64.exe`
2. 雙擊運行
3. 點擊「安裝」
4. 重啟命令提示符
5. 重新運行 `start.bat`

### 方法 2: 使用舊版 PyTorch（如果方法 1 不行）

```batch
pip uninstall torch torchvision -y
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

舊版本對 VC++ 依賴較少。

### 方法 3: 檢查是否已安裝但損壞

打開「控制台」→「程式和功能」，搜尋 **Microsoft Visual C++**。

如果已安裝但仍報錯：
1. 解除安裝所有 Microsoft Visual C++ 2015-2022 Redistributable
2. 重新安裝最新版本（方法 1）

## 驗證修復

安裝 VC++ Redistributable 後，運行：

```batch
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

應該顯示：
```
PyTorch version: 2.9.1+cpu
```

## 如果仍然失敗

聯繫我並提供錯誤訊息。

---

**建議**: 先安裝 VC++ Redistributable，這是 Windows 上運行 PyTorch 的標準要求。
