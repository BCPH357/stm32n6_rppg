@echo off
echo ========================================================================
echo Fix Dynamic Batch and Upload to Server
echo ========================================================================
echo.

REM 步驟 1: 上傳修正腳本
echo [Step 1] Uploading scripts to server...
scp "D:\MIAT\rppg\server_training\convert_to_4d_for_stm32_v2.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
if errorlevel 1 (
    echo [ERROR] Failed to upload convert_to_4d_for_stm32_v2.py
    pause
    exit /b 1
)

scp "D:\MIAT\rppg\server_training\fix_onnx_dynamic_batch.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/
if errorlevel 1 (
    echo [ERROR] Failed to upload fix_onnx_dynamic_batch.py
    pause
    exit /b 1
)

echo [OK] Scripts uploaded successfully
echo.

REM 步驟 2: 顯示服務器端執行指令
echo ========================================================================
echo [Step 2] Next: Run on Server
echo ========================================================================
echo.
echo SSH to server:
echo   ssh miat@140.115.53.67
echo.
echo Then run ONE of the following:
echo.
echo Option 1 - Reconvert model (Recommended):
echo   cd /mnt/data_8T/ChenPinHao/server_training/
echo   conda activate rppg_training
echo   python convert_to_4d_for_stm32_v2.py
echo.
echo Option 2 - Fix existing ONNX:
echo   cd /mnt/data_8T/ChenPinHao/server_training/
echo   conda activate rppg_training
echo   python fix_onnx_dynamic_batch.py
echo.
echo ========================================================================
echo [Step 3] After conversion/fixing, download:
echo ========================================================================
echo.
echo Download fixed FP32 ONNX:
echo   scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_fp32.onnx D:\MIAT\rppg\quantization\models\
echo.
echo Or download INT8 quantized model (if already quantized):
echo   scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\MIAT\rppg\quantization\models\
echo.
echo ========================================================================

pause
