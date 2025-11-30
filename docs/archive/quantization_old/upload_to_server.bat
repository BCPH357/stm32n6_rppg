@echo off
REM ====================================================================
REM 上傳 quantization 目錄到服務器
REM ====================================================================
REM
REM 目標服務器: miat@140.115.53.67
REM 目標路徑: /mnt/data_8T/ChenPinHao/server_training/quantization/
REM
REM ====================================================================

echo.
echo ====================================================================
echo Uploading quantization scripts to server
echo ====================================================================
echo.

set SERVER=miat@140.115.53.67
set REMOTE_PATH=/mnt/data_8T/ChenPinHao/server_training/

echo [1/2] Creating quantization directory on server...
ssh %SERVER% "mkdir -p %REMOTE_PATH%quantization/models"
if errorlevel 1 (
    echo ERROR: Failed to create directory on server
    pause
    exit /b 1
)
echo.

echo [2/2] Uploading files...
echo.

REM 上傳 Python 腳本
echo Uploading Python scripts...
scp quantize_utils.py %SERVER%:%REMOTE_PATH%quantization/
scp export_onnx.py %SERVER%:%REMOTE_PATH%quantization/
scp quantize_onnx.py %SERVER%:%REMOTE_PATH%quantization/
scp verify_quantization.py %SERVER%:%REMOTE_PATH%quantization/
echo.

REM 上傳 shell 腳本
echo Uploading shell script...
scp run_quantization.sh %SERVER%:%REMOTE_PATH%quantization/
echo.

REM 上傳文檔和配置
echo Uploading documentation...
scp README.md %SERVER%:%REMOTE_PATH%quantization/
scp README_SERVER.md %SERVER%:%REMOTE_PATH%quantization/
scp requirements_server.txt %SERVER%:%REMOTE_PATH%quantization/
echo.

REM 設置執行權限
echo Setting execute permission for shell script...
ssh %SERVER% "chmod +x %REMOTE_PATH%quantization/run_quantization.sh"
echo.

echo ====================================================================
echo Upload completed!
echo ====================================================================
echo.
echo Next steps on server:
echo 1. ssh %SERVER%
echo 2. cd %REMOTE_PATH%quantization
echo 3. conda activate rppg_training
echo 4. pip install -r requirements_server.txt
echo 5. bash run_quantization.sh
echo.
echo ====================================================================
pause
