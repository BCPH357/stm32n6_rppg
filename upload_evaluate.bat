@echo off
echo ========================================================================
echo Upload Evaluation Script
echo ========================================================================
echo.

scp "D:\MIAT\rppg\server_training\evaluate_quantized_model.py" miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/

if errorlevel 1 (
    echo [ERROR] Upload failed!
    pause
    exit /b 1
)

echo [OK] Script uploaded successfully
echo.
echo ========================================================================
echo Next: Run on Server
echo ========================================================================
echo.
echo SSH command:
echo   ssh miat@140.115.53.67
echo.
echo Run evaluation:
echo   cd /mnt/data_8T/ChenPinHao/server_training/
echo   conda activate rppg_training
echo   pip install tqdm matplotlib  # if needed
echo   python evaluate_quantized_model.py
echo.
echo Download results:
echo   scp -r miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/evaluation_results/ D:\MIAT\rppg\
echo.
echo ========================================================================

pause
