@echo off
REM ====================================================================
REM rPPG Multi-ROI Model INT8 Quantization - Complete Workflow
REM ====================================================================
REM
REM 此腳本將執行完整的量化流程：
REM 1. 準備校準數據集
REM 2. 導出 FP32 ONNX 模型
REM 3. 執行 INT8 量化
REM 4. 驗證量化精度
REM
REM 使用環境: zerodce_tf conda environment
REM
REM ====================================================================

echo.
echo ====================================================================
echo rPPG Multi-ROI Model INT8 Quantization Workflow
echo ====================================================================
echo.

REM 檢查是否在正確目錄
if not exist "quantize_utils.py" (
    echo ERROR: Please run this script from the quantization directory!
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM 激活 conda 環境
echo [Step 0/4] Activating conda environment: zerodce_tf
call conda activate zerodce_tf
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'zerodce_tf'
    echo Please ensure the environment exists.
    pause
    exit /b 1
)
echo.

REM Step 1: 準備校準數據
echo ====================================================================
echo [Step 1/4] Preparing Calibration Dataset
echo ====================================================================
python quantize_utils.py
if errorlevel 1 (
    echo ERROR: Calibration data preparation failed!
    pause
    exit /b 1
)
echo.
echo Press any key to continue to Step 2...
pause >nul
echo.

REM Step 2: 導出 FP32 ONNX
echo ====================================================================
echo [Step 2/4] Exporting FP32 ONNX Model
echo ====================================================================
python export_onnx.py
if errorlevel 1 (
    echo ERROR: FP32 ONNX export failed!
    pause
    exit /b 1
)
echo.
echo Press any key to continue to Step 3...
pause >nul
echo.

REM Step 3: INT8 量化
echo ====================================================================
echo [Step 3/4] Quantizing to INT8
echo ====================================================================
python quantize_onnx.py
if errorlevel 1 (
    echo ERROR: INT8 quantization failed!
    pause
    exit /b 1
)
echo.
echo Press any key to continue to Step 4...
pause >nul
echo.

REM Step 4: 驗證精度
echo ====================================================================
echo [Step 4/4] Verifying Quantization Accuracy
echo ====================================================================
python verify_quantization.py
set VERIFY_RESULT=%errorlevel%
echo.

REM 最終報告
echo ====================================================================
echo Quantization Workflow Completed!
echo ====================================================================
echo.

if %VERIFY_RESULT%==0 (
    echo Status: SUCCESS - Quantization acceptable
    echo.
    echo Next steps:
    echo 1. INT8 model is ready: models/rppg_int8_qdq.onnx
    echo 2. Use X-CUBE-AI to convert for STM32N6
    echo 3. Refer to: ../stm32n6_deployment/deployment_guide.md
) else if %VERIFY_RESULT%==2 (
    echo Status: WARNING - Quantization degradation significant
    echo.
    echo Suggestions:
    echo 1. Increase calibration samples: python quantize_utils.py --num_samples 500
    echo 2. Consider Quantization-Aware Training (QAT)
    echo 3. Check calibration data distribution
) else (
    echo Status: ERROR - Quantization failed
    echo Please check error messages above
)

echo.
echo ====================================================================
pause
