@echo off
echo ============================================
echo rPPG Web Application - Installation
echo ============================================
echo.

echo Step 1: Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)
echo.

echo Step 2: Installing Python dependencies (except PyTorch)...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo.

echo Step 3: Installing PyTorch...
echo Note: Using PyTorch 2.2.0 (2.1.0 no longer available)
echo Trying CUDA version first (for NVIDIA GPU)...
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo CUDA version failed, trying CPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install PyTorch!
        echo Please try manually:
        echo pip install torch torchvision
        echo.
        pause
        exit /b 1
    )
)
echo.

echo Step 4: Checking model file...
if exist "models\best_model.pth" (
    echo ✓ Model file found: models\best_model.pth
) else (
    echo.
    echo WARNING: Model file not found!
    echo Please copy best_model.pth from server_training/checkpoints/ to models/
    echo.
    echo You can run:
    echo copy ..\server_training\checkpoints\best_model.pth models\best_model.pth
    echo.
    pause
)
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To start the application, run:
echo     start.bat
echo.
echo Then open your browser and visit:
echo     http://localhost:5000
echo.
pause
