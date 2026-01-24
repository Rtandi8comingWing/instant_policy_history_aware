@echo off
REM =============================================================================
REM Instant Policy - Environment Installation Script (Windows)
REM =============================================================================

echo ============================================
echo Instant Policy Environment Setup
echo ============================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo [Step 1/4] Installing PyTorch with CUDA 11.8...
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [WARNING] PyTorch installation may have issues. Continuing...
)

echo.
echo [Step 2/4] Installing PyTorch Geometric dependencies...
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
if errorlevel 1 (
    echo [WARNING] PyG dependencies installation may have issues. Continuing...
)

pip install torch-geometric==2.5.0

echo.
echo [Step 3/4] Installing other dependencies...
pip install pytorch-lightning>=2.4.0 torchmetrics>=1.0.0
pip install open3d>=0.18.0
pip install numpy>=1.26.0 scipy>=1.14.0
pip install pyyaml>=6.0 tqdm>=4.66.0
pip install matplotlib>=3.9.0

echo.
echo [Step 4/4] Installing Instant Policy package...
pip install -e .

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Run 'python test_env.py' to verify installation.
echo.
pause
