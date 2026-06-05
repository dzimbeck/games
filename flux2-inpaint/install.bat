@echo off
setlocal enabledelayedexpansion

echo ===========================================================
echo   Local AI Image Editing / Inpainting - FLUX.2 Installer
echo ===========================================================
echo.

:: Determine install directory relative to where this script lives
set "SCRIPT_DIR=%~dp0"
set "AI_DIR=%SCRIPT_DIR%ai-model"

echo All files will be installed to: %AI_DIR%
echo.

:: ----------------------------------------------------------------
:: Ask the user how much GPU VRAM they have. We use this to pick the
:: most appropriate FLUX.2 model + runtime mode from Hugging Face.
:: Reference: https://github.com/black-forest-labs/flux2 (Model Overview)
:: We default to 12 GB (e.g. an RTX 3060 12GB / RTX 4070).
:: ----------------------------------------------------------------
echo How much GPU VRAM does your graphics card have (in GB)?
echo   Examples:  6, 8, 12, 16, 24, 48, 80
echo   - 6 or 7 GB ........ FLUX.2 [klein] 4B  (4-bit quantized)
echo   - 8 to 11 GB ....... FLUX.2 [klein] 4B  (CPU offload)
echo   - 12 to 15 GB ...... FLUX.2 [klein] 4B  (full GPU)   ^<-- default
echo   - 16 to 23 GB ...... FLUX.2 [klein] 9B  (CPU offload)
echo   - 24 to 47 GB ...... FLUX.2 [klein] 9B  (full GPU)
echo   - 48 GB or more .... FLUX.2 [dev] 32B   (CPU offload)
echo.
set "VRAM=12"
set /p VRAM="Enter your VRAM in GB [default 12]: "

:: Validate that VRAM is a positive integer
set "VRAM=%VRAM: =%"
for /f "delims=0123456789" %%A in ("%VRAM%") do (
    echo Invalid VRAM value "%VRAM%". Please enter a whole number, e.g. 12.
    pause
    exit /b 1
)
if "%VRAM%"=="" set "VRAM=12"

:: Choose model + runtime mode based on VRAM.
::   MODEL_NAME = Hugging Face repo id
::   PIPELINE   = klein | dev   (which diffusers pipeline to use)
::   RUN_MODE   = cuda | offload | quant
::   STEPS      = default denoising steps for this model
if %VRAM% GEQ 48 (
    set "MODEL_NAME=black-forest-labs/FLUX.2-dev"
    set "PIPELINE=dev"
    set "RUN_MODE=offload"
    set "STEPS=50"
    set "MODEL_LABEL=FLUX.2 [dev] 32B (CPU offload)"
) else if %VRAM% GEQ 24 (
    set "MODEL_NAME=black-forest-labs/FLUX.2-klein-9B"
    set "PIPELINE=klein"
    set "RUN_MODE=cuda"
    set "STEPS=4"
    set "MODEL_LABEL=FLUX.2 [klein] 9B (full GPU)"
) else if %VRAM% GEQ 16 (
    set "MODEL_NAME=black-forest-labs/FLUX.2-klein-9B"
    set "PIPELINE=klein"
    set "RUN_MODE=offload"
    set "STEPS=4"
    set "MODEL_LABEL=FLUX.2 [klein] 9B (CPU offload)"
) else if %VRAM% GEQ 12 (
    set "MODEL_NAME=black-forest-labs/FLUX.2-klein-4B"
    set "PIPELINE=klein"
    set "RUN_MODE=cuda"
    set "STEPS=4"
    set "MODEL_LABEL=FLUX.2 [klein] 4B (full GPU)"
) else if %VRAM% GEQ 8 (
    set "MODEL_NAME=black-forest-labs/FLUX.2-klein-4B"
    set "PIPELINE=klein"
    set "RUN_MODE=offload"
    set "STEPS=4"
    set "MODEL_LABEL=FLUX.2 [klein] 4B (CPU offload)"
) else (
    set "MODEL_NAME=black-forest-labs/FLUX.2-klein-4B"
    set "PIPELINE=klein"
    set "RUN_MODE=quant"
    set "STEPS=4"
    set "MODEL_LABEL=FLUX.2 [klein] 4B (4-bit quantized)"
)

echo.
echo Detected VRAM: %VRAM% GB
echo Selected model: %MODEL_LABEL%
echo Hugging Face repo: %MODEL_NAME%
echo.
echo NOTE: FLUX.2 [klein] 4B is Apache-2.0 and downloads without a login.
echo       FLUX.2 [klein] 9B and FLUX.2 [dev] are gated, non-commercial models.
echo       For those you must accept the license on Hugging Face and run:
echo           huggingface-cli login
echo.

:: Create ai-model directory
if not exist "%AI_DIR%" mkdir "%AI_DIR%"

:: ---- Install pyenv-win locally ----
set "PYENV_ROOT=%AI_DIR%\.pyenv"
set "PYENV=%PYENV_ROOT%\pyenv-win"
set "PATH=%PYENV%\bin;%PYENV%\shims;%PATH%"
set "PYENV_HOME=%PYENV%"

if not exist "%PYENV%\bin\pyenv.bat" (
    echo Installing pyenv-win locally...
    if not exist "%PYENV_ROOT%" mkdir "%PYENV_ROOT%"
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/pyenv-win/pyenv-win/archive/refs/heads/master.zip' -OutFile '%AI_DIR%\pyenv-win.zip'"
    if errorlevel 1 (
        echo ERROR: Failed to download pyenv-win.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%AI_DIR%\pyenv-win.zip' -DestinationPath '%PYENV_ROOT%\temp' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract pyenv-win.
        pause
        exit /b 1
    )
    :: Move contents from extracted folder
    xcopy /E /Y /Q "%PYENV_ROOT%\temp\pyenv-win-master\*" "%PYENV_ROOT%\" >nul
    rd /S /Q "%PYENV_ROOT%\temp" 2>nul
    del "%AI_DIR%\pyenv-win.zip" 2>nul
    echo pyenv-win installed to %PYENV_ROOT%
) else (
    echo pyenv-win already installed.
)
echo.

:: ---- Install Python 3.11 via pyenv ----
set "PYTHON_VERSION=3.11.9"
echo Installing Python %PYTHON_VERSION% via pyenv (localized)...

call "%PYENV%\bin\pyenv.bat" install %PYTHON_VERSION% --skip-existing
if errorlevel 1 (
    echo ERROR: Failed to install Python %PYTHON_VERSION%.
    pause
    exit /b 1
)

call "%PYENV%\bin\pyenv.bat" local %PYTHON_VERSION%
echo Python %PYTHON_VERSION% installed.
echo.

:: Get the path to the installed Python
for /f "tokens=*" %%i in ('call "%PYENV%\bin\pyenv.bat" which python') do set "PYENV_PYTHON=%%i"

:: ---- Create virtual environment ----
set "VENV_DIR=%AI_DIR%\venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    "%PYENV_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: ---- Activate venv and install dependencies ----
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch...
:: Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    echo WARNING: FLUX.2 is very slow without a CUDA GPU.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo Installing FLUX.2 dependencies...
:: FLUX.2 pipelines (Flux2Pipeline / Flux2KleinPipeline) currently live on
:: diffusers main, and the klein text encoder needs a recent transformers.
pip install "git+https://github.com/huggingface/diffusers.git"
pip install "transformers>=4.57.0" accelerate safetensors "huggingface_hub>=0.26.0" Pillow
if "%RUN_MODE%"=="quant" (
    echo Installing bitsandbytes for 4-bit quantization...
    pip install bitsandbytes
)

echo.
echo Downloading model: %MODEL_NAME%
echo This may take a while depending on your internet connection...

python "%SCRIPT_DIR%download_model.py" "%MODEL_NAME%" "%AI_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to download the model.
    echo If the model is gated, accept its license on Hugging Face and run:
    echo     pip install huggingface_hub ^&^& huggingface-cli login
    pause
    exit /b 1
)

:: ---- Generate a convenience launcher that remembers the chosen settings ----
set "RUN_BAT=%SCRIPT_DIR%run_inpaint.bat"
echo Writing launcher: %RUN_BAT%
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_DIR=%%~dp0"
    echo call "%%SCRIPT_DIR%%ai-model\venv\Scripts\activate.bat"
    echo python "%%SCRIPT_DIR%%inpaint.py" --model-dir "%%SCRIPT_DIR%%ai-model\model" --pipeline %PIPELINE% --mode %RUN_MODE% --steps %STEPS% %%*
) > "%RUN_BAT%"

echo.
echo ===========================================================
echo   Installation complete!
echo ===========================================================
echo.
echo Model installed to: %AI_DIR%\model
echo.
echo To edit / inpaint an image, run:
echo.
echo   run_inpaint.bat --image input.png --prompt "make the sky a starry night" --output result.png
echo.
echo To generate an image from text only:
echo.
echo   run_inpaint.bat --prompt "a cat holding a sign that says hello world" --output cat.png
echo.
echo To restrict edits to a region, also pass a black/white mask
echo (white = area to change):
echo.
echo   run_inpaint.bat --image input.png --mask mask.png --prompt "add a red hat" --output result.png
echo.
pause
