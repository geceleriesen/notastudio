@echo off
chcp 65001 >nul
echo ============================================================
echo   NotaStudyo Maestro Pro - Studio AI (ACE-Step 1.5) Kurulumu
echo ============================================================
echo.

set "PY_EXE=python"
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
)
echo Kullanilan Python: %PY_EXE%
"%PY_EXE%" --version
echo.

echo [1/3] Temel ses ve sunucu kutuphaneleri guncelleniyor...
"%PY_EXE%" -m pip install --upgrade pip
"%PY_EXE%" -m pip install -r requirements.txt

echo.
echo [2/3] PyTorch kuruluyor (yoksa indirilir)...
"%PY_EXE%" -c "import torch; print('PyTorch zaten kurulu:', torch.__version__)" 2>nul
if errorlevel 1 (
    "%PY_EXE%" -m pip install torch torchaudio
)

echo.
echo [3/3] Hugging Face Diffusers (ACE-Step 1.5 Motoru) kuruluyor...
"%PY_EXE%" -m pip install --upgrade diffusers transformers accelerate safetensors huggingface-hub

echo.
echo [Dogrulama] Studio AI motoru test ediliyor...
"%PY_EXE%" -c "import torch, diffusers; from diffusers import AceStepPipeline; print('BASARILI! ACE-Step 1.5 Studio AI hazir. Diffusers:', diffusers.__version__)"

echo.
echo ============================================================
echo   Kurulum tamamlandi!
echo   Simdi "%PY_EXE% backend\main.py" ile uygulamayi baslatin.
echo   Ilk sarki uretiminde ACE-Step 1.5 agirliklari (~4.8 GB)
echo   bir kez "models_cache\ace_step_15" klasorune indirilecektir.
echo ============================================================
pause
