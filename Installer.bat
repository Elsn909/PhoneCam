@echo off
rem Installe PhoneCam sur ce PC : environnement Python isole dans le dossier "venv", modeles dans "data".
rem A lancer une seule fois (le relancer reprend ce qui manque). Il faut Internet (environ 8 Go) et une carte NVIDIA.
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "TEMP=%ROOT%\data\temp"
set "TMP=%ROOT%\data\temp"
if not exist "%TEMP%" mkdir "%TEMP%"

echo.
echo === 1/6  Python 3.12
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
  python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.12 absent : installation avec winget...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
)
if not defined PY (
  echo.
  echo Python 3.12 introuvable. Installe-le depuis https://www.python.org/downloads/release/python-31210/
  echo ^(cocher "Add python.exe to PATH"^), puis relance Installer.bat.
  pause
  exit /b 1
)

echo.
echo === 2/6  Environnement isole (dossier venv)
if not exist "%ROOT%\venv\Scripts\python.exe" (
  %PY% -m venv "%ROOT%\venv" || goto :fail
)
set "VPY=%ROOT%\venv\Scripts\python.exe"
"%VPY%" -m pip install --no-cache-dir --upgrade pip || goto :fail

echo.
echo === 3/6  Paquets (PyTorch CUDA, onnxruntime, TensorRT, PySide6...) : long
rem insightface est fourni deja compile : sinon il faudrait Visual Studio Build Tools.
"%VPY%" -m pip install --no-cache-dir --no-deps "%ROOT%\wheels\insightface-0.7.3-cp312-cp312-win_amd64.whl" || goto :fail
"%VPY%" -m pip install --no-cache-dir -r "%ROOT%\requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128 || goto :fail

echo.
echo === 4/6  Modeles (environ 3,5 Go, depuis leurs sources officielles)
rem Comme les lanceurs : "~" = data\home, pour que les modeles d'insightface restent dans le dossier.
if not exist "%ROOT%\data\home" mkdir "%ROOT%\data\home"
setlocal
set "USERPROFILE=%ROOT%\data\home"
set "HOME=%ROOT%\data\home"
"%VPY%" "%ROOT%\PhoneCam\telecharger_modeles.py"
if errorlevel 1 (endlocal & goto :fail)
endlocal

echo.
echo === 5/6  Logiciels du PC : OBS (camera virtuelle), ffmpeg (Studio), adb (telephone par cable)
"%VPY%" -c "import cv2, sys; from cv2_enumerate_cameras import enumerate_cameras as e; sys.exit(0 if any(c.name == 'OBS Virtual Camera' for c in e(cv2.CAP_DSHOW)) else 1)" >nul 2>&1
if errorlevel 1 (
  echo OBS Studio absent : installation avec winget ^(Windows peut demander l'autorisation^)...
  winget install -e --id OBSProject.OBSStudio --accept-package-agreements --accept-source-agreements
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo ffmpeg absent : installation avec winget...
  winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
)
if not exist "%ROOT%\tools\platform-tools\adb.exe" (
  echo Telechargement d'adb ^(outils officiels Android de Google^)...
  if not exist "%ROOT%\tools" mkdir "%ROOT%\tools"
  powershell -NoProfile -Command "Invoke-WebRequest https://dl.google.com/android/repository/platform-tools-latest-windows.zip -OutFile '%TEMP%\pt.zip'; Expand-Archive -Force '%TEMP%\pt.zip' '%ROOT%\tools'"
)

echo.
echo === 6/6  Verifications
"%VPY%" -c "import torch, onnxruntime, insightface, PySide6, cv2; print('Carte graphique :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUNE carte NVIDIA vue par PyTorch'); print('onnxruntime :', ', '.join(onnxruntime.get_available_providers()))"
"%VPY%" -c "import cv2, sys; from cv2_enumerate_cameras import enumerate_cameras as e; sys.exit(0 if any(c.name == 'OBS Virtual Camera' for c in e(cv2.CAP_DSHOW)) else 1)" >nul 2>&1 || echo [!] OBS Virtual Camera pas encore visible : redemarrer le PC apres l'installation d'OBS.
"%VPY%" -c "import sounddevice as s, sys; sys.exit(0 if any('CABLE-A' in d['name'] for d in s.query_devices()) else 1)" >nul 2>&1 || echo [i] VB-Cable A+B absent : seulement utile pour le micro simule ^(voir LISEZMOI.txt^).
rmdir /s /q "%TEMP%" 2>nul
echo.
echo Installation terminee. Lance "PhoneCam\Lancer PhoneCam.bat".
pause
exit /b 0

:fail
echo.
echo Echec de l'installation ^(message ci-dessus^). Relancer Installer.bat reprend ou il s'est arrete.
pause
exit /b 1
