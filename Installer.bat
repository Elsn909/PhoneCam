@echo off
rem Installe PhoneCam sur ce PC : environnement Python isole dans "Programmeenv", modeles dans "Programme\data".
rem A lancer une seule fois (le relancer reprend ce qui manque). Il faut Internet (environ 8 Go) et une carte NVIDIA.
setlocal
set "TOP=%~dp0"
set "TOP=%TOP:~0,-1%"
set "ROOT=%TOP%\Programme"
cd /d "%ROOT%"
set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "TEMP=%ROOT%\data\temp"
set "TMP=%ROOT%\data\temp"
if not exist "%TEMP%" mkdir "%TEMP%"

echo.
echo === 1/6  Python 3.12
call :find_python
if not defined PY (
  echo Python 3.12 absent : installation avec winget...
  winget install -e --id Python.Python.3.12 --source winget --scope user --accept-package-agreements --accept-source-agreements
  call :find_python
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
echo === 4/6  Modeles (environ 4 Go, depuis leurs sources officielles) et voix fournies
rem Comme les lanceurs : "~" = data\home, pour que les modeles d'insightface restent dans le dossier.
if not exist "%ROOT%\data\home" mkdir "%ROOT%\data\home"
setlocal
set "USERPROFILE=%ROOT%\data\home"
set "HOME=%ROOT%\data\home"
"%VPY%" "%ROOT%\PhoneCam\telecharger_modeles.py"
if errorlevel 1 (endlocal & goto :fail)
endlocal

echo.
echo === 5/6  Pilotes et logiciels manquants (installation automatique)
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%ROOT%\Install-Drivers.ps1" -Install
set "DRIVER_RESULT=%ERRORLEVEL%"
if "%DRIVER_RESULT%"=="3010" echo [!] Un redemarrage de Windows est necessaire pour finaliser les pilotes.
if not "%DRIVER_RESULT%"=="0" if not "%DRIVER_RESULT%"=="3010" goto :fail

echo.
echo === 6/6  Verifications
"%VPY%" -c "import torch, onnxruntime, insightface, PySide6, cv2; print('Carte graphique :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUNE carte NVIDIA vue par PyTorch'); print('onnxruntime :', ', '.join(onnxruntime.get_available_providers()))"
"%VPY%" -c "import cv2, sys; from cv2_enumerate_cameras import enumerate_cameras as e; sys.exit(0 if any(c.name == 'OBS Virtual Camera' for c in e(cv2.CAP_DSHOW)) else 1)" >nul 2>&1 || echo [!] OBS Virtual Camera pas encore visible : redemarrer le PC apres l'installation d'OBS.
"%VPY%" -c "import sounddevice as s, sys; sys.exit(0 if any(d['name'].startswith(('CABLE Input', 'CABLE-A Input', 'CABLE-B Input')) for d in s.query_devices()) else 1)" >nul 2>&1 || echo [!] Micro virtuel non detecte : redemarre Windows ou lance "Installer les pilotes.bat".
if not exist "%TOP%\Captures" mkdir "%TOP%\Captures"
rmdir /s /q "%TEMP%" 2>nul
echo.
echo Installation terminee. Lance "Lancer PhoneCam.bat".
pause
exit /b 0

:fail
echo.
echo Echec de l'installation ^(message ci-dessus^). Relancer Installer.bat reprend ou il s'est arrete.
pause
exit /b 1

:find_python
rem Le PATH de ce terminal reste ancien apres l'installation par winget.
set "PY="
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try_python "%ProgramFiles%\Python312\python.exe"
call :try_python "%ProgramFiles(x86)%\Python312\python.exe"
if defined PY exit /b 0
py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PY=py -3.12"
if defined PY exit /b 0
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && set "PY=python"
exit /b 0

:try_python
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 set "PY="%~1""
exit /b 0
