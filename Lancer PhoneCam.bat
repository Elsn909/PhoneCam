@echo off
rem PhoneCam : le logiciel. Tous les caches restent dans le dossier "Programme" : rien n'est ecrit dans le profil Windows.
setlocal
set "TOP=%~dp0"
set "TOP=%TOP:~0,-1%"
set "ROOT=%TOP%\Programme"
if not exist "%ROOT%\venv\Scripts\python.exe" (
  echo PhoneCam n'est pas encore installe : lance d'abord "Installer.bat".
  pause
  exit /b 1
)
set "DLC_HOME=%ROOT%\data\home"
if not exist "%DLC_HOME%" mkdir "%DLC_HOME%"
if not exist "%ROOT%\data\temp" mkdir "%ROOT%\data\temp"
if not exist "%TOP%\Captures" mkdir "%TOP%\Captures"

set "USERPROFILE=%DLC_HOME%"
set "HOME=%DLC_HOME%"
set "KERAS_HOME=%DLC_HOME%\.keras"
set "TORCH_HOME=%DLC_HOME%\.cache\torch"
set "XDG_CACHE_HOME=%DLC_HOME%\.cache"
set "HF_HOME=%DLC_HOME%\.cache\huggingface"
set "ANDROID_USER_HOME=%DLC_HOME%\.android"
set "TEMP=%ROOT%\data\temp"
set "TMP=%ROOT%\data\temp"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "KERAS_BACKEND=torch"

cd /d "%ROOT%\PhoneCam"
"%ROOT%\venv\Scripts\python.exe" phonecam.py %*
endlocal
