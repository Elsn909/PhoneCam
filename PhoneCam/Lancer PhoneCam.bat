@echo off
rem PhoneCam : camera du telephone -> Deep-Live-Cam -> effets -> camera virtuelle.
rem Comme le lanceur principal, tous les caches restent dans le dossier Deep Live Cam.
setlocal
set "PC=%~dp0"
set "PC=%PC:~0,-1%"
for %%I in ("%PC%\..") do set "ROOT=%%~fI"
set "DLC_HOME=%ROOT%\data\home"
if not exist "%DLC_HOME%" mkdir "%DLC_HOME%"
if not exist "%ROOT%\data\temp" mkdir "%ROOT%\data\temp"

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

cd /d "%PC%"
"%ROOT%\venv\Scripts\python.exe" phonecam.py %*
endlocal
