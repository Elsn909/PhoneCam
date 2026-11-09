@echo off
rem Lance Deep-Live-Cam (GPU NVIDIA / CUDA) depuis son venv.
rem Tous les caches et modeles restent dans ce dossier : rien n'est ecrit dans le profil Windows.
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "DLC_HOME=%ROOT%\data\home"
if not exist "%DLC_HOME%" mkdir "%DLC_HOME%"
if not exist "%ROOT%\data\temp" mkdir "%ROOT%\data\temp"

rem ~ (insightface, keras, opennsfw2...) redirige vers data\home
set "USERPROFILE=%DLC_HOME%"
set "HOME=%DLC_HOME%"
set "KERAS_HOME=%DLC_HOME%\.keras"
set "TORCH_HOME=%DLC_HOME%\.cache\torch"
set "XDG_CACHE_HOME=%DLC_HOME%\.cache"
set "HF_HOME=%DLC_HOME%\.cache\huggingface"
set "TEMP=%ROOT%\data\temp"
set "TMP=%ROOT%\data\temp"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "KERAS_BACKEND=torch"

cd /d "%ROOT%\app"
"%ROOT%\venv\Scripts\python.exe" run.py --execution-provider cuda %*
endlocal
