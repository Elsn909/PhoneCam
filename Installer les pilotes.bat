@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0Programme\Install-Drivers.ps1" -Install
set "DRIVER_RESULT=%ERRORLEVEL%"
if "%DRIVER_RESULT%"=="3010" echo Redemarre Windows pour finaliser le pilote, puis relance PhoneCam.
if not "%DRIVER_RESULT%"=="0" if not "%DRIVER_RESULT%"=="3010" echo Installation non terminee. Consulte le message ci-dessus.
pause
exit /b %DRIVER_RESULT%
