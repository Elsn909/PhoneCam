@echo off
rem Retire tout ce qui a servi a fabriquer l'appli Android : outils portables, fichiers de compilation,
rem et la seule cle de registre qu'ils ont creee. Le code source de l'appli et ADB sont conserves.
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
echo Va supprimer :
echo   %ROOT%\tools\android-dev   (JDK, SDK Android, Gradle et leurs caches, ~1,6 Go)
echo   %HERE%app\build, %HERE%build, %HERE%.gradle   (fichiers de compilation)
echo   HKCU\Software\JavaSoft\Prefs\google   (identifiant de statistiques des outils Google)
echo L'appli deja installee sur le telephone continue de fonctionner.
choice /c ON /m "Continuer"
if errorlevel 2 exit /b
rmdir /s /q "%ROOT%\tools\android-dev" 2>nul
rmdir /s /q "%HERE%app\build" 2>nul
rmdir /s /q "%HERE%build" 2>nul
rmdir /s /q "%HERE%.gradle" 2>nul
powershell -NoProfile -Command "Remove-Item 'HKCU:\Software\JavaSoft\Prefs\google' -Recurse -ErrorAction SilentlyContinue; foreach ($k in 'HKCU:\Software\JavaSoft\Prefs', 'HKCU:\Software\JavaSoft') { $i = Get-Item $k -ErrorAction SilentlyContinue; if ($i -and $i.SubKeyCount -eq 0 -and $i.ValueCount -eq 0) { Remove-Item $k } }"
echo.
choice /c ON /m "Desinstaller aussi l'appli PhoneCam du telephone branche"
if errorlevel 2 goto fin
"%ROOT%\tools\platform-tools\adb.exe" uninstall com.phonecam.app
:fin
echo Termine.
pause
