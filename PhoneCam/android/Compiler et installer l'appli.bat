@echo off
rem Recompile l'appli Android PhoneCam et l'installe sur le telephone branche (debogage USB).
rem Outils attendus dans "tools\android-dev" (JDK 17, SDK Android 35, Gradle) : rien n'est ecrit dans le profil Windows.
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
set "D=%ROOT%\tools\android-dev"
if not exist "%D%\jdk\bin\java.exe" (
  echo Outils Android absents : %D%
  pause
  exit /b 1
)
set "JAVA_HOME=%D%\jdk"
set "ANDROID_HOME=%D%\sdk"
set "ANDROID_SDK_ROOT=%D%\sdk"
set "ANDROID_USER_HOME=%D%\android-home"
set "GRADLE_USER_HOME=%D%\gradle-home"
set "JAVA_TOOL_OPTIONS=-Duser.home="%D%\java-home""
set "PATH=%D%\jdk\bin;%D%\gradle\bin;%PATH%"
cd /d "%HERE%"
call gradle --no-daemon --console=plain assembleDebug
if errorlevel 1 (
  echo Echec de la compilation.
  pause
  exit /b 1
)
"%ROOT%\tools\platform-tools\adb.exe" install -r "app\build\outputs\apk\debug\app-debug.apk"
pause
