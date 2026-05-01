@echo off
setlocal

pushd "%~dp0"
set "ROOT_DIR=%CD%"
popd
set "BUILD_DIR=%ROOT_DIR%\build"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 exit /b 1

set "APP_EXIT=0"
pushd "%BUILD_DIR%"
".\Release\rayTracer.exe"
set "APP_EXIT=%ERRORLEVEL%"
popd
endlocal & exit /b %APP_EXIT%