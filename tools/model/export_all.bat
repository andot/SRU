@echo off
REM export_all.bat - wrapper to run export.py (now located in the same folder)
setlocal enabledelayedexpansion

REM Script directory (with trailing backslash)
set "SCRIPT_DIR=%~dp0"

REM Determine backend from first positional argument if it matches known backends
set "BACKEND=all"
if not "%~1"=="" set "FIRST=%~1"
if /I "%FIRST%"=="all" (
  set "BACKEND=%FIRST%"
  shift
)
if /I "%FIRST%"=="onnx" (
  set "BACKEND=%FIRST%"
  shift
)
if /I "%FIRST%"=="coreml" (
  set "BACKEND=%FIRST%"
  shift
)
if /I "%FIRST%"=="ncnn" (
  set "BACKEND=%FIRST%"
  shift
)

REM Build ARGS from remaining positional parameters (safe against parsing quirks)
set "ARGS="
:build_args
if "%~1"=="" goto :args_done
if defined ARGS (
  set "ARGS=%ARGS% %~1"
) else (
  set "ARGS=%~1"
)
shift
goto :build_args
:args_done

REM If user provided -m or --model in ARGS, forward directly to the Python exporter
echo %ARGS% | findstr /I /C:"-m" /C:"--model" >nul
if %errorlevel%==0 (
  echo Running: python "%SCRIPT_DIR%export.py" %BACKEND% %ARGS%
  python "%SCRIPT_DIR%export.py" %BACKEND% %ARGS%
  goto :end
)

REM No explicit model: iterate .pth files under models directory (two levels up from tools\model)
if not exist "%SCRIPT_DIR%..\..\models" (
  echo [Warn] models directory not found: "%SCRIPT_DIR%..\..\models"
  goto :end
)

for /R "%SCRIPT_DIR%..\..\models" %%f in (*.pth) do (
  set "modelname=%%~nf"
  echo [Info] Exporting !modelname! to backend: %BACKEND%
  echo Running: python "%SCRIPT_DIR%export.py" %BACKEND% -m "!modelname!" %ARGS%
  python "%SCRIPT_DIR%export.py" %BACKEND% -m "!modelname!" %ARGS%
)

:end
endlocal
exit /B 0
