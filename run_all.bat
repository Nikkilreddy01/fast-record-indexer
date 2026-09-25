@echo off
REM ==============================================================================
REM Amazon ML Challenge 2026 - Windows One-Click Runner (Auto Multi-Core)
REM ==============================================================================
cd /d "%~dp0"
title Odin Entity Resolution Master Runner

echo ============================================================================
echo    AMAZON ML CHALLENGE 2026: ODIN WINDOWS ONE-CLICK RUNNER
echo ============================================================================

REM Check for python or py launcher
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto FOUND_PY
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py -3"
    goto FOUND_PY
)

IF EXIST "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto FOUND_PY
)
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto FOUND_PY
)
IF EXIST "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto FOUND_PY
)

echo [ERROR] Python was not found in your Windows PATH!
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo Make sure you check "Add python.exe to PATH" during installation.
pause
exit /b 1

:FOUND_PY
echo [System] Using Python Command: %PY_CMD%
echo [System] Starting Odin Master Runner (run.py)...
echo.

%PY_CMD% run.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================================
echo All steps completed successfully!
echo 1. Output matching TSV: output\matching_results.tsv
echo 2. Final Submission ZIP: Odin_submission.zip
echo ============================================================================
pause
