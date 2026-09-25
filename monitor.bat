@echo off
REM ==============================================================================
REM  Odin Business Entity Resolution — Live Progress & ETA Monitor (Windows)
REM ==============================================================================
cd /d "%~dp0"
title Odin Entity Resolution Monitor

IF EXIST ".venv\Scripts\python.exe" (
    SET "PY_CMD=.venv\Scripts\python.exe"
) ELSE (
    SET "PY_CMD=python"
)

%PY_CMD% monitor.py %*
pause
