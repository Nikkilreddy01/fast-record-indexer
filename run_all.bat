@echo off
REM ==============================================================================
REM Amazon ML Challenge 2026 - Windows One-Click Runner (Auto Multi-Core)
REM ==============================================================================

echo ============================================================================
echo    AMAZON ML CHALLENGE 2026: WINDOWS ONE-CLICK RUNNER
echo ============================================================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo [System] Detected Processor: %PROCESSOR_ARCHITECTURE%
echo [System] Detected Logical Cores: %NUMBER_OF_PROCESSORS%

REM Run universal Python runner
python run_all.py

if %errorlevel% neq 0 (
    echo [ERROR] Pipeline failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo ============================================================================
echo All steps completed successfully!
echo 1. Output matching TSV: output\matching_results.tsv
echo 2. Final Submission ZIP: Odin_submission.zip
echo ============================================================================
pause
