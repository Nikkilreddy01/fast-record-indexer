# ==============================================================================
# Amazon ML Challenge 2026 - Windows PowerShell Runner (Auto Multi-Core)
# ==============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "   AMAZON ML CHALLENGE 2026: WINDOWS POWERSHELL RUNNER" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan

# Detect Python
try {
    $pyVer = python --version
    Write-Host "[System] Python found: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from python.org and ensure 'Add to PATH' is checked." -ForegroundColor Yellow
    Exit 1
}

# Detect CPU cores
$cores = [System.Environment]::ProcessorCount
Write-Host "[System] Detected CPU Cores / Logical Processors: $cores" -ForegroundColor Green

# Run universal runner
python run_all.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host " SUCCESS: Output matching TSV: output\matching_results.tsv" -ForegroundColor Green
    Write-Host " Final submission package:    Odin_submission.zip" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
} else {
    Write-Host " Pipeline failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
