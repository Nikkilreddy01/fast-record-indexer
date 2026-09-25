# Odin Business Entity Resolution — Live Progress & ETA Monitor (PowerShell)
$Host.UI.RawUI.WindowTitle = "Odin Entity Resolution Monitor"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PyCmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PyCmd = ".venv\Scripts\python.exe"
}

& $PyCmd monitor.py $args
