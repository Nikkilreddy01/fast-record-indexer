#!/bin/bash
# Odin Live Pipeline Progress & ETA Monitor
cd "$(dirname "$0")"

# Prefer virtual environment python if available
if [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD monitor.py "$@"
