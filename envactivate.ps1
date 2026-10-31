# Activate Python virtual environment
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Verify Python
python --version
python -c "import sys; print(sys.executable)"
