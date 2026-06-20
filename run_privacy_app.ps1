$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workspace

$venvPython = Join-Path $workspace ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    py -3 -m venv .venv
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Checking and installing required libraries..."
& $venvPython -m pip install -r requirements.txt

Write-Host "Starting Privacy App..."
& $venvPython main.py
