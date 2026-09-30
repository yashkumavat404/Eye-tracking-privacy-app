$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\bootstrap_all.ps1 first."
}

Write-Host "==> Installing PyInstaller" -ForegroundColor Cyan
& $python -m pip install --upgrade pyinstaller

Write-Host "==> Removing previous build output" -ForegroundColor Cyan
foreach ($path in @((Join-Path $root "build"), (Join-Path $root "dist\PrivacySpotlight"))) {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

Write-Host "==> Building PrivacySpotlight (onedir)" -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean --windowed --onedir --name PrivacySpotlight privacy_dashboard.py

$exe = Join-Path $root "dist\PrivacySpotlight\PrivacySpotlight.exe"
if (-not (Test-Path $exe)) {
    throw "PyInstaller completed without producing $exe"
}

Write-Host "`nBuild complete: $exe" -ForegroundColor Green
Write-Host "Test this folder before creating a single-file installer." -ForegroundColor Yellow
