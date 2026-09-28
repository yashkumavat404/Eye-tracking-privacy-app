param(
    [switch]$ForceRecreateVenv,
    [switch]$SkipCudaCheck,
    [switch]$InstallCudaTorch
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Test-Command([string]$Name) {
    try {
        Get-Command $Name -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Find-PythonCommand {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "py"; Args = @("-3.10") },
        @{ Command = "python"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            $versionOutput = & $candidate.Command @($candidate.Args + @("--version")) 2>&1
            if ($versionOutput -match "Python 3\.(10|11)") {
                return $candidate
            }
        }
        catch {
        }
    }

    return $null
}

function Invoke-PythonCommand($PythonCommand, [string[]]$ExtraArgs) {
    & $PythonCommand.Command @($PythonCommand.Args + $ExtraArgs)
}

Write-Step "Checking for winget"
if (-not (Test-Command "winget")) {
    throw "winget is not available. Install App Installer from Microsoft Store or install Python manually, then rerun this script."
}

Write-Step "Checking Python installation"
$pythonCommand = Find-PythonCommand
if (-not $pythonCommand) {
    Write-Step "Installing Python 3.11"
    winget install --id Python.Python.3.11 --source winget --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $pythonCommand = Find-PythonCommand
}

if (-not $pythonCommand) {
    throw "Python 3.11 was installed, but this terminal cannot see it yet. Close PowerShell, open a new one, and rerun the script."
}

$pythonVersionOutput = Invoke-PythonCommand $pythonCommand @("--version") 2>&1
Write-Host "Using $($pythonCommand.Command) $($pythonCommand.Args -join ' '): $pythonVersionOutput" -ForegroundColor Green

$projectRoot = $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
if ($ForceRecreateVenv -and (Test-Path $venvPath)) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Step "Creating project virtual environment"
    Invoke-PythonCommand $pythonCommand @("-m", "venv", $venvPath)
}
else {
    Write-Step "Using existing virtual environment"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

Write-Step "Upgrading pip, setuptools, and wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing project dependencies"
& $venvPip install -r (Join-Path $projectRoot "requirements.txt")

if ($InstallCudaTorch) {
    Write-Step "Installing CUDA-enabled PyTorch"
    & $venvPip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}
else {
    Write-Step "Installing default PyTorch"
    & $venvPip install --upgrade torch torchvision torchaudio
}

Write-Step "Verifying core packages"
& $venvPython -c "import sys, cv2, mediapipe, mss, numpy, torch; from PyQt6 import QtCore; print('python', sys.version.split()[0]); print('opencv', cv2.__version__); print('mediapipe', mediapipe.__version__); print('mss', mss.__version__); print('numpy', numpy.__version__); print('pyqt5', QtCore.PYQT_VERSION_STR); print('torch', torch.__version__)"

if (-not $SkipCudaCheck) {
    Write-Step "Checking CUDA availability in PyTorch"
    & $venvPython -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
}

Write-Step "Bootstrap complete"
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd `"$projectRoot`"" -ForegroundColor White
Write-Host "2. .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. python privacy_dashboard.py" -ForegroundColor White
Write-Host "Optional: rerun with -InstallCudaTorch to fetch the NVIDIA CUDA build of PyTorch." -ForegroundColor White
