param(
    [switch]$ForceRecreateVenv,
    [switch]$InstallCudaTorch,
    [switch]$SkipTorchCheck
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
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

    throw "Python 3.10 or 3.11 was not found. Install Python from https://www.python.org/downloads/ and enable the 'Add python.exe to PATH' option."
}

function Invoke-PythonCommand($PythonCommand, [string[]]$ExtraArgs) {
    & $PythonCommand.Command @($PythonCommand.Args + $ExtraArgs)
}

Write-Step "Checking Python installation"
$pythonCommand = Find-PythonCommand
$pythonVersionOutput = Invoke-PythonCommand $pythonCommand @("--version") 2>&1
Write-Host "Using $($pythonCommand.Command) $($pythonCommand.Args -join ' '): $pythonVersionOutput" -ForegroundColor Green

$venvPath = Join-Path $PSScriptRoot ".venv"
if ($ForceRecreateVenv -and (Test-Path $venvPath)) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Step "Creating virtual environment"
    Invoke-PythonCommand $pythonCommand @("-m", "venv", $venvPath)
}
else {
    Write-Step "Using existing virtual environment"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing project dependencies"
& $venvPip install -r (Join-Path $PSScriptRoot "requirements.txt")

if ($InstallCudaTorch) {
    Write-Step "Installing CUDA-enabled PyTorch"
    & $venvPip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}
else {
    Write-Step "Installing default PyTorch"
    & $venvPip install --upgrade torch torchvision torchaudio
}

if (-not $SkipTorchCheck) {
    Write-Step "Checking PyTorch CUDA availability"
    & $venvPython -c "import torch; from PyQt5 import QtCore; print('pyqt5', QtCore.PYQT_VERSION_STR); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
}

Write-Step "Setup complete"
Write-Host "Activate the environment in VS Code with:" -ForegroundColor Yellow
Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Then run the app with:" -ForegroundColor Yellow
Write-Host "python main.py" -ForegroundColor White
