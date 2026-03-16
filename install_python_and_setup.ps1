param(
    [switch]$ForceRecreateVenv,
    [switch]$SkipTorchCheck
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

Write-Step "Checking Python installation"
$pythonCommand = Find-PythonCommand

if (-not $pythonCommand) {
    if (-not (Test-Command "winget")) {
        throw "Python 3.10/3.11 is missing and winget is not available. Install Python 3.11 manually from https://www.python.org/downloads/windows/ and rerun this script."
    }

    Write-Step "Installing Python 3.11 with winget"
    winget install --id Python.Python.3.11 --source winget --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $pythonCommand = Find-PythonCommand
}

if (-not $pythonCommand) {
    throw "Python installation completed, but PowerShell still cannot find it. Close this terminal, open a new PowerShell window, and rerun the script."
}

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
& $venvPython -m pip install --upgrade pip

Write-Step "Installing project dependencies"
& $venvPip install -r (Join-Path $PSScriptRoot "requirements.txt")

if (-not $SkipTorchCheck) {
    Write-Step "Checking PyTorch CUDA availability"
    & $venvPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
}

Write-Step "Setup complete"
Write-Host "Activate the environment with:" -ForegroundColor Yellow
Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Then run the app with:" -ForegroundColor Yellow
Write-Host "python main.py" -ForegroundColor White
