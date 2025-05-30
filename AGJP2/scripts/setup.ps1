# Build conda env, install deps, download weights
param([string]$Env = "deepseek-ode")

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "Project root: $ProjectRoot" -ForegroundColor Yellow
Set-Location $ProjectRoot

Write-Host "[1/3] Creating/activating Conda env $Env" -ForegroundColor Cyan
try {
    conda env create -f environment.yml -n $Env
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Environment may already exist, updating instead..." -ForegroundColor Yellow
        conda env update -f environment.yml -n $Env
    }
} catch {
    Write-Host "Error creating conda environment: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Installing pip extras" -ForegroundColor Cyan
conda run -n $Env pip install -r requirements.txt

Write-Host "[3/3] Downloading model weights" -ForegroundColor Cyan
conda run -n $Env python download_weights.py

Write-Host "✅ Setup complete! You can now run:" -ForegroundColor Green
Write-Host "  scripts\run_cli.ps1   # for interactive mode" -ForegroundColor Cyan
Write-Host "  scripts\run_api.ps1   # for REST API mode" -ForegroundColor Cyan 