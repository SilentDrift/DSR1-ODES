param([string]$Env = "deepseek-ode")

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "Starting DeepSeek ODE Tutor API server..." -ForegroundColor Cyan
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Set-Location $ProjectRoot

conda run -n $Env uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1 