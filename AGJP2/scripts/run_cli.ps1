param([string]$Env = "deepseek-ode")

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "Starting DeepSeek ODE Tutor CLI..." -ForegroundColor Cyan
Set-Location $ProjectRoot

conda run -n $Env python src/cli.py 