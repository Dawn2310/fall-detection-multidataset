# run_pipeline.ps1 — Run the full experiment pipeline with logging.
# Usage: .\run_pipeline.ps1
#        .\run_pipeline.ps1 -UseDL

param(
    [switch]$UseDL = $false,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $LogFile) {
    $stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $ProjectRoot "results\pipeline_$stamp.log"
}
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

$cmdArgs = @("src/run_experiments.py")
if ($UseDL) { $cmdArgs += "--use_dl" }

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Fall Detection Pipeline"                        -ForegroundColor Cyan
Write-Host "  Start : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  Log   : $LogFile"
Write-Host "  Cmd   : python $($cmdArgs -join ' ')"
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

& python @cmdArgs 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "===============================================" -ForegroundColor Green
Write-Host "  Finish: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  Log   : $LogFile"
Write-Host "===============================================" -ForegroundColor Green
