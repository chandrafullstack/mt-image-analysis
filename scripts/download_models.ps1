# Downloads pre-trained model weights from the GitHub release.
# Run from the repo root:    powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
#
# If you trained your own weights, you can skip this — just drop them into outputs/models/.

$ErrorActionPreference = "Stop"

$repo    = "chandrafullstack/mt-image-analysis"
$release = "beta"             # GitHub release tag holding the .pt files
$dest    = Join-Path $PSScriptRoot "..\outputs\models"

$files = @(
    "resnet50_best.pt"
    # Add "unet_best.pt" here once you publish a trained U-Net.
)

New-Item -ItemType Directory -Force -Path $dest | Out-Null

foreach ($f in $files) {
    $url = "https://github.com/$repo/releases/download/$release/$f"
    $out = Join-Path $dest $f
    if (Test-Path $out) {
        Write-Host "[skip] $f already present"
        continue
    }
    Write-Host "[get ] $f"
    try {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    } catch {
        Write-Host "    FAILED — release '$release' or file '$f' may not exist yet." -ForegroundColor Yellow
        Write-Host "    Create one at: https://github.com/$repo/releases/new" -ForegroundColor Yellow
        throw
    }
}

Write-Host ""
Write-Host "Done. Weights are in $dest" -ForegroundColor Green
Write-Host "Next: scripts\start_dashboard.bat   then open  http://localhost:8000"
