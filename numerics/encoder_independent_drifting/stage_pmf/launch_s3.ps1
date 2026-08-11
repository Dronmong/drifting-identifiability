param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")),
    [string]$LogPath = (Join-Path $PSScriptRoot "s3_full_run.log")
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& uv run --python 3.12 `
  --extra-index-url https://download.pytorch.org/whl/cu126 `
  --index-strategy unsafe-best-match `
  --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 `
  --with numpy --with scipy --with pillow `
  python -m numerics.encoder_independent_drifting.stage_pmf.run_two_unit `
  --device cuda --threads 4 --resume *>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
    throw "S3 runner exited with code $LASTEXITCODE"
}
