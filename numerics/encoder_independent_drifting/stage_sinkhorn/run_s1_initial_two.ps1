$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$Uv = Join-Path $HOME ".local\bin\uv.exe"
$Log = Join-Path $PSScriptRoot "s1_initial_two.stdout.txt"
$Result = Join-Path $PSScriptRoot "s1_initial_two.json"

if (-not (Test-Path -LiteralPath $Uv)) {
    throw "uv was not found at $Uv"
}
if (Test-Path -LiteralPath $Result) {
    throw "S1 result already exists: $Result"
}
if (Test-Path -LiteralPath $Log) {
    throw "S1 log already exists: $Log"
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Set-Location -LiteralPath $Root

& $Uv run `
    --python 3.12 `
    --extra-index-url https://download.pytorch.org/whl/cu126 `
    --index-strategy unsafe-best-match `
    --with torch==2.7.1+cu126 `
    --with torchvision==0.22.1+cu126 `
    --with numpy `
    --with scipy `
    --with pillow `
    python -m numerics.encoder_independent_drifting.stage_sinkhorn.s1 `
    --device cuda `
    --threads 4 2>&1 | Tee-Object -FilePath $Log

if ($LASTEXITCODE -ne 0) {
    throw "S1 initial-two runner failed with exit code $LASTEXITCODE"
}
