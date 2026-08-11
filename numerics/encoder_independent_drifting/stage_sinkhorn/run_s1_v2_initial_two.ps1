$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$Uv = Join-Path $HOME ".local\bin\uv.exe"
$Log = Join-Path $PSScriptRoot "s1_v2_initial_two.stdout.txt"
$Result = Join-Path $PSScriptRoot "s1_v2_initial_two.json"
$Freeze = Join-Path $PSScriptRoot "s1_v2_freeze.json"

if (-not (Test-Path -LiteralPath $Uv)) {
    throw "uv was not found at $Uv"
}
if (-not (Test-Path -LiteralPath $Freeze)) {
    throw "corrected S1 freeze does not exist: $Freeze"
}
if (Test-Path -LiteralPath $Result) {
    throw "corrected S1 result already exists: $Result"
}
if (Test-Path -LiteralPath $Log) {
    throw "corrected S1 log already exists: $Log"
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "0"
Set-Location -LiteralPath $Root

& $Uv run `
    --python 3.12 `
    --index https://download.pytorch.org/whl/cu126 `
    --with torch==2.7.1 `
    --with torchvision==0.22.1 `
    --with numpy `
    --with scipy `
    --with pillow `
    python -m numerics.encoder_independent_drifting.stage_sinkhorn.s1 `
    --freeze $Freeze `
    --device cuda `
    --threads 4 2>&1 | Tee-Object -FilePath $Log

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "corrected S1 initial-two runner failed with exit code $exitCode"
}
