$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root

# Required by deterministic CUDA matrix multiplication. core.py checks this
# again before the first training update.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$UvPrefix = @(
    "run",
    "--python", "3.12",
    "--extra-index-url", "https://download.pytorch.org/whl/cu126",
    "--index-strategy", "unsafe-best-match",
    "--with", "torch==2.7.1+cu126",
    "--with", "torchvision==0.22.1+cu126",
    "--with", "numpy",
    "--with", "scipy",
    "--with", "pillow",
    "python", "-m"
)

$LogDirectory = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$Log = Join-Path $LogDirectory ("b3_overnight_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Invoke-B3PythonModule {
    param([string[]] $ModuleArguments)
    & uv @UvPrefix @ModuleArguments 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "B3 command failed with exit code ${LASTEXITCODE}: $($ModuleArguments -join ' ')"
    }
}

foreach ($Unit in 600, 601, 602) {
    $Result = Join-Path $PSScriptRoot "b3_unit_$Unit.json"
    $Sidecar = "$Result.sha256"
    if ((Test-Path $Result) -and (Test-Path $Sidecar)) {
        "Skipping completed B3 unit $Unit" | Tee-Object -FilePath $Log -Append
        continue
    }
    if ((Test-Path $Result) -or (Test-Path $Sidecar)) {
        throw "Incomplete final artifact exists for B3 unit $Unit; inspect it before resuming"
    }
    Invoke-B3PythonModule @(
        "numerics.encoder_independent_drifting.stage_b3.run_unit",
        "--unit", "$Unit",
        "--device", "cuda"
    )
}

$Aggregate = Join-Path $PSScriptRoot "b3_matched_reference.json"
$AggregateSidecar = "$Aggregate.sha256"
if ((Test-Path $Aggregate) -or (Test-Path $AggregateSidecar)) {
    throw "B3 aggregate already exists; refusing to overwrite a consumed result"
}
Invoke-B3PythonModule @(
    "numerics.encoder_independent_drifting.stage_b3.aggregate"
)

"B3 overnight run and aggregation completed." | Tee-Object -FilePath $Log -Append
