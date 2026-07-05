$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$lake = (& elan which lake).Trim()
if (-not (Test-Path -LiteralPath $lake)) { throw 'Unable to locate the active Lake executable.' }

& (Join-Path $PSScriptRoot 'TrustAudit.ps1')

Push-Location $root
try {
  & $lake build --wfail
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  $conditionalModules = @(
    'CharacteristicIdentifiability',
    'GaussianNondegeneracy'
  )
  foreach ($module in $conditionalModules) {
    $source = Join-Path $root "DriftingIdentifiability/$module.lean"
    $output = Join-Path $root ".lake/build/lib/lean/DriftingIdentifiability/$module.olean"
    $interface = Join-Path $root ".lake/build/lib/lean/DriftingIdentifiability/$module.ilean"
    & $lake env lean $source -o $output -i $interface
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  & $lake env lean (Join-Path $root 'DriftingIdentifiability/Conditional.lean')
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
}

& (Join-Path $PSScriptRoot 'AxiomAudit.ps1')

Write-Host 'All project checks passed.'
