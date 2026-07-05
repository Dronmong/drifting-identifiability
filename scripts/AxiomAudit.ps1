$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$lake = (& elan which lake).Trim()
if (-not (Test-Path -LiteralPath $lake)) { throw 'Unable to locate the active Lake executable.' }
$temporaryFile = Join-Path ([System.IO.Path]::GetTempPath()) 'DriftingIdentifiabilityPromotedAxiomAudit.lean'

$declarations = @(
  'DriftingIdentifiability.PaperFiniteIdentifiability.finiteBasisDensitiesEqual',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies_of_energy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.coefficientStability',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empiricalInteractionFrameBound'
)

$allowedProjectAxioms = @(
  'antisymmetric_kernel_induces_basis_antisymmetry',
  'equation_11_bilinear_mean_shift',
  'equation_31_bilinear_expansion'
)

$sourceLines = [System.Collections.Generic.List[string]]::new()
$sourceLines.Add('import DriftingIdentifiability')
foreach ($declaration in $declarations) {
  $sourceLines.Add("#print axioms $declaration")
}

$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
  $temporaryFile,
  ($sourceLines -join [Environment]::NewLine) + [Environment]::NewLine,
  $utf8WithoutBom
)

Push-Location $root
try {
  $output = (& $lake env lean $temporaryFile 2>&1 | Out-String)
  if ($LASTEXITCODE -ne 0) {
    Write-Error $output
    exit $LASTEXITCODE
  }
} finally {
  Pop-Location
  Remove-Item -LiteralPath $temporaryFile -Force -ErrorAction SilentlyContinue
}

$found = [regex]::Matches(
  $output,
  'DriftingIdentifiability\.Paper\.([A-Za-z_][A-Za-z0-9_]*)'
) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

$unexpected = @($found | Where-Object { $_ -notin $allowedProjectAxioms })
if ($unexpected.Count -gt 0) {
  Write-Error (
    'Promoted theorem dependency audit found forbidden project axioms: ' +
    ($unexpected -join ', ')
  )
  exit 1
}

Write-Host (
  "Promoted theorem axiom audit passed: $($declarations.Count) declarations; " +
  'no conditional Gaussian/RKHS dependencies.'
)
