$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $root 'DriftingIdentifiability'
$paperAxioms = Join-Path $sourceRoot 'Paperaxioms.lean'
$boundary = Join-Path $sourceRoot 'TrustedBoundary.lean'
$hashManifest = Join-Path $root '.trusted/Paperaxioms.sha256'
$errors = [System.Collections.Generic.List[string]]::new()
$rootPrefix = $root.TrimEnd([char[]]@('\', '/')) + [System.IO.Path]::DirectorySeparatorChar

$paperEstablishedAxioms = @(
  'equation_6_loss_value',
  'equation_11_bilinear_mean_shift',
  'equation_16_cfg_target',
  'cfg_weighted_negative_is_mixture',
  'cfg_strength_closed_form',
  'equation_17_matched_batch_drift_zero',
  'antisymmetric_kernel_induces_basis_antisymmetry',
  'equation_31_bilinear_expansion',
  'stopGradientPointGradient_is_gradient',
  'mmdDrift_eq_negative_half_pointGradient',
  'radial_mmd_attraction_repulsion',
  'equation_44_eq_meanShiftDrift',
  'gaussian_mmd_meanShiftKernel',
  'equation_47_matched_zero',
  'sampleMean_meanSquare_le'
) | Sort-Object

$conditionalExternalAxioms = @(
  'IsCharacteristic',
  'characteristic_gradientEmbedding_injective',
  'gaussian_gradient_isCharacteristic',
  'gaussianMeanShift_injective',
  'gaussian_mmd_metrizes_weakConvergence',
  'gaussian_gram_linearIndependent'
) | Sort-Object

$allowedPaperAxioms = @($paperEstablishedAxioms + $conditionalExternalAxioms) | Sort-Object

if (-not (Test-Path -LiteralPath $hashManifest)) {
  $errors.Add('Missing .trusted/Paperaxioms.sha256 trust manifest.')
} else {
  $manifestLine = (Get-Content -LiteralPath $hashManifest -Raw -Encoding UTF8).Trim()
  $expectedHash = ($manifestLine -split '\s+')[0].ToUpperInvariant()
  $actualHash = (Get-FileHash -LiteralPath $paperAxioms -Algorithm SHA256).Hash.ToUpperInvariant()
  if ($actualHash -ne $expectedHash) {
    $errors.Add(
      'Paperaxioms.lean changed without an approved trust-boundary update. ' +
      "Expected $expectedHash but found $actualHash."
    )
  }
}

$leanFiles = @(
  Get-Item -LiteralPath (Join-Path $root 'DriftingIdentifiability.lean')
  Get-ChildItem -LiteralPath $sourceRoot -Filter '*.lean' -File -Recurse
)
$rootModule = Join-Path $root 'DriftingIdentifiability.lean'
foreach ($file in $leanFiles) {
  $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
  # The scanner targets declarations and proof terms, not documentation.
  # Lean permits nested block comments; project comments are non-nested here.
  $code = [regex]::Replace($text, '(?s)/-.*?-/', '')
  $code = [regex]::Replace($code, '(?m)--.*$', '')
  $relative = $file.FullName.Substring($rootPrefix.Length).Replace('\', '/')

  if ($file.FullName -eq $rootModule -and
      $code -match '(?m)^\s*import\s+DriftingIdentifiability\.(Conditional|CharacteristicIdentifiability|GaussianNondegeneracy)\s*$') {
    $errors.Add('The default root module imports conditional research results.')
  }

  if ($code -match '(?m)\b(sorryAx|sorry|admit)\b') {
    $errors.Add("$relative contains a forbidden proof escape: sorry/admit/sorryAx.")
  }
  if ($code -match '(?m)^\s*set_option\s+autoImplicit\s+true\b') {
    $errors.Add("$relative re-enables autoImplicit.")
  }
  if ($code -match '(?m)^\s*(run_tac|elab|syntax|macro|macro_rules|builtin_initialize)\b') {
    $errors.Add("$relative uses metaprogramming that is outside the trusted proof workflow.")
  }
  if ($code -match '(?m)^\s*unsafe\s+(def|theorem|instance)\b') {
    $errors.Add("$relative introduces an unsafe declaration.")
  }

  $declarations = [regex]::Matches(
    $code,
    '(?m)^\s*(axiom|constant|opaque)\s+([A-Za-z_][A-Za-z0-9_.'']*)'
  )

  if ($file.FullName -eq $paperAxioms) {
    $actualNames = @($declarations | ForEach-Object { $_.Groups[2].Value } | Sort-Object)
    $difference = Compare-Object $allowedPaperAxioms $actualNames
    if ($difference) {
      $errors.Add(
        'Paperaxioms.lean axiom allowlist changed: ' +
        (($difference | ForEach-Object { "$($_.SideIndicator)$($_.InputObject)" }) -join ', ')
      )
    }
    if ($declarations | Where-Object { $_.Groups[1].Value -ne 'axiom' }) {
      $errors.Add('Paperaxioms.lean contains a constant/opaque declaration; only reviewed axioms are allowed.')
    }
  } elseif ($declarations.Count -gt 0) {
    foreach ($declaration in $declarations) {
      $errors.Add(
        "$relative declares forbidden $($declaration.Groups[1].Value) " +
        "$($declaration.Groups[2].Value)."
      )
    }
  }

  if ($file.FullName -ne $paperAxioms -and $file.FullName -ne $boundary -and
      $code -match '(?m)^\s*import\s+DriftingIdentifiability\.Paperaxioms\s*$') {
    $errors.Add("$relative bypasses TrustedBoundary and imports Paperaxioms directly.")
  }
}

if ($errors.Count -gt 0) {
  Write-Error ("Trust audit failed:`n - " + ($errors -join "`n - "))
  exit 1
}

Write-Host (
  "Trust audit passed: $($leanFiles.Count) Lean files checked; " +
  "$($paperEstablishedAxioms.Count) paper axioms and " +
  "$($conditionalExternalAxioms.Count) conditional external axioms classified."
)
