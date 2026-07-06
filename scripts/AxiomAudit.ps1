$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$lake = (& elan which lake).Trim()
if (-not (Test-Path -LiteralPath $lake)) { throw 'Unable to locate the active Lake executable.' }
$temporaryFile = Join-Path ([System.IO.Path]::GetTempPath()) 'DriftingIdentifiabilityPromotedAxiomAudit.lean'

$declarations = @(
  'DriftingIdentifiability.PaperFiniteIdentifiability.finiteBasisDensitiesEqual',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies_of_probeZero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.finitePopulationMeanShift_identifies_of_energy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.coefficientStability',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.coefficientStability_probeEnergy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.interactionFrameBound_of_linearIndependent',
  'DriftingIdentifiability.PaperFiniteIdentifiability.interactionFrameBound_inverseCertificate',
  'DriftingIdentifiability.PaperFiniteIdentifiability.interactionFrameBound_of_uniformPerturbation',
  'DriftingIdentifiability.PaperFiniteIdentifiability.interactionFrameBound_of_dualCertificate',
  'DriftingIdentifiability.PaperFiniteIdentifiability.interactionFrameBound_le_interactionNorm',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_frameConstant_le',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointFrameConstant_le',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_identifies_of_probeZero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Gaussian_identifies_of_probeZero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.bumpDensity_contDiff',
  'DriftingIdentifiability.PaperFiniteIdentifiability.bumpBasisMeasure_noAtoms',
  'DriftingIdentifiability.PaperFiniteIdentifiability.bumpInteractionFrameBound',
  'DriftingIdentifiability.PaperFiniteIdentifiability.bumpGaussian_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.coefficientStability_of_estimate',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.estimate_failure_measure_le',
  'DriftingIdentifiability.PaperFiniteIdentifiability.meas_gt_le_meanSquare_div',
  'DriftingIdentifiability.PaperFiniteIdentifiability.PopulationMeanShiftFiniteSetup.estimate_failure_le_meanSquare',
  'DriftingIdentifiability.PaperFiniteIdentifiability.sampleMean_concentration',
  'DriftingIdentifiability.SelfNormalized.selfNormalized_meanSquare_le',
  'DriftingIdentifiability.Algorithm2.algorithm2Affinity_nonneg',
  'DriftingIdentifiability.Algorithm2.algorithm2Affinity_le_one',
  'DriftingIdentifiability.Algorithm2.algorithm2PositiveWeight_le',
  'DriftingIdentifiability.Algorithm2.algorithm2NegativeWeight_le',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_eq_affinityPairSum',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_eq_massScaledCentroid',
  'DriftingIdentifiability.Algorithm2.algorithm2PositiveMass_pos',
  'DriftingIdentifiability.Algorithm2.algorithm2NegativeMass_pos',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_eq_massProduct_centroidDiff',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_norm_le',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_norm_le_affinityMass',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_matched_zero',
  'DriftingIdentifiability.Algorithm2.algorithm2ColumnReweightedWeight_pos',
  'DriftingIdentifiability.Algorithm2.algorithm2Affinity_false_eq_rowScale_mul_columnReweightedWeight',
  'DriftingIdentifiability.Algorithm2.algorithm2PositiveCentroid_false_eq_columnReweighted',
  'DriftingIdentifiability.Algorithm2.algorithm2NegativeCentroid_false_eq_columnReweighted',
  'DriftingIdentifiability.Algorithm2.algorithm2Drift_false_eq_zero_iff_centroidDiff_eq_zero',
  'DriftingIdentifiability.Algorithm2.centroidDiff_norm_le_inv_massProduct_mul_drift_norm',
  'DriftingIdentifiability.Algorithm2.algorithm2PositiveCentroid_false_meanSquare_le',
  'DriftingIdentifiability.Algorithm2.algorithm2NegativeCentroid_false_meanSquare_le',
  'DriftingIdentifiability.Algorithm2.meanSquare_sub_sub_le_two_add',
  'DriftingIdentifiability.PaperFiniteIdentifiability.linearIndependent_weightedGeometricProfiles',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_basisNondegenerate',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointFrameBound',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointCertifiedFrameBound',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_identifies',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_coefficientStability',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_coefficientStability_one',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPoint_coefficientStability_probeEnergy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empiricalInteractionFrameBound',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Gaussian_identifies',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Gaussian_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Gaussian_coefficientStability',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Gaussian_coefficientStability_probeEnergy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointND_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointND_coefficientStability_probeEnergy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointCertifiedProbe_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.gaussianEmpiricalPointCertifiedProbe_coefficientStability_probeEnergy',
  'DriftingIdentifiability.PaperFiniteIdentifiability.continuousPerturbation_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Laplace_identifies_of_probeEnergy_eq_zero',
  'DriftingIdentifiability.PaperFiniteIdentifiability.empirical01Laplace_coefficientStability_probeEnergy'
)

$allowedProjectAxioms = @(
  'antisymmetric_kernel_induces_basis_antisymmetry',
  'equation_11_bilinear_mean_shift',
  'equation_31_bilinear_expansion',
  'sampleMean_meanSquare_le'
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
