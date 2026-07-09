param(
  [Parameter(Mandatory = $true)]
  [string]$Declaration,
  [string]$ImportModule = 'DriftingIdentifiability'
)

$ErrorActionPreference = 'Stop'

if ($Declaration -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
  throw 'Declaration must be a fully qualified Lean identifier.'
}
if ($ImportModule -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
  throw 'ImportModule must be a fully qualified Lean module name.'
}

$root = Split-Path -Parent $PSScriptRoot
$lake = (& elan which lake).Trim()
if (-not (Test-Path -LiteralPath $lake)) { throw 'Unable to locate the active Lake executable.' }
$temporaryFile = Join-Path ([System.IO.Path]::GetTempPath()) 'DriftingIdentifiabilityAxiomAudit.lean'
$source = "import $ImportModule`n#print axioms $Declaration`n"

$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($temporaryFile, $source, $utf8WithoutBom)
Push-Location $root
try {
  & $lake env lean $temporaryFile
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
  Remove-Item -LiteralPath $temporaryFile -Force -ErrorAction SilentlyContinue
}
