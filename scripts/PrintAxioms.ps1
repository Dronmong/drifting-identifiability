param(
  [Parameter(Mandatory = $true)]
  [string]$Declaration
)

$ErrorActionPreference = 'Stop'

if ($Declaration -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
  throw 'Declaration must be a fully qualified Lean identifier.'
}

$root = Split-Path -Parent $PSScriptRoot
$temporaryFile = Join-Path ([System.IO.Path]::GetTempPath()) 'DriftingIdentifiabilityAxiomAudit.lean'
$source = "import DriftingIdentifiability`n#print axioms $Declaration`n"

$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($temporaryFile, $source, $utf8WithoutBom)
Push-Location $root
try {
  lake env lean $temporaryFile
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
  Remove-Item -LiteralPath $temporaryFile -Force -ErrorAction SilentlyContinue
}
