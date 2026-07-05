$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot 'TrustAudit.ps1')

Push-Location $root
try {
  lake build --wfail
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
}

Write-Host 'All project checks passed.'
