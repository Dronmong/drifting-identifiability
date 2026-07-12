# 3B (multiple mean-shift zeros) — reproducible check of the residue-sign structure
# behind the derivation plan in DriftingIdentifiability/LaplaceACDerivation.md.
#
# For a bimodal p (two Gaussians) the mean shift m = D_p/Z_p has SEVERAL zeros.
# The plan proves W == 0 on every interior interval via the boundedness argument
# at UPWARD (even) crossings: there m'>0 and mu'>0, so the Abel residue
# gamma = 2 mu'/m' > 0, forcing |W| -> infinity unless W == 0; W is bounded
# (continuous, a.c.), so W == 0 on both flanking intervals. Downward (odd)
# crossings have gamma < 0 (harmless).  Upward zeros are adjacent to every
# interior interval (consecutive zeros alternate parity), so W == 0 throughout.
#
# This script prints, for each mean-shift zero: m'(z), mu'(z), the crossing type,
# and the residue gamma, confirming gamma>0 exactly at upward crossings and
# mu'(z)>0 at every zero.  PowerShell only (no Python on this machine).
#
# Run:  powershell -ExecutionPolicy Bypass -File numerics/ac_converse_3b_residues.ps1

$ErrorActionPreference='Stop'
$tau=0.6
$nY=1201; $ylo=-10.0; $yhi=10.0; $dy=($yhi-$ylo)/$nY
$Ygrid=New-Object 'double[]' $nY; $Wt=New-Object 'double[]' $nY
$mass=0.0
for($j=0;$j -lt $nY;$j++){ $yy=$ylo+($j+0.5)*$dy; $Ygrid[$j]=$yy
  $dd=[math]::Exp(-([math]::Pow($yy-3.0,2))/(2*0.7*0.7))+[math]::Exp(-([math]::Pow($yy+3.0,2))/(2*0.7*0.7))
  $Wt[$j]=$dd; $mass+=$dd*$dy }
for($j=0;$j -lt $nY;$j++){ $Wt[$j]=$Wt[$j]/$mass }
function Zf($x){ $s=0.0; for($j=0;$j -lt $nY;$j++){ $s+=[math]::Exp(-[math]::Abs($x-$Ygrid[$j])/$tau)*$Wt[$j]*$dy }; $s }
function Af($x){ $s=0.0; for($j=0;$j -lt $nY;$j++){ $s+=$Ygrid[$j]*[math]::Exp(-[math]::Abs($x-$Ygrid[$j])/$tau)*$Wt[$j]*$dy }; $s }
function Muf($x){ (Af $x)/(Zf $x) }
function Mf($x){ (Muf $x)-$x }
function Mup($x){ $h=2e-4; ((Muf ($x+$h))-(Muf ($x-$h)))/(2*$h) }
function Mpf($x){ $h=2e-4; ((Mf ($x+$h))-(Mf ($x-$h)))/(2*$h) }

$nx=601; $xa=-8.0; $xb=8.0; $hx=($xb-$xa)/($nx-1)
$prevx=$xa; $prevm=Mf $xa; $zeros=@()
for($i=1;$i -lt $nx;$i++){ $xx=$xa+$i*$hx; $mm=Mf $xx
  if($prevm*$mm -lt 0){ $zz=($prevx*[math]::Abs($mm)+$xx*[math]::Abs($prevm))/([math]::Abs($prevm)+[math]::Abs($mm)); $zeros+=$zz }
  $prevx=$xx; $prevm=$mm }
Write-Output ("tau={0}, #mean-shift zeros = {1}  (want >1 to exercise 3B)" -f $tau,$zeros.Count)
Write-Output ""
Write-Output "   z_k       m'(z)     mu'(z)   crossing    gamma=2mu'/m'"
foreach($zz in $zeros){
  $mpz=Mpf $zz; $mupz=Mup $zz; $g=2*$mupz/$mpz
  $kind = if($mpz -gt 0){"UP (even) "}else{"DOWN(odd) "}
  $flag = if($g -gt 0){"g>0 -> |W|->inf -> W=0 FORCED"}else{"g<0 -> harmless leak"}
  Write-Output ("  {0,8:F4}  {1,8:F4}  {2,8:F4}  {3}  {4,7:F3}  {5}" -f $zz,$mpz,$mupz,$kind,$g,$flag)
}
$mn=[double]::PositiveInfinity
for($i=0;$i -lt 121;$i++){ $xx=-8.0+16.0*$i/120.0; $v=Mup $xx; if($v -lt $mn){$mn=$v} }
Write-Output ""
Write-Output ("  min mu' over grid = {0:F5}   (want > 0: strict monotone tilted mean, lemma B1)" -f $mn)
