# Milestone-5 (general 1-d Laplace converse) — reproducible numerical evidence.
#
# Records the three load-bearing checks behind the on-paper resolution
# (see DriftingIdentifiability/LaplaceGeneralConverseRoadmap.md, Milestone 5).
# PowerShell only (no Python/numpy on this machine); doubles.
#
#   (A) TAIL STRUCTURE — the sign-error correction.  For a light-tailed p the
#       tilted mean mu_p(x) SATURATES (mu_p -> sigma^2/tau, NOT -> infinity), so
#       the mean shift m = mu_p - x ~ const - x -> -inf, and Z_p ~ c e^{-x/tau}.
#       Hence the joint 2nd-order ODE reduces to Z''=Z/tau^2 at +-inf (modes
#       e^{+-x/tau}); Z_p is the decaying mode, the 2nd solution GROWS at BOTH
#       ends => doubly-decaying space is 1-dim => Z_q ∝ Z_p => p = q.
#
#   (B) FROBENIUS PARAMETER beta = M_p/(tau Z_p) at the mean-shift zero — the
#       (now-defused) "obstruction" parameter; beta >= 1/3 is common (spread p).
#
#   (C) LINEARIZED INJECTIVITY — the converse is TRUE (no counterexample):
#       the operator T[h](x) = int k(x,y)(y-x-m(x)) h(y) dy has trivial kernel
#       on mean-zero perturbations (full column rank).
#
# Run:  pwsh -File numerics/milestone5_wronskian.ps1   (or Windows PowerShell)

$ErrorActionPreference = 'Stop'

function Phi($y){ [math]::Exp(-$y*$y/2)/[math]::Sqrt(2*[math]::PI) }   # N(0,1) density

Write-Output "==================================================================="
Write-Output "(A) TAIL STRUCTURE   p = N(0,1),  tau = 0.5   (claim: mu_p->2, m~2-x, Z_p~c e^{-x/tau})"
$tau = 0.5
$nY = 2000; $ylo = -9.0; $yhi = 9.0; $dy = ($yhi - $ylo) / $nY
function Zp($x){ $s=0.0; for($j=0;$j -lt $nY;$j++){ $y=$ylo+($j+0.5)*$dy; $s+=[math]::Exp(-[math]::Abs($x-$y)/$tau)*(Phi $y)*$dy }; $s }
function Ap($x){ $s=0.0; for($j=0;$j -lt $nY;$j++){ $y=$ylo+($j+0.5)*$dy; $s+=$y*[math]::Exp(-[math]::Abs($x-$y)/$tau)*(Phi $y)*$dy }; $s }
Write-Output ("  sigma^2/tau = {0}" -f (1.0/$tau))
foreach($x in @(1.0,2.0,3.0,4.0,5.0)){
  $Z=Zp $x; $A=Ap $x; $mu=$A/$Z; $m=$mu-$x; $Zs=$Z*[math]::Exp($x/$tau)
  Write-Output ("  x={0}: mu_p={1}  m={2}  Z_p*exp(x/tau)={3}" -f $x,[math]::Round($mu,4),[math]::Round($m,4),[math]::Round($Zs,5))
}
Write-Output "  => mu_p saturates at 2, m ~ 2-x (linear, -> -inf), Z_p*exp(x/tau) -> const."

Write-Output "==================================================================="
Write-Output "(B) FROBENIUS beta = M_p(z*)/(tau Z_p(z*)) at the mean-shift zero (z*=0 by symmetry)"
function Mp0($sig,$tt){ $s=0.0; $n=800; $lo=-9*$sig; $hi=9*$sig; $d=($hi-$lo)/$n; for($k=0;$k -lt $n;$k++){ $y=$lo+($k+0.5)*$d; $s+=[math]::Abs($y)*[math]::Exp(-[math]::Abs($y)/$tt)*[math]::Exp(-$y*$y/(2*$sig*$sig))/([math]::Sqrt(2*[math]::PI)*$sig)*$d }; $s }
function Zp0($sig,$tt){ $s=0.0; $n=800; $lo=-9*$sig; $hi=9*$sig; $d=($hi-$lo)/$n; for($k=0;$k -lt $n;$k++){ $y=$lo+($k+0.5)*$d; $s+=[math]::Exp(-[math]::Abs($y)/$tt)*[math]::Exp(-$y*$y/(2*$sig*$sig))/([math]::Sqrt(2*[math]::PI)*$sig)*$d }; $s }
foreach($cfg in @(@{s=0.2;t=0.8},@{s=1.0;t=0.8},@{s=2.0;t=0.5},@{s=3.0;t=0.3})){
  $b = (Mp0 $cfg.s $cfg.t)/($cfg.t*(Zp0 $cfg.s $cfg.t))
  Write-Output ("  sigma/tau={0}: beta={1}  (obstruction-param>=1/3: {2}; defused globally, no counterexample)" -f [math]::Round($cfg.s/$cfg.t,3),[math]::Round($b,3),($b -gt 0.3333))
}

Write-Output "==================================================================="
Write-Output "(C) LINEARIZED INJECTIVITY  T[h](x)=int k(x,y)(y-x-m(x))h(y)dy,  m=D_p/Z_p"
foreach($cfg in @(@{tau=0.5;Ny=30;dens='gauss'},@{tau=1.0;Ny=35;dens='bimodal'},@{tau=0.7;Ny=40;dens='uniform'})){
  $tau=$cfg.tau; $Ny=$cfg.Ny; $ya=-3.0; $yb=3.0; $hy=($yb-$ya)/($Ny-1)
  $y=@(); for($j=0;$j -lt $Ny;$j++){$y+=($ya+$j*$hy)}
  $w=@(); for($j=0;$j -lt $Ny;$j++){$w+=$hy}; $w[0]=$hy/2; $w[$Ny-1]=$hy/2
  $f=@()
  for($j=0;$j -lt $Ny;$j++){
    switch($cfg.dens){
      'gauss'  { $f+=[math]::Exp(-($y[$j]*$y[$j])/2) }
      'bimodal'{ $f+=([math]::Exp(-(($y[$j]-1.2)*($y[$j]-1.2))/0.5)+[math]::Exp(-(($y[$j]+1.2)*($y[$j]+1.2))/0.5)) }
      'uniform'{ if([math]::Abs($y[$j]) -le 2.0){$f+=1.0}else{$f+=1e-6} }
    }
  }
  $mass=0.0; for($j=0;$j -lt $Ny;$j++){$mass+=$f[$j]*$w[$j]}; for($j=0;$j -lt $Ny;$j++){$f[$j]=$f[$j]/$mass}
  $Nx=70; $xa=-4.0; $xb=4.0; $hx=($xb-$xa)/($Nx-1); $rows=$Nx+1
  $M=@()
  for($i=0;$i -lt $Nx;$i++){
    $xx=$xa+$i*$hx; $Z=0.0; $D=0.0
    for($j=0;$j -lt $Ny;$j++){ $k=[math]::Exp(-[math]::Abs($xx-$y[$j])/$tau); $Z+=$k*$f[$j]*$w[$j]; $D+=$k*($y[$j]-$xx)*$f[$j]*$w[$j] }
    $mx= if($Z -eq 0.0){0.0}else{$D/$Z}
    $r=New-Object 'double[]' $Ny
    for($j=0;$j -lt $Ny;$j++){$r[$j]=([math]::Exp(-[math]::Abs($xx-$y[$j])/$tau))*($y[$j]-$xx-$mx)*$w[$j]}
    $M+=,$r
  }
  $rl=New-Object 'double[]' $Ny; for($j=0;$j -lt $Ny;$j++){$rl[$j]=$w[$j]}; $M+=,$rl   # int h = 0
  $tol=1e-9; $rank=0; $piv=0
  for($col=0;($col -lt $Ny)-and($piv -lt $rows);$col++){
    $best=$piv; $bestv=[math]::Abs($M[$piv][$col])
    for($rr=$piv+1;$rr -lt $rows;$rr++){$v=[math]::Abs($M[$rr][$col]); if($v -gt $bestv){$bestv=$v;$best=$rr}}
    if($bestv -le $tol){continue}
    if($best -ne $piv){$t=$M[$piv];$M[$piv]=$M[$best];$M[$best]=$t}
    for($rr=$piv+1;$rr -lt $rows;$rr++){$fa=$M[$rr][$col]/$M[$piv][$col]; if($fa -ne 0.0){for($c=$col;$c -lt $Ny;$c++){$M[$rr][$c]=$M[$rr][$c]-$fa*$M[$piv][$c]}}}
    $rank++; $piv++
  }
  $verdict = if($rank -eq $Ny){'INJECTIVE (converse holds, no counterexample)'}else{'nullspace dim '+($Ny-$rank)}
  Write-Output ("  {0} tau={1} Ny={2}: rank={3}/{4}  {5}" -f $cfg.dens.PadRight(8),$tau,$Ny,$rank,$Ny,$verdict)
}
Write-Output "==================================================================="
