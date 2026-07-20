```text
D3: held-out benchmark; base=(tau*=0.35, eta*=0.0525, mask on) vs modified=geo-fixed
  H-1d-K4-eq/missing          : base 0.00376  mod 0.00323  ratio 0.857
  H-1d-K4-eq/covered          : base 0.00383  mod 0.00329  ratio 0.858
  H-1d-K5-uneq/missing        : base 0.01993  mod 0.01983  ratio 0.995
  H-1d-K5-uneq/covered        : base 0.00660  mod 0.00680  ratio 1.031
  H-2d-K6-sep/missing         : base 0.00754  mod 0.00754  ratio 1.000
  H-2d-K6-sep/covered         : base 0.00775  mod 0.00831  ratio 1.073
  H-2d-K3-overlap/missing     : base 0.00604  mod 0.00666  ratio 1.103
  H-2d-K3-overlap/covered     : base 0.00694  mod 0.00715  ratio 1.030
  H-2d-K4-hetero/missing      : base 0.00592  mod 0.00650  ratio 1.099
  H-2d-K4-hetero/covered      : base 0.00596  mod 0.00629  ratio 1.055
  H-ring/missing              : base 0.00679  mod 0.00427  ratio 0.628
  H-ring/covered              : base 0.00577  mod 0.00457  ratio 0.792
  H-circles/missing           : base 0.00806  mod 0.00863  ratio 1.071
  H-circles/covered           : base 0.00752  mod 0.00787  ratio 1.046
  H-moons/missing             : base 0.00673  mod 0.00355  ratio 0.527
  H-moons/covered             : base 0.00580  mod 0.00342  ratio 0.590
  H-skew/missing              : base 0.00349  mod 0.00368  ratio 1.055
  H-skew/covered              : base 0.00357  mod 0.00388  ratio 1.088
  aggregate paired ratio 1.031 CI[1.019,1.041]  degraded cells 1/18  KM base 1 mod 1  non-gauss ratio 1.046 CI[1.035,1.059]
  GATE D3: FAIL (['crit1_ratio_le_0.8', 'crit2_ci_hi_lt_1', 'crit5_nongauss_holds'] failing)
```
