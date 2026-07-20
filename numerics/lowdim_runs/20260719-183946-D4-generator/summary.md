```text
D4: learned-generator transfer; base=(tau*=0.35, eta*=0.0525, mask on) vs modified=geo-fixed
  invariant MLP gradcheck: PASS
  invariant matched-batch zero generator update: PASS
  H-1d-K4-eq    : base 0.01303  mod 0.01354  ratio 1.039
  H-2d-K6-sep   : base 0.01153  mod 0.01112  ratio 0.964
  H-moons       : base 0.00757  mod 0.00747  ratio 0.987
  H-skew        : base 0.00751  mod 0.00751  ratio 0.999
  aggregate paired ratio 0.998 CI[0.959,1.030]  (no significant transfer)
```
