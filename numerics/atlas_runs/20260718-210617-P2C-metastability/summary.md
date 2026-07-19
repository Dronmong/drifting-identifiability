```
P2C: scale-consistent metastability; censoring-aware fit;
     L/tau control test; eta control; two-bandwidth rescue
  L=5.0: exp-law slope c=1.692 (censored cells: 0); AIC pure-exp=-32.3 vs poly*exp=-39.4 (poly coeff -6.27)
  L=8.0: exp-law slope c=1.698 (censored cells: 0); AIC pure-exp=-32.8 vs poly*exp=-39.7 (poly coeff -6.15)
  L/tau=3.0 median T/tau -> L=5.0: 6.1; L=8.0: 3.8  (agreement = ratio controls)
  L/tau=4.0 median T/tau -> L=5.0: 22.3; L=8.0: 14.1  (agreement = ratio controls)
  eta=0.1*tau at L/tau=4: T/tau=45.9
  eta=0.05*tau at L/tau=4: T/tau=45.6
  fixed-abs sigma=0.3 at L/tau=4: T/tau=46.5 (compare scale-consistent cell)
  two-bandwidth intervention V = (V_tau + V_tau')/2:
    L/tau=4.0: single T/tau=45.9  mix(tau'=2.5) T/tau=10.5  speedup x4.4
    L/tau=4.0: single T/tau=45.9  mix(tau'=5.0) T/tau=8.2  speedup x5.6
    L/tau=5.0: single T/tau=679.5  mix(tau'=2.5) T/tau=21.5  speedup x31.6
    L/tau=5.0: single T/tau=679.5  mix(tau'=5.0) T/tau=17.7  speedup x38.4
```
