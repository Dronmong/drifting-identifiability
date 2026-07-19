```
C1: bandwidth ladder vs single-best vs paper-multi
target: K Gaussians, sep L, width 0.15L; N=20K particles, batch=64, 800 steps, eta=0.1*tau_fine
  --- K=4 d=1 L=5.0: single-best tau=1.500 (sigma=0.750) ---
    single-fine : ED=0.1755 cover=1.00 massErr=0.338 steps->tol=30
    single-L    : ED=0.0293 cover=1.00 massErr=0.025 steps->tol=10
    paper-multi : ED=0.1127 cover=1.00 massErr=0.262 steps->tol=30
    ladder      : ED=0.0572 cover=1.00 massErr=0.200 steps->tol=20
    single-best : ED=0.0064 cover=1.00 massErr=0.013 steps->tol=20
  --- K=4 d=2 L=5.0: single-best tau=1.500 (sigma=0.750) ---
    single-fine : ED=0.2681 cover=1.00 massErr=0.300 steps->tol=5039
    single-L    : ED=0.0210 cover=1.00 massErr=0.063 steps->tol=20
    paper-multi : ED=0.2262 cover=1.00 massErr=0.287 steps->tol=5049
    ladder      : ED=0.1968 cover=1.00 massErr=0.287 steps->tol=5039
    single-best : ED=0.0106 cover=1.00 massErr=0.050 steps->tol=20
  --- K=8 d=2 L=5.0: single-best tau=1.500 (sigma=0.750) ---
    single-fine : ED=0.1041 cover=1.00 massErr=0.200 steps->tol=60
    single-L    : ED=0.0212 cover=1.00 massErr=0.200 steps->tol=20
    paper-multi : ED=0.0752 cover=1.00 massErr=0.188 steps->tol=70
    ladder      : ED=0.0459 cover=1.00 massErr=0.312 steps->tol=50
    single-best : ED=0.0095 cover=1.00 massErr=0.050 steps->tol=20
  --- K=4 d=5 L=5.0: single-best tau=5.000 (sigma=0.750) ---
    single-fine : ED=0.1067 cover=1.00 massErr=0.188 steps->tol=60
    single-L    : ED=0.0232 cover=1.00 massErr=0.025 steps->tol=20
    paper-multi : ED=0.0934 cover=1.00 massErr=0.163 steps->tol=80
    ladder      : ED=0.0782 cover=1.00 massErr=0.125 steps->tol=60
    single-best : ED=0.0232 cover=1.00 massErr=0.025 steps->tol=20
```
