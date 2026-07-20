```text
C1 validation: bandwidth-only factorial + joint schedule
  invariant snis: translation/finite PASS
  invariant paper: translation/finite PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  target K=4 d=1: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    K=4 d=1 fine-common             : ED 0.0140 [0.0135,0.0146] massErr=0.250 KMsteps=1 censored=0/2
    K=4 d=1 coarse-common           : ED 0.0126 [0.0119,0.0133] massErr=0.125 KMsteps=1 censored=0/2
    K=4 d=1 average-common          : ED 0.0119 [0.0113,0.0124] massErr=0.188 KMsteps=1 censored=0/2
    K=4 d=1 paper-fixed-common      : ED 0.0209 [0.0205,0.0214] massErr=0.250 KMsteps=1 censored=0/2
    K=4 d=1 anneal-oracle-common    : ED 0.0052 [0.0048,0.0057] massErr=0.156 KMsteps=1 censored=0/2
    K=4 d=1 anneal-estimated-common : ED 0.0052 [0.0048,0.0057] massErr=0.156 KMsteps=1 censored=0/2
    K=4 d=1 oracle-grid-common      : ED 0.0118 [0.0111,0.0124] massErr=0.125 KMsteps=1 censored=0/2
    K=4 d=1 fine-joint              : ED 0.0268 [0.0254,0.0282] massErr=0.250 KMsteps=1 censored=0/2
    K=4 d=1 coarse-joint            : ED 0.0088 [0.0087,0.0089] massErr=0.094 KMsteps=1 censored=0/2
    K=4 d=1 anneal-oracle-joint     : ED 0.0097 [0.0095,0.0100] massErr=0.156 KMsteps=1 censored=0/2
    K=4 d=1 anneal-estimated-joint  : ED 0.0097 [0.0095,0.0099] massErr=0.156 KMsteps=1 censored=0/2
  target K=4 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.999,0.150)
    K=4 d=2 fine-common             : ED 0.1004 [0.0927,0.1081] massErr=0.469 KMsteps=NR censored=2/2
    K=4 d=2 coarse-common           : ED 0.0265 [0.0247,0.0282] massErr=0.125 KMsteps=43 censored=0/2
    K=4 d=2 average-common          : ED 0.0598 [0.0571,0.0625] massErr=0.406 KMsteps=39 censored=1/2
    K=4 d=2 paper-fixed-common      : ED 0.1345 [0.1263,0.1426] massErr=0.469 KMsteps=NR censored=2/2
    K=4 d=2 anneal-oracle-common    : ED 0.0331 [0.0311,0.0352] massErr=0.281 KMsteps=31 censored=0/2
    K=4 d=2 anneal-estimated-common : ED 0.0331 [0.0311,0.0352] massErr=0.281 KMsteps=31 censored=0/2
    K=4 d=2 oracle-grid-common      : ED 0.0162 [0.0147,0.0177] massErr=0.219 KMsteps=33 censored=0/2
    K=4 d=2 fine-joint              : ED 0.2002 [0.1955,0.2050] massErr=0.469 KMsteps=NR censored=2/2
    K=4 d=2 coarse-joint            : ED 0.0175 [0.0161,0.0188] massErr=0.094 KMsteps=21 censored=0/2
    K=4 d=2 anneal-oracle-joint     : ED 0.0331 [0.0310,0.0353] massErr=0.281 KMsteps=27 censored=0/2
    K=4 d=2 anneal-estimated-joint  : ED 0.0331 [0.0310,0.0353] massErr=0.281 KMsteps=27 censored=0/2
  target K=8 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.987,0.150)
    K=8 d=2 fine-common             : ED 0.0216 [0.0198,0.0235] massErr=0.281 KMsteps=1 censored=0/2
    K=8 d=2 coarse-common           : ED 0.0251 [0.0244,0.0257] massErr=0.156 KMsteps=1 censored=0/2
    K=8 d=2 average-common          : ED 0.0223 [0.0222,0.0223] massErr=0.234 KMsteps=1 censored=0/2
    K=8 d=2 paper-fixed-common      : ED 0.0242 [0.0238,0.0245] massErr=0.266 KMsteps=1 censored=0/2
    K=8 d=2 anneal-oracle-common    : ED 0.0147 [0.0134,0.0160] massErr=0.219 KMsteps=1 censored=0/2
    K=8 d=2 anneal-estimated-common : ED 0.0147 [0.0134,0.0160] massErr=0.219 KMsteps=1 censored=0/2
    K=8 d=2 oracle-grid-common      : ED 0.0121 [0.0114,0.0128] massErr=0.109 KMsteps=1 censored=0/2
    K=8 d=2 fine-joint              : ED 0.0345 [0.0332,0.0357] massErr=0.281 KMsteps=1 censored=0/2
    K=8 d=2 coarse-joint            : ED 0.0176 [0.0175,0.0176] massErr=0.094 KMsteps=1 censored=0/2
    K=8 d=2 anneal-oracle-joint     : ED 0.0176 [0.0171,0.0180] massErr=0.234 KMsteps=1 censored=0/2
    K=8 d=2 anneal-estimated-joint  : ED 0.0176 [0.0172,0.0181] massErr=0.234 KMsteps=1 censored=0/2
  target K=4 d=5: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    K=4 d=5 fine-common             : ED 0.0219 [0.0196,0.0242] massErr=0.125 KMsteps=32 censored=0/2
    K=4 d=5 coarse-common           : ED 0.0533 [0.0531,0.0536] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 average-common          : ED 0.0493 [0.0487,0.0498] massErr=0.125 KMsteps=40 censored=1/2
    K=4 d=5 paper-fixed-common      : ED 0.0691 [0.0670,0.0711] massErr=0.125 KMsteps=NR censored=2/2
    K=4 d=5 anneal-oracle-common    : ED 0.0200 [0.0189,0.0210] massErr=0.125 KMsteps=33 censored=0/2
    K=4 d=5 anneal-estimated-common : ED 0.0199 [0.0189,0.0210] massErr=0.125 KMsteps=33 censored=0/2
    K=4 d=5 oracle-grid-common      : ED 0.0130 [0.0128,0.0132] massErr=0.125 KMsteps=30 censored=0/2
    K=4 d=5 fine-joint              : ED 0.2078 [0.2051,0.2105] massErr=0.125 KMsteps=NR censored=2/2
    K=4 d=5 coarse-joint            : ED 0.0226 [0.0224,0.0229] massErr=0.094 KMsteps=42 censored=0/2
    K=4 d=5 anneal-oracle-joint     : ED 0.0251 [0.0251,0.0252] massErr=0.125 KMsteps=37 censored=0/2
    K=4 d=5 anneal-estimated-joint  : ED 0.0251 [0.0250,0.0252] massErr=0.125 KMsteps=37 censored=0/2
```
