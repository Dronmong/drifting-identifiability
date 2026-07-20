```text
C1 validation: bandwidth-only factorial + joint schedule
  invariant snis: translation/finite PASS
  invariant paper: translation/finite PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  target K=4 d=1: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    held-out oracle grid: best tau=0.5641; selection cost=33,177,600 kernel pairs
    K=4 d=1 fine-common             : ED 0.2555 [0.2515,0.2589] massErr=1.000 KMsteps=NR censored=8/8
    K=4 d=1 coarse-common           : ED 0.0498 [0.0459,0.0579] massErr=0.375 KMsteps=NR censored=8/8
    K=4 d=1 average-common          : ED 0.2397 [0.2319,0.2547] massErr=1.000 KMsteps=NR censored=8/8
    K=4 d=1 paper-fixed-common      : ED 0.3277 [0.3164,0.3469] massErr=1.000 KMsteps=NR censored=8/8
    K=4 d=1 anneal-oracle-common    : ED 0.1668 [0.1577,0.1823] massErr=0.844 KMsteps=NR censored=8/8
    K=4 d=1 anneal-estimated-common : ED 0.1666 [0.1576,0.1821] massErr=0.844 KMsteps=NR censored=8/8
    K=4 d=1 oracle-grid-common      : ED 0.0414 [0.0348,0.0482] massErr=0.438 KMsteps=NR censored=8/8
    K=4 d=1 fine-joint              : ED 0.2743 [0.2696,0.2813] massErr=1.000 KMsteps=NR censored=8/8
    K=4 d=1 coarse-joint            : ED 0.0058 [0.0054,0.0065] massErr=0.094 KMsteps=67 censored=0/8
    K=4 d=1 anneal-oracle-joint     : ED 0.0058 [0.0049,0.0078] massErr=0.125 KMsteps=76 censored=0/8
    K=4 d=1 anneal-estimated-joint  : ED 0.0058 [0.0049,0.0078] massErr=0.125 KMsteps=76 censored=0/8
  target K=4 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.999,0.150)
    held-out oracle grid: best tau=0.3459; selection cost=33,177,600 kernel pairs
    K=4 d=2 fine-common             : ED 0.0339 [0.0296,0.0556] massErr=0.281 KMsteps=NR censored=6/8
    K=4 d=2 coarse-common           : ED 0.0521 [0.0497,0.0562] massErr=0.125 KMsteps=NR censored=8/8
    K=4 d=2 average-common          : ED 0.0659 [0.0612,0.0761] massErr=0.219 KMsteps=NR censored=8/8
    K=4 d=2 paper-fixed-common      : ED 0.0848 [0.0768,0.0960] massErr=0.281 KMsteps=NR censored=8/8
    K=4 d=2 anneal-oracle-common    : ED 0.0157 [0.0116,0.0182] massErr=0.062 KMsteps=387 censored=3/8
    K=4 d=2 anneal-estimated-common : ED 0.0157 [0.0116,0.0181] massErr=0.062 KMsteps=386 censored=3/8
    K=4 d=2 oracle-grid-common      : ED 0.0106 [0.0095,0.0124] massErr=0.062 KMsteps=325 censored=0/8
    K=4 d=2 fine-joint              : ED 0.0428 [0.0384,0.0594] massErr=0.281 KMsteps=NR censored=8/8
    K=4 d=2 coarse-joint            : ED 0.0069 [0.0063,0.0075] massErr=0.062 KMsteps=131 censored=0/8
    K=4 d=2 anneal-oracle-joint     : ED 0.0047 [0.0037,0.0057] massErr=0.062 KMsteps=101 censored=0/8
    K=4 d=2 anneal-estimated-joint  : ED 0.0047 [0.0037,0.0057] massErr=0.062 KMsteps=101 censored=0/8
  target K=8 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.987,0.150)
    held-out oracle grid: best tau=0.3459; selection cost=88,473,600 kernel pairs
    K=8 d=2 fine-common             : ED 0.0307 [0.0244,0.0448] massErr=0.250 KMsteps=NR censored=8/8
    K=8 d=2 coarse-common           : ED 0.0533 [0.0497,0.0587] massErr=0.172 KMsteps=NR censored=8/8
    K=8 d=2 average-common          : ED 0.1030 [0.1008,0.1093] massErr=0.219 KMsteps=NR censored=8/8
    K=8 d=2 paper-fixed-common      : ED 0.1347 [0.1300,0.1419] massErr=0.234 KMsteps=NR censored=8/8
    K=8 d=2 anneal-oracle-common    : ED 0.0186 [0.0173,0.0212] massErr=0.188 KMsteps=NR censored=7/8
    K=8 d=2 anneal-estimated-common : ED 0.0186 [0.0173,0.0212] massErr=0.188 KMsteps=NR censored=7/8
    K=8 d=2 oracle-grid-common      : ED 0.0086 [0.0080,0.0119] massErr=0.109 KMsteps=399 censored=4/8
    K=8 d=2 fine-joint              : ED 0.0425 [0.0381,0.0566] massErr=0.234 KMsteps=NR censored=8/8
    K=8 d=2 coarse-joint            : ED 0.0052 [0.0051,0.0055] massErr=0.031 KMsteps=112 censored=0/8
    K=8 d=2 anneal-oracle-joint     : ED 0.0041 [0.0035,0.0050] massErr=0.094 KMsteps=110 censored=1/8
    K=8 d=2 anneal-estimated-joint  : ED 0.0041 [0.0034,0.0050] massErr=0.094 KMsteps=110 censored=1/8
  target K=4 d=5: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    held-out oracle grid: best tau=0.3459; selection cost=33,177,600 kernel pairs
    K=4 d=5 fine-common             : ED 0.0379 [0.0327,0.0406] massErr=0.219 KMsteps=NR censored=8/8
    K=4 d=5 coarse-common           : ED 0.0365 [0.0313,0.0392] massErr=0.188 KMsteps=NR censored=8/8
    K=4 d=5 average-common          : ED 0.0485 [0.0443,0.0517] massErr=0.188 KMsteps=NR censored=8/8
    K=4 d=5 paper-fixed-common      : ED 0.0540 [0.0495,0.0582] massErr=0.188 KMsteps=NR censored=8/8
    K=4 d=5 anneal-oracle-common    : ED 0.0316 [0.0286,0.0325] massErr=0.156 KMsteps=NR censored=8/8
    K=4 d=5 anneal-estimated-common : ED 0.0316 [0.0286,0.0324] massErr=0.156 KMsteps=NR censored=8/8
    K=4 d=5 oracle-grid-common      : ED 0.0176 [0.0157,0.0184] massErr=0.125 KMsteps=400 censored=4/8
    K=4 d=5 fine-joint              : ED 0.0408 [0.0352,0.0432] massErr=0.219 KMsteps=NR censored=8/8
    K=4 d=5 coarse-joint            : ED 0.0088 [0.0084,0.0088] massErr=0.062 KMsteps=95 censored=0/8
    K=4 d=5 anneal-oracle-joint     : ED 0.0194 [0.0187,0.0203] massErr=0.094 KMsteps=102 censored=0/8
    K=4 d=5 anneal-estimated-joint  : ED 0.0194 [0.0187,0.0203] massErr=0.094 KMsteps=102 censored=0/8
```
