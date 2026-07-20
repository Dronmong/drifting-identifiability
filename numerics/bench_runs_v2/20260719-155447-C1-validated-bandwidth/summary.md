```text
C1 validation: bandwidth-only factorial + joint schedule
  invariant snis: translation/finite PASS
  invariant paper: translation/finite PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  target K=4 d=1: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    held-out oracle grid: best tau=0.5641; selection cost=2,211,840 kernel pairs
    K=4 d=1 fine-common             : ED 0.2436 [0.2420,0.2451] massErr=1.000 KMsteps=NR censored=2/2
    K=4 d=1 coarse-common           : ED 0.0420 [0.0401,0.0439] massErr=0.344 KMsteps=NR censored=2/2
    K=4 d=1 average-common          : ED 0.2276 [0.2237,0.2314] massErr=1.000 KMsteps=NR censored=2/2
    K=4 d=1 paper-fixed-common      : ED 0.3181 [0.3145,0.3217] massErr=1.000 KMsteps=NR censored=2/2
    K=4 d=1 anneal-oracle-common    : ED 0.1662 [0.1564,0.1759] massErr=0.875 KMsteps=NR censored=2/2
    K=4 d=1 anneal-estimated-common : ED 0.1660 [0.1563,0.1758] massErr=0.875 KMsteps=NR censored=2/2
    K=4 d=1 oracle-grid-common      : ED 0.0339 [0.0326,0.0352] massErr=0.375 KMsteps=NR censored=2/2
    K=4 d=1 fine-joint              : ED 0.3645 [0.3609,0.3680] massErr=1.000 KMsteps=NR censored=2/2
    K=4 d=1 coarse-joint            : ED 0.0138 [0.0133,0.0144] massErr=0.156 KMsteps=67 censored=0/2
    K=4 d=1 anneal-oracle-joint     : ED 0.1733 [0.1656,0.1810] massErr=0.875 KMsteps=NR censored=2/2
    K=4 d=1 anneal-estimated-joint  : ED 0.1733 [0.1657,0.1810] massErr=0.875 KMsteps=NR censored=2/2
  target K=4 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.999,0.150)
    held-out oracle grid: best tau=0.2121; selection cost=2,211,840 kernel pairs
    K=4 d=2 fine-common             : ED 0.0854 [0.0772,0.0936] massErr=0.438 KMsteps=NR censored=2/2
    K=4 d=2 coarse-common           : ED 0.0496 [0.0446,0.0546] massErr=0.156 KMsteps=NR censored=2/2
    K=4 d=2 average-common          : ED 0.0721 [0.0672,0.0769] massErr=0.344 KMsteps=NR censored=2/2
    K=4 d=2 paper-fixed-common      : ED 0.0976 [0.0892,0.1060] massErr=0.438 KMsteps=NR censored=2/2
    K=4 d=2 anneal-oracle-common    : ED 0.0274 [0.0258,0.0290] massErr=0.250 KMsteps=NR censored=2/2
    K=4 d=2 anneal-estimated-common : ED 0.0274 [0.0258,0.0290] massErr=0.250 KMsteps=NR censored=2/2
    K=4 d=2 oracle-grid-common      : ED 0.0212 [0.0197,0.0227] massErr=0.156 KMsteps=61 censored=1/2
    K=4 d=2 fine-joint              : ED 0.1274 [0.1181,0.1367] massErr=0.469 KMsteps=NR censored=2/2
    K=4 d=2 coarse-joint            : ED 0.0255 [0.0232,0.0279] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=2 anneal-oracle-joint     : ED 0.0504 [0.0463,0.0545] massErr=0.250 KMsteps=NR censored=2/2
    K=4 d=2 anneal-estimated-joint  : ED 0.0504 [0.0463,0.0545] massErr=0.250 KMsteps=NR censored=2/2
  target K=8 d=2: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.987,0.150)
    held-out oracle grid: best tau=0.2121; selection cost=5,898,240 kernel pairs
    K=8 d=2 fine-common             : ED 0.0318 [0.0303,0.0333] massErr=0.328 KMsteps=NR censored=2/2
    K=8 d=2 coarse-common           : ED 0.0549 [0.0522,0.0576] massErr=0.219 KMsteps=NR censored=2/2
    K=8 d=2 average-common          : ED 0.1017 [0.0988,0.1047] massErr=0.328 KMsteps=NR censored=2/2
    K=8 d=2 paper-fixed-common      : ED 0.1322 [0.1282,0.1362] massErr=0.328 KMsteps=NR censored=2/2
    K=8 d=2 anneal-oracle-common    : ED 0.0227 [0.0217,0.0237] massErr=0.234 KMsteps=NR censored=2/2
    K=8 d=2 anneal-estimated-common : ED 0.0226 [0.0216,0.0236] massErr=0.234 KMsteps=NR censored=2/2
    K=8 d=2 oracle-grid-common      : ED 0.0125 [0.0117,0.0133] massErr=0.141 KMsteps=76 censored=1/2
    K=8 d=2 fine-joint              : ED 0.2738 [0.2685,0.2790] massErr=0.312 KMsteps=NR censored=2/2
    K=8 d=2 coarse-joint            : ED 0.0231 [0.0228,0.0235] massErr=0.172 KMsteps=NR censored=2/2
    K=8 d=2 anneal-oracle-joint     : ED 0.0788 [0.0744,0.0833] massErr=0.250 KMsteps=NR censored=2/2
    K=8 d=2 anneal-estimated-joint  : ED 0.0795 [0.0750,0.0840] massErr=0.250 KMsteps=NR censored=2/2
  target K=4 d=5: minSep=1.000 oracle(L,sigma)=(1.000,0.150) estimated=(0.997,0.151)
    held-out oracle grid: best tau=0.2121; selection cost=2,211,840 kernel pairs
    K=4 d=5 fine-common             : ED 0.0335 [0.0323,0.0348] massErr=0.188 KMsteps=NR censored=2/2
    K=4 d=5 coarse-common           : ED 0.0426 [0.0409,0.0442] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 average-common          : ED 0.0531 [0.0505,0.0557] massErr=0.125 KMsteps=NR censored=2/2
    K=4 d=5 paper-fixed-common      : ED 0.0570 [0.0544,0.0596] massErr=0.125 KMsteps=NR censored=2/2
    K=4 d=5 anneal-oracle-common    : ED 0.0312 [0.0298,0.0326] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 anneal-estimated-common : ED 0.0312 [0.0298,0.0325] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 oracle-grid-common      : ED 0.0177 [0.0173,0.0182] massErr=0.125 KMsteps=68 censored=1/2
    K=4 d=5 fine-joint              : ED 0.0753 [0.0708,0.0799] massErr=0.125 KMsteps=NR censored=2/2
    K=4 d=5 coarse-joint            : ED 0.0254 [0.0243,0.0265] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 anneal-oracle-joint     : ED 0.0494 [0.0474,0.0514] massErr=0.094 KMsteps=NR censored=2/2
    K=4 d=5 anneal-estimated-joint  : ED 0.0494 [0.0474,0.0514] massErr=0.094 KMsteps=NR censored=2/2
```
