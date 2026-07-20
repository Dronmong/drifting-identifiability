```text
C3 validation: mask x N/K x batch x SNIS/paper estimator
  invariant snis: translation/finite PASS
  invariant paper: translation/finite PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  equal   snis  N/K= 1 B= 16 mask=False: ED 0.1553 [0.1399,0.1633] stable-wrong-candidates=0/6
  equal   snis  N/K= 1 B= 16 mask=True : ED 3.1443 [2.7990,3.5253] stable-wrong-candidates=0/6
  equal   snis  N/K= 1 B= 64 mask=False: ED 0.1816 [0.1420,0.3570] stable-wrong-candidates=0/6
  equal   snis  N/K= 1 B= 64 mask=True : ED 3.8850 [3.5896,9.6866] stable-wrong-candidates=0/6
  equal   snis  N/K= 2 B= 16 mask=False: ED 0.1160 [0.0995,0.1400] stable-wrong-candidates=0/6
  equal   snis  N/K= 2 B= 16 mask=True : ED 0.3023 [0.2666,0.3337] stable-wrong-candidates=0/6
  equal   snis  N/K= 2 B= 64 mask=False: ED 0.1291 [0.0609,0.1665] stable-wrong-candidates=0/6
  equal   snis  N/K= 2 B= 64 mask=True : ED 0.3548 [0.2909,0.3990] stable-wrong-candidates=0/6
  equal   snis  N/K= 4 B= 16 mask=False: ED 0.0540 [0.0256,0.0961] stable-wrong-candidates=0/6
  equal   snis  N/K= 4 B= 16 mask=True : ED 0.0597 [0.0558,0.0618] stable-wrong-candidates=0/6
  equal   snis  N/K= 4 B= 64 mask=False: ED 0.0151 [0.0123,0.0288] stable-wrong-candidates=0/6
  equal   snis  N/K= 4 B= 64 mask=True : ED 0.0661 [0.0639,0.0679] stable-wrong-candidates=0/6
  equal   snis  N/K= 8 B= 16 mask=False: ED 0.0312 [0.0166,0.0384] stable-wrong-candidates=0/6
  equal   snis  N/K= 8 B= 16 mask=True : ED 0.0187 [0.0136,0.0227] stable-wrong-candidates=0/6
  equal   snis  N/K= 8 B= 64 mask=False: ED 0.0152 [0.0128,0.0335] stable-wrong-candidates=0/6
  equal   snis  N/K= 8 B= 64 mask=True : ED 0.0189 [0.0181,0.0211] stable-wrong-candidates=0/6
  equal   snis  N/K=16 B= 16 mask=False: ED 0.0060 [0.0048,0.0075] stable-wrong-candidates=0/6
  equal   snis  N/K=16 B= 16 mask=True : ED 0.0050 [0.0037,0.0058] stable-wrong-candidates=0/6
  equal   snis  N/K=16 B= 64 mask=False: ED 0.0083 [0.0055,0.0110] stable-wrong-candidates=0/6
  equal   snis  N/K=16 B= 64 mask=True : ED 0.0081 [0.0065,0.0097] stable-wrong-candidates=0/6
  equal   paper N/K= 1 B= 16 mask=False: ED 0.2141 [0.0702,0.2395] stable-wrong-candidates=0/6
  equal   paper N/K= 1 B= 16 mask=True : ED 0.4947 [0.4412,0.5821] stable-wrong-candidates=0/6
  equal   paper N/K= 1 B= 64 mask=False: ED 0.2112 [0.1200,0.2740] stable-wrong-candidates=0/6
  equal   paper N/K= 1 B= 64 mask=True : ED 0.4415 [0.4284,0.4715] stable-wrong-candidates=0/6
  equal   paper N/K= 2 B= 16 mask=False: ED 0.1643 [0.1579,0.1945] stable-wrong-candidates=0/6
  equal   paper N/K= 2 B= 16 mask=True : ED 0.3128 [0.2741,0.3188] stable-wrong-candidates=0/6
  equal   paper N/K= 2 B= 64 mask=False: ED 0.1262 [0.0992,0.1396] stable-wrong-candidates=0/6
  equal   paper N/K= 2 B= 64 mask=True : ED 0.1985 [0.1447,0.2545] stable-wrong-candidates=0/6
  equal   paper N/K= 4 B= 16 mask=False: ED 0.1610 [0.1398,0.1763] stable-wrong-candidates=0/6
  equal   paper N/K= 4 B= 16 mask=True : ED 0.1723 [0.1578,0.1901] stable-wrong-candidates=0/6
  equal   paper N/K= 4 B= 64 mask=False: ED 0.0872 [0.0577,0.1080] stable-wrong-candidates=0/6
  equal   paper N/K= 4 B= 64 mask=True : ED 0.1399 [0.0991,0.1697] stable-wrong-candidates=0/6
  equal   paper N/K= 8 B= 16 mask=False: ED 0.1223 [0.1095,0.1539] stable-wrong-candidates=0/6
  equal   paper N/K= 8 B= 16 mask=True : ED 0.1234 [0.1072,0.1501] stable-wrong-candidates=0/6
  equal   paper N/K= 8 B= 64 mask=False: ED 0.0520 [0.0427,0.0596] stable-wrong-candidates=0/6
  equal   paper N/K= 8 B= 64 mask=True : ED 0.0562 [0.0502,0.0653] stable-wrong-candidates=0/6
  equal   paper N/K=16 B= 16 mask=False: ED 0.1060 [0.0981,0.1312] stable-wrong-candidates=0/6
  equal   paper N/K=16 B= 16 mask=True : ED 0.1021 [0.0960,0.1248] stable-wrong-candidates=0/6
  equal   paper N/K=16 B= 64 mask=False: ED 0.0914 [0.0829,0.1119] stable-wrong-candidates=0/6
  equal   paper N/K=16 B= 64 mask=True : ED 0.0937 [0.0721,0.1069] stable-wrong-candidates=0/6
  unequal snis  N/K= 1 B= 16 mask=False: ED 0.1502 [0.0700,0.1777] stable-wrong-candidates=0/6
  unequal snis  N/K= 1 B= 16 mask=True : ED 3.1896 [2.6346,3.6498] stable-wrong-candidates=0/6
  unequal snis  N/K= 1 B= 64 mask=False: ED 0.1374 [0.1116,0.4836] stable-wrong-candidates=0/6
  unequal snis  N/K= 1 B= 64 mask=True : ED 3.3706 [3.2625,3.6224] stable-wrong-candidates=0/6
  unequal snis  N/K= 2 B= 16 mask=False: ED 0.0840 [0.0527,0.1048] stable-wrong-candidates=0/6
  unequal snis  N/K= 2 B= 16 mask=True : ED 0.3130 [0.3029,0.3281] stable-wrong-candidates=0/6
  unequal snis  N/K= 2 B= 64 mask=False: ED 0.0775 [0.0408,0.0906] stable-wrong-candidates=0/6
  unequal snis  N/K= 2 B= 64 mask=True : ED 0.4474 [0.3924,0.5096] stable-wrong-candidates=0/6
  unequal snis  N/K= 4 B= 16 mask=False: ED 0.0285 [0.0222,0.0410] stable-wrong-candidates=0/6
  unequal snis  N/K= 4 B= 16 mask=True : ED 0.0655 [0.0617,0.0670] stable-wrong-candidates=0/6
  unequal snis  N/K= 4 B= 64 mask=False: ED 0.0154 [0.0107,0.0355] stable-wrong-candidates=0/6
  unequal snis  N/K= 4 B= 64 mask=True : ED 0.0953 [0.0918,0.1033] stable-wrong-candidates=0/6
  unequal snis  N/K= 8 B= 16 mask=False: ED 0.0137 [0.0111,0.0247] stable-wrong-candidates=0/6
  unequal snis  N/K= 8 B= 16 mask=True : ED 0.0147 [0.0123,0.0175] stable-wrong-candidates=0/6
  unequal snis  N/K= 8 B= 64 mask=False: ED 0.0265 [0.0105,0.0364] stable-wrong-candidates=0/6
  unequal snis  N/K= 8 B= 64 mask=True : ED 0.0324 [0.0297,0.0370] stable-wrong-candidates=0/6
  unequal snis  N/K=16 B= 16 mask=False: ED 0.0092 [0.0084,0.0128] stable-wrong-candidates=0/6
  unequal snis  N/K=16 B= 16 mask=True : ED 0.0066 [0.0049,0.0101] stable-wrong-candidates=0/6
  unequal snis  N/K=16 B= 64 mask=False: ED 0.0111 [0.0101,0.0124] stable-wrong-candidates=0/6
  unequal snis  N/K=16 B= 64 mask=True : ED 0.0137 [0.0128,0.0171] stable-wrong-candidates=0/6
  unequal paper N/K= 1 B= 16 mask=False: ED 0.2459 [0.1447,0.3323] stable-wrong-candidates=0/6
  unequal paper N/K= 1 B= 16 mask=True : ED 0.4862 [0.4525,0.4963] stable-wrong-candidates=0/6
  unequal paper N/K= 1 B= 64 mask=False: ED 0.1763 [0.0903,0.3077] stable-wrong-candidates=0/6
  unequal paper N/K= 1 B= 64 mask=True : ED 0.5376 [0.4396,0.6760] stable-wrong-candidates=0/6
  unequal paper N/K= 2 B= 16 mask=False: ED 0.0901 [0.0621,0.1725] stable-wrong-candidates=0/6
  unequal paper N/K= 2 B= 16 mask=True : ED 0.2235 [0.1514,0.2762] stable-wrong-candidates=0/6
  unequal paper N/K= 2 B= 64 mask=False: ED 0.1877 [0.1377,0.2673] stable-wrong-candidates=0/6
  unequal paper N/K= 2 B= 64 mask=True : ED 0.3106 [0.2845,0.4146] stable-wrong-candidates=0/6
  unequal paper N/K= 4 B= 16 mask=False: ED 0.1741 [0.1244,0.2647] stable-wrong-candidates=0/6
  unequal paper N/K= 4 B= 16 mask=True : ED 0.1712 [0.1474,0.2700] stable-wrong-candidates=0/6
  unequal paper N/K= 4 B= 64 mask=False: ED 0.0817 [0.0481,0.0991] stable-wrong-candidates=0/6
  unequal paper N/K= 4 B= 64 mask=True : ED 0.1239 [0.1180,0.1456] stable-wrong-candidates=0/6
  unequal paper N/K= 8 B= 16 mask=False: ED 0.1298 [0.1124,0.1457] stable-wrong-candidates=0/6
  unequal paper N/K= 8 B= 16 mask=True : ED 0.1344 [0.1107,0.1682] stable-wrong-candidates=0/6
  unequal paper N/K= 8 B= 64 mask=False: ED 0.0934 [0.0537,0.1408] stable-wrong-candidates=0/6
  unequal paper N/K= 8 B= 64 mask=True : ED 0.1251 [0.1081,0.1330] stable-wrong-candidates=0/6
  unequal paper N/K=16 B= 16 mask=False: ED 0.1214 [0.1029,0.1553] stable-wrong-candidates=0/6
  unequal paper N/K=16 B= 16 mask=True : ED 0.1288 [0.1114,0.1532] stable-wrong-candidates=0/6
  unequal paper N/K=16 B= 64 mask=False: ED 0.0977 [0.0728,0.1332] stable-wrong-candidates=0/6
  unequal paper N/K=16 B= 64 mask=True : ED 0.0800 [0.0660,0.1209] stable-wrong-candidates=0/6
```
