```text
C3 validation: mask x N/K x batch x SNIS/paper estimator
  invariant snis: translation/finite PASS
  invariant paper: translation/finite PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  equal   snis  N/K= 1 B= 16 mask=False: ED 0.1452 [0.1430,0.1474] stable-wrong-candidates=0/2
  equal   snis  N/K= 1 B= 16 mask=True : ED 1.5046 [1.1286,1.8805] stable-wrong-candidates=0/2
  equal   snis  N/K= 1 B= 64 mask=False: ED 0.3058 [0.2323,0.3794] stable-wrong-candidates=0/2
  equal   snis  N/K= 1 B= 64 mask=True : ED 1.2150 [0.9654,1.4646] stable-wrong-candidates=0/2
  equal   snis  N/K= 2 B= 16 mask=False: ED 0.1077 [0.0996,0.1157] stable-wrong-candidates=0/2
  equal   snis  N/K= 2 B= 16 mask=True : ED 0.2411 [0.2313,0.2508] stable-wrong-candidates=0/2
  equal   snis  N/K= 2 B= 64 mask=False: ED 0.1704 [0.1418,0.1991] stable-wrong-candidates=0/2
  equal   snis  N/K= 2 B= 64 mask=True : ED 0.3380 [0.3189,0.3571] stable-wrong-candidates=0/2
  equal   snis  N/K= 4 B= 16 mask=False: ED 0.0701 [0.0535,0.0868] stable-wrong-candidates=0/2
  equal   snis  N/K= 4 B= 16 mask=True : ED 0.1266 [0.1207,0.1326] stable-wrong-candidates=0/2
  equal   snis  N/K= 4 B= 64 mask=False: ED 0.0440 [0.0413,0.0468] stable-wrong-candidates=0/2
  equal   snis  N/K= 4 B= 64 mask=True : ED 0.1048 [0.1009,0.1088] stable-wrong-candidates=0/2
  equal   snis  N/K= 8 B= 16 mask=False: ED 0.0417 [0.0415,0.0418] stable-wrong-candidates=0/2
  equal   snis  N/K= 8 B= 16 mask=True : ED 0.0606 [0.0604,0.0609] stable-wrong-candidates=0/2
  equal   snis  N/K= 8 B= 64 mask=False: ED 0.0552 [0.0399,0.0705] stable-wrong-candidates=0/2
  equal   snis  N/K= 8 B= 64 mask=True : ED 0.0407 [0.0403,0.0412] stable-wrong-candidates=0/2
  equal   snis  N/K=16 B= 16 mask=False: ED 0.0188 [0.0170,0.0207] stable-wrong-candidates=0/2
  equal   snis  N/K=16 B= 16 mask=True : ED 0.0276 [0.0268,0.0283] stable-wrong-candidates=0/2
  equal   snis  N/K=16 B= 64 mask=False: ED 0.0353 [0.0283,0.0424] stable-wrong-candidates=0/2
  equal   snis  N/K=16 B= 64 mask=True : ED 0.0511 [0.0441,0.0580] stable-wrong-candidates=0/2
  equal   paper N/K= 1 B= 16 mask=False: ED 0.3249 [0.3077,0.3421] stable-wrong-candidates=0/2
  equal   paper N/K= 1 B= 16 mask=True : ED 0.4613 [0.4255,0.4972] stable-wrong-candidates=0/2
  equal   paper N/K= 1 B= 64 mask=False: ED 0.2522 [0.2164,0.2881] stable-wrong-candidates=0/2
  equal   paper N/K= 1 B= 64 mask=True : ED 0.3833 [0.3686,0.3980] stable-wrong-candidates=0/2
  equal   paper N/K= 2 B= 16 mask=False: ED 0.3501 [0.3270,0.3732] stable-wrong-candidates=0/2
  equal   paper N/K= 2 B= 16 mask=True : ED 0.3034 [0.2915,0.3154] stable-wrong-candidates=0/2
  equal   paper N/K= 2 B= 64 mask=False: ED 0.2798 [0.2433,0.3163] stable-wrong-candidates=0/2
  equal   paper N/K= 2 B= 64 mask=True : ED 0.3079 [0.3024,0.3133] stable-wrong-candidates=0/2
  equal   paper N/K= 4 B= 16 mask=False: ED 0.1996 [0.1798,0.2193] stable-wrong-candidates=0/2
  equal   paper N/K= 4 B= 16 mask=True : ED 0.2013 [0.1844,0.2181] stable-wrong-candidates=0/2
  equal   paper N/K= 4 B= 64 mask=False: ED 0.1483 [0.1408,0.1557] stable-wrong-candidates=0/2
  equal   paper N/K= 4 B= 64 mask=True : ED 0.1742 [0.1728,0.1755] stable-wrong-candidates=0/2
  equal   paper N/K= 8 B= 16 mask=False: ED 0.1341 [0.1334,0.1348] stable-wrong-candidates=0/2
  equal   paper N/K= 8 B= 16 mask=True : ED 0.1287 [0.1272,0.1301] stable-wrong-candidates=0/2
  equal   paper N/K= 8 B= 64 mask=False: ED 0.0966 [0.0956,0.0976] stable-wrong-candidates=0/2
  equal   paper N/K= 8 B= 64 mask=True : ED 0.1050 [0.1035,0.1065] stable-wrong-candidates=0/2
  equal   paper N/K=16 B= 16 mask=False: ED 0.1028 [0.0924,0.1131] stable-wrong-candidates=0/2
  equal   paper N/K=16 B= 16 mask=True : ED 0.1012 [0.0911,0.1113] stable-wrong-candidates=0/2
  equal   paper N/K=16 B= 64 mask=False: ED 0.1445 [0.1316,0.1574] stable-wrong-candidates=0/2
  equal   paper N/K=16 B= 64 mask=True : ED 0.1450 [0.1327,0.1572] stable-wrong-candidates=0/2
  unequal snis  N/K= 1 B= 16 mask=False: ED 0.1734 [0.1569,0.1898] stable-wrong-candidates=0/2
  unequal snis  N/K= 1 B= 16 mask=True : ED 1.5421 [1.2951,1.7890] stable-wrong-candidates=0/2
  unequal snis  N/K= 1 B= 64 mask=False: ED 0.1272 [0.1188,0.1356] stable-wrong-candidates=0/2
  unequal snis  N/K= 1 B= 64 mask=True : ED 0.7718 [0.7444,0.7992] stable-wrong-candidates=0/2
  unequal snis  N/K= 2 B= 16 mask=False: ED 0.0739 [0.0633,0.0845] stable-wrong-candidates=0/2
  unequal snis  N/K= 2 B= 16 mask=True : ED 0.2096 [0.1883,0.2310] stable-wrong-candidates=0/2
  unequal snis  N/K= 2 B= 64 mask=False: ED 0.0589 [0.0477,0.0701] stable-wrong-candidates=0/2
  unequal snis  N/K= 2 B= 64 mask=True : ED 0.4973 [0.3942,0.6004] stable-wrong-candidates=0/2
  unequal snis  N/K= 4 B= 16 mask=False: ED 0.0486 [0.0386,0.0585] stable-wrong-candidates=0/2
  unequal snis  N/K= 4 B= 16 mask=True : ED 0.1079 [0.0953,0.1205] stable-wrong-candidates=0/2
  unequal snis  N/K= 4 B= 64 mask=False: ED 0.0634 [0.0580,0.0688] stable-wrong-candidates=0/2
  unequal snis  N/K= 4 B= 64 mask=True : ED 0.1030 [0.0958,0.1101] stable-wrong-candidates=0/2
  unequal snis  N/K= 8 B= 16 mask=False: ED 0.0366 [0.0348,0.0384] stable-wrong-candidates=0/2
  unequal snis  N/K= 8 B= 16 mask=True : ED 0.0584 [0.0582,0.0585] stable-wrong-candidates=0/2
  unequal snis  N/K= 8 B= 64 mask=False: ED 0.0751 [0.0733,0.0770] stable-wrong-candidates=0/2
  unequal snis  N/K= 8 B= 64 mask=True : ED 0.0941 [0.0912,0.0970] stable-wrong-candidates=0/2
  unequal snis  N/K=16 B= 16 mask=False: ED 0.0349 [0.0313,0.0386] stable-wrong-candidates=0/2
  unequal snis  N/K=16 B= 16 mask=True : ED 0.0484 [0.0438,0.0530] stable-wrong-candidates=0/2
  unequal snis  N/K=16 B= 64 mask=False: ED 0.0258 [0.0218,0.0298] stable-wrong-candidates=0/2
  unequal snis  N/K=16 B= 64 mask=True : ED 0.0391 [0.0344,0.0438] stable-wrong-candidates=0/2
  unequal paper N/K= 1 B= 16 mask=False: ED 0.3827 [0.2899,0.4754] stable-wrong-candidates=0/2
  unequal paper N/K= 1 B= 16 mask=True : ED 0.5086 [0.4413,0.5759] stable-wrong-candidates=0/2
  unequal paper N/K= 1 B= 64 mask=False: ED 0.1697 [0.1500,0.1895] stable-wrong-candidates=0/2
  unequal paper N/K= 1 B= 64 mask=True : ED 0.3782 [0.3169,0.4395] stable-wrong-candidates=0/2
  unequal paper N/K= 2 B= 16 mask=False: ED 0.1484 [0.1338,0.1630] stable-wrong-candidates=0/2
  unequal paper N/K= 2 B= 16 mask=True : ED 0.1592 [0.1456,0.1729] stable-wrong-candidates=0/2
  unequal paper N/K= 2 B= 64 mask=False: ED 0.1267 [0.0846,0.1687] stable-wrong-candidates=0/2
  unequal paper N/K= 2 B= 64 mask=True : ED 0.2133 [0.1917,0.2348] stable-wrong-candidates=0/2
  unequal paper N/K= 4 B= 16 mask=False: ED 0.1954 [0.1783,0.2125] stable-wrong-candidates=0/2
  unequal paper N/K= 4 B= 16 mask=True : ED 0.1779 [0.1700,0.1858] stable-wrong-candidates=0/2
  unequal paper N/K= 4 B= 64 mask=False: ED 0.0890 [0.0716,0.1063] stable-wrong-candidates=0/2
  unequal paper N/K= 4 B= 64 mask=True : ED 0.1391 [0.1269,0.1513] stable-wrong-candidates=0/2
  unequal paper N/K= 8 B= 16 mask=False: ED 0.1756 [0.1638,0.1874] stable-wrong-candidates=0/2
  unequal paper N/K= 8 B= 16 mask=True : ED 0.1742 [0.1614,0.1870] stable-wrong-candidates=0/2
  unequal paper N/K= 8 B= 64 mask=False: ED 0.1642 [0.1594,0.1690] stable-wrong-candidates=0/2
  unequal paper N/K= 8 B= 64 mask=True : ED 0.1611 [0.1532,0.1691] stable-wrong-candidates=0/2
  unequal paper N/K=16 B= 16 mask=False: ED 0.1370 [0.1306,0.1435] stable-wrong-candidates=0/2
  unequal paper N/K=16 B= 16 mask=True : ED 0.1306 [0.1233,0.1379] stable-wrong-candidates=0/2
  unequal paper N/K=16 B= 64 mask=False: ED 0.0952 [0.0829,0.1076] stable-wrong-candidates=0/2
  unequal paper N/K=16 B= 64 mask=True : ED 0.0951 [0.0824,0.1077] stable-wrong-candidates=0/2
```
