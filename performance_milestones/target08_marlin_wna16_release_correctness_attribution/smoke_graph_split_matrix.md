# Smoke And Graph Split Matrix

| Run | Status | Graph | Replay/Eager/Greedy | Output sanity | Token prefixes |
| --- | --- | --- | --- | --- | --- |
| 08.35 baseline reuse | pass/pass | on | 9/0/9 | pass/pass/pass | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| 08.35 prebuild reuse | pass/pass | on | 9/0/9 | pass/pass/pass | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| 08.35 release reuse | warn/warn | on | 63/0/63 | fail/fail/fail | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| release graph mt1 | warn/warn | on | 0/0/0 | fail/fail/pass | 20<br>671<br>24463 |
| release graph mt2 | warn/warn | on | 1/0/1 | fail/fail/pass | 20 940<br>671 12709<br>24463 389 |
| release graph mt4 | warn/warn | on | 3/0/3 | fail/pass/pass | 20 940 223 0<br>671 12709 344 8295<br>24463 389 21614 0 |
| release graph mt16 | warn/warn | on | 15/0/15 | fail/pass/pass | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| release eager mt16 | warn/warn | off | 0/15/0 | fail/pass/pass | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 768 32349 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| release graph no-greedy mt6 | warn/warn | on | 5/0/0 | fail/pass/pass | 20 940 223 0 0 0<br>671 12709 344 8295 0 0<br>24463 389 21614 0 0 0 |
