# Release Lifetime A/B Matrix

| Case | Status | Text sanity | Released GiB/rank | Notes | Token prefixes |
| --- | --- | --- | ---: | --- | --- |
| force-prepacked raw-present | pass/pass | pass/pass/pass | 0.0000 | raw attrs present; force prepacked path | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| keep-hidden-ref | pass/pass | pass/pass/pass | 17.1328 | attrs removed; original tensors hidden-ref alive | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| release-after-capture | pass/pass | pass/pass/pass | 17.1328 | release after KV allocation and graph capture | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| weights-only | warn/warn | fail/pass/pass | 16.1250 | release w13/w2 only | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| scales-only | pass/pass | pass/pass/pass | 1.0078 | release scale tensors only | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| layer0 | pass/pass | pass/pass/pass | 0.3984 | partial release layer 0 | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| layers0-7 | pass/pass | pass/pass/pass | 3.1875 | partial release layers 0-7 | 20 940 223 20 223 15120 223 22 320<br>671 12709 344 8295 377 260 4521 2173 16<br>24463 389 21614 883 3740 21241 3415 122961 320 |
| layers0-15 | warn/warn | fail/pass/pass | 6.3750 | partial release layers 0-15 | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| layers0-20 | warn/warn | fail/pass/pass | 8.3672 | partial release layers 0-20 | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
| layers21-42 | warn/warn | fail/pass/pass | 8.7656 | partial release layers 21-42 | 20 940 223 0 0 0 0 0 0 0 0 0 0 0 0 0<br>671 12709 344 8295 0 0 0 0 0 0 0 0 0 0 0 0<br>24463 389 21614 0 0 0 0 0 0 0 0 0 0 0 0 0 |
