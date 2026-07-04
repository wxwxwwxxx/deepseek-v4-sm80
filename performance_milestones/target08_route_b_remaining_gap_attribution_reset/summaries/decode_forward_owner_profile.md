# Decode Forward Owner Profile

CUDA owner-timing labels grouped by forward compute/communication owner. Prepare-side metadata/replay labels are excluded here and summarized in `prepare_owner_profile.md`. Values are max-rank ms.

| mode | decode forward s | attention ms | indexer/compressor ms | MoE/shared ms | communication ms | other owner ms |
| --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 15.0479 | 636.7226 | 119.4024 | 347.2494 | 4909.5034 | 0.0000 |
| Route B graph baseline | 14.6251 | 1142.9150 | 112.9800 | 1126.5050 | 5948.2426 | 0.0000 |
| Route B direct C4 | 14.8697 | 655.7333 | 112.5127 | 2122.0638 | 6263.7858 | 0.0000 |
| Route B direct SWA+C4+C128 | 14.4944 | 1490.7817 | 114.8328 | 1070.4746 | 6607.6030 | 0.0000 |
