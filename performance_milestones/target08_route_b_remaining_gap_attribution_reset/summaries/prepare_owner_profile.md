# Decode Prepare Owner Profile

Owner timing profile runs only; do not use these rows as final throughput evidence.

| mode | decode prepare s | host attention metadata ms | component tables ms | full page table ms | SWA idx ms | C4 idx ms | C128 idx ms | write locs ms | replay fused copy ms | replay comp tables ms | direct index ms | build bytes | copy bytes | direct bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 prefix on | 1.9935 | 1814.1543 | 5.5495 | 70.0731 | 90.1665 | 584.1073 | 326.8966 | 180.2352 | 16.7356 | 0 | 0.0000 | 19976768 | 19976768 | 0 |
| Route B graph baseline | 4.7805 | 4591.5918 | 3350.0037 | 69.4883 | 84.5014 | 691.7250 | 420.1730 | 245.8265 | 17.6241 | 104.0867 | 0.0000 | 20039936 | 20039936 | 0 |
| Route B direct C4 | 4.4941 | 4304.6365 | 3341.1692 | 69.8436 | 83.8588 | 409.5431 | 421.9036 | 245.3735 | 17.3471 | 103.2981 | 71.1383 | 3557120 | 3524864 | 16515072 |
| Route B direct SWA+C4+C128 | 4.1048 | 3914.8209 | 3351.5761 | 69.5198 | 1.5669 | 423.0593 | 112.0447 | 248.2373 | 16.8173 | 104.6296 | 66.9185 | 159488 | 84224 | 19955712 |
