# Decode Metadata Deforest Microbench

- device: `cuda`
- repeats: `30`
- all_equal: `True`
- sentinel policy: exact `-1` padding equality; no tolerated differences observed.

| BS | Max Seq | Old us | New us | Speedup | Equal |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 128 | 1054.14 | 153.99 | 6.85x | `True` |
| 1 | 4096 | 1073.91 | 149.13 | 7.20x | `True` |
| 1 | 5120 | 1059.43 | 147.97 | 7.16x | `True` |
| 2 | 128 | 1131.67 | 149.32 | 7.58x | `True` |
| 2 | 4096 | 1129.69 | 148.02 | 7.63x | `True` |
| 2 | 5120 | 1135.36 | 151.06 | 7.52x | `True` |
| 4 | 128 | 1249.14 | 148.12 | 8.43x | `True` |
| 4 | 4096 | 1253.58 | 147.82 | 8.48x | `True` |
| 4 | 5120 | 1246.39 | 148.99 | 8.37x | `True` |
