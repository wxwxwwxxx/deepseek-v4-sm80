# TARGET 08.06 CUDA Graph Memory Attribution Summary

## Runs

| run | kind | buckets | max_seq_len | pages | greedy | metadata in graph | captured | delta GiB rank0 | delta GiB mean | capture s | pool reuse | replay/eager |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on` | bucket_set | `[1, 2, 4, 8, 16]` | 2048 | 128 | on | on | `[16, 8, 4, 2, 1]` | 19.04 | 19.04 | 17.12 | yes | 63/0 |
| `bucketset_1_2_4_8_np128_sl2048_greedy_on_metadata_on` | bucket_set | `[1, 2, 4, 8]` | 2048 | 128 | on | on | `[8, 4, 2, 1]` | 18.96 | 18.96 | 15.02 | yes | 48/15 |
| `bucketset_1_2_4_np128_sl2048_greedy_on_metadata_on` | bucket_set | `[1, 2, 4]` | 2048 | 128 | on | on | `[4, 2, 1]` | 18.90 | 18.90 | 14.69 | yes | 40/23 |
| `greedy_off_np128_sl2048_metadata_on` | greedy_ab | `[1, 2, 4, 8, 16]` | 2048 | 128 | off | on | `[16, 8, 4, 2, 1]` | 19.04 | 19.04 | 15.00 | yes | 63/0 |
| `metadata_off_np128_sl2048_greedy_on` | metadata_ab | `[1, 2, 4, 8, 16]` | 2048 | 128 | on | off | `[16, 8, 4, 2, 1]` | 19.04 | 19.04 | 14.78 | yes | 63/0 |
| `seq1280_np128_greedy_on_metadata_on` | seq_pages | `[1, 2, 4, 8, 16]` | 1280 | 128 | on | on | `[16, 8, 4, 2, 1]` | 19.04 | 19.04 | 15.33 | yes | 63/0 |
| `seq1280_np64_greedy_on_metadata_on` | seq_pages | `[1, 2, 4, 8, 16]` | 1280 | 64 | on | on | `[16, 8, 4, 2, 1]` | 19.00 | 19.00 | 16.99 | yes | 63/0 |
| `seq2048_np64_greedy_on_metadata_on` | seq_pages | `[1, 2, 4, 8, 16]` | 2048 | 64 | on | on | `[16, 8, 4, 2, 1]` | 19.00 | 19.00 | 17.36 | yes | 63/0 |
| `seq5120_np128_greedy_on_metadata_on` | seq_pages | `[1, 2, 4, 8, 16]` | 5120 | 128 | on | on | `[16, 8, 4, 2, 1]` | 19.04 | 19.04 | 15.31 | yes | 63/0 |
| `seq5120_np64_greedy_on_metadata_on` | seq_pages | `[1, 2, 4, 8, 16]` | 5120 | 64 | on | on | `[16, 8, 4, 2, 1]` | 19.00 | 19.00 | 17.27 | yes | 63/0 |
| `single_16_np128_sl2048_greedy_on_metadata_on` | single_bucket | `[16]` | 2048 | 128 | on | on | `[16]` | 18.83 | 18.83 | 13.57 | single/none | 63/0 |
| `single_1_np128_sl2048_greedy_on_metadata_on` | single_bucket | `[1]` | 2048 | 128 | on | on | `[1]` | 18.79 | 18.79 | 11.95 | single/none | 16/47 |
| `single_4_np128_sl2048_greedy_on_metadata_on` | single_bucket | `[4]` | 2048 | 128 | on | on | `[4]` | 18.80 | 18.80 | 11.50 | single/none | 40/23 |
| `single_8_np128_sl2048_greedy_on_metadata_on` | single_bucket | `[8]` | 2048 | 128 | on | on | `[8]` | 18.81 | 18.81 | 13.77 | single/none | 48/15 |

## Per-Bucket Free-Memory Delta

| run | per-bucket delta GiB |
| --- | --- |
| `bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on` | bs16: 18.83<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `bucketset_1_2_4_8_np128_sl2048_greedy_on_metadata_on` | bs8: 18.81<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `bucketset_1_2_4_np128_sl2048_greedy_on_metadata_on` | bs4: 18.80<br>bs2: 0.05<br>bs1: 0.05 |
| `greedy_off_np128_sl2048_metadata_on` | bs16: 18.83<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `metadata_off_np128_sl2048_greedy_on` | bs16: 18.83<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `seq1280_np128_greedy_on_metadata_on` | bs16: 18.83<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `seq1280_np64_greedy_on_metadata_on` | bs16: 18.79<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `seq2048_np64_greedy_on_metadata_on` | bs16: 18.79<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `seq5120_np128_greedy_on_metadata_on` | bs16: 18.83<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `seq5120_np64_greedy_on_metadata_on` | bs16: 18.79<br>bs8: 0.06<br>bs4: 0.06<br>bs2: 0.05<br>bs1: 0.05 |
| `single_16_np128_sl2048_greedy_on_metadata_on` | bs16: 18.83 |
| `single_1_np128_sl2048_greedy_on_metadata_on` | bs1: 18.79 |
| `single_4_np128_sl2048_greedy_on_metadata_on` | bs4: 18.80 |
| `single_8_np128_sl2048_greedy_on_metadata_on` | bs8: 18.81 |
