| mode | pass | mean TTFT s | mean output tok/s | decode prepare s | saved prefill | graph replay/eager |
| --- | --- | --- | --- | --- | --- | --- |
| prefix_off | 12/12 | 1.0818 | 50.6558 | 1.4710 | 0 | 679/0 |
| phase1_prefix_on | 12/12 | 0.6981 | 64.8296 | 1.4436 | 183040 | 679/0 |
| route_b_graph_baseline | 12/12 | 0.7909 | 52.5981 | 7.3987 | 165376 | 679/0 |
| route_b_metadata_deforest | 12/12 | 0.7733 | 47.4662 | 13.2645 | 165376 | 679/0 |
