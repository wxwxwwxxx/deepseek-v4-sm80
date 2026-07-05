| scenario | bound | current retained pages | tail SWA pages | tail SWA tokens | BF16 lifecycle GiB | remaining SWA-only FP8 value GiB | assumption |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_1024_bs4 | low | 124 | 4 | 1024 | 1.001 | 0.018 | bs4 long prompts; one page-aligned SWA tail per active/retained branch |
| historical_4096_1024_bs4 | high | 124 | 4 | 1024 | 1.001 | 0.018 | bs4 long prompts; one page-aligned SWA tail per active/retained branch |
| serving_mixed_112req_wave16 | low | 106 | 16 | 4096 | 1.127 | 0.072 | wave16 active-tail lower bound; short no-share branch retention upper bound |
| serving_mixed_112req_wave16 | high | 106 | 106 | 27136 | 2.071 | 0.477 | wave16 active-tail lower bound; short no-share branch retention upper bound |
| prefix_multi_112req_wave16 | low | 87 | 8 | 2048 | 1.043 | 0.036 | eight shared 512-token prefixes lower bound; wave16 active tails upper bound |
| prefix_multi_112req_wave16 | high | 87 | 16 | 4096 | 1.127 | 0.072 | eight shared 512-token prefixes lower bound; wave16 active tails upper bound |
| serving_mixed_256req_wave64_est | low | estimated | 64 | 16384 | 1.631 | 0.288 | synthetic higher-concurrency wave64 serving estimate |
| serving_mixed_256req_wave64_est | high | estimated | 64 | 16384 | 1.631 | 0.288 | synthetic higher-concurrency wave64 serving estimate |
