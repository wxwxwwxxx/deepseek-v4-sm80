# all_gather Probe Summary

| backend | shape | first event max ms | warm event mean ms | warm event max ms | graph event mean ms | graph event max ms | correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| torch | `[16, 16160]` | 1343.938 | 0.322 | 2.354 | 0.139 | 0.350 | True |
| torch | `[8, 16160]` | 0.244 | 0.206 | 0.764 | 0.108 | 0.186 | True |
| torch | `[4, 16160]` | 0.166 | 0.162 | 0.597 | 0.110 | 0.586 | True |
| torch | `[2, 16160]` | 0.153 | 0.166 | 0.911 | 0.075 | 0.173 | True |
| torch | `[1, 16160]` | 0.452 | 0.151 | 0.475 | 0.072 | 0.194 | True |
| pynccl_threshold32m | `[16, 16160]` | 646.057 | 0.315 | 3.214 | 0.133 | 0.236 | True |
| pynccl_threshold32m | `[8, 16160]` | 0.268 | 0.130 | 0.191 | 0.113 | 0.263 | True |
| pynccl_threshold32m | `[4, 16160]` | 0.136 | 0.116 | 0.171 | 0.393 | 7.200 | True |
| pynccl_threshold32m | `[2, 16160]` | 0.149 | 0.111 | 0.177 | 0.072 | 0.136 | True |
| pynccl_threshold32m | `[1, 16160]` | 0.146 | 0.108 | 0.316 | 0.075 | 0.193 | True |
