# Nsight Kernel/Memcpy Summary

## serving_mixed_112req_wave16

### total

- kernels: 381082 / 5.122601s
- graph trace: 441 / 8.971415s
- memcpy: 216214 / 10.714902s / 196460160471 bytes

| NCCL kernel | count | duration s |
| --- | ---: | ---: |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 870 | 0.188856 |
| `ncclSymDevKernel_AllReduce_AGxLL_R_sum_bf16(ncclSymDevArgs)` | 174 | 0.007054 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 12 | 0.001053 |

| Memcpy kind | src | dst | count | bytes | duration s |
| --- | --- | --- | ---: | ---: | ---: |
| Host-to-Device | Pinned | Device | 69544 | 156016570844 | 10.391990 |
| Device-to-Device | Device | Device | 84132 | 40431332298 | 0.222126 |
| Host-to-Device | Pageable | Device | 45939 | 12036352 | 0.064920 |
| Device-to-Host | Device | Pageable | 455 | 150528 | 0.000968 |
| Device-to-Host | Device | Pinned | 16144 | 70449 | 0.034898 |

### repeat_window

- kernels: 177278 / 3.737685s
- graph trace: 441 / 8.971415s
- memcpy: 57834 / 0.133070s / 27386079063 bytes

| NCCL kernel | count | duration s |
| --- | ---: | ---: |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 609 | 0.169882 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 7 | 0.000509 |

| Memcpy kind | src | dst | count | bytes | duration s |
| --- | --- | --- | ---: | ---: | ---: |
| Device-to-Device | Device | Device | 38227 | 27384962570 | 0.094592 |
| Host-to-Device | Pinned | Device | 1932 | 872704 | 0.002159 |
| Device-to-Host | Device | Pageable | 455 | 150528 | 0.000968 |
| Device-to-Host | Device | Pinned | 15652 | 68173 | 0.033674 |
| Host-to-Device | Pageable | Device | 1568 | 25088 | 0.001677 |

## historical_4096_128_bs4

### total

- kernels: 262071 / 5.798571s
- graph trace: 127 / 2.544768s
- memcpy: 196984 / 8.234588s / 171422790511 bytes

| NCCL kernel | count | duration s |
| --- | ---: | ---: |
| `ncclSymDevKernel_AllReduce_RSxLD_AGxST_sum_bf16(ncclSymDevArgs)` | 261 | 0.153342 |
| `ncclSymDevKernel_AllReduce_AGxLL_R_sum_bf16(ncclSymDevArgs)` | 174 | 0.001708 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 87 | 0.105632 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 6 | 0.001259 |

| Memcpy kind | src | dst | count | bytes | duration s |
| --- | --- | --- | ---: | ---: | ---: |
| Host-to-Device | Pinned | Device | 68132 | 156016253100 | 7.956982 |
| Device-to-Device | Device | Device | 81630 | 15394379240 | 0.211645 |
| Host-to-Device | Pageable | Device | 44675 | 12015552 | 0.060246 |
| Device-to-Host | Device | Pageable | 129 | 133104 | 0.000278 |
| Device-to-Host | Device | Pinned | 2418 | 9515 | 0.005437 |

### repeat_window

- kernels: 58357 / 4.328672s
- graph trace: 127 / 2.544768s
- memcpy: 38634 / 0.090738s / 2348709343 bytes

| NCCL kernel | count | duration s |
| --- | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 87 | 0.105632 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | 1 | 0.000038 |

| Memcpy kind | src | dst | count | bytes | duration s |
| --- | --- | --- | ---: | ---: | ---: |
| Device-to-Device | Device | Device | 35725 | 2348009512 | 0.085242 |
| Host-to-Device | Pinned | Device | 520 | 554960 | 0.000604 |
| Device-to-Host | Device | Pageable | 129 | 133104 | 0.000278 |
| Device-to-Host | Device | Pinned | 1956 | 7479 | 0.004297 |
| Host-to-Device | Pageable | Device | 304 | 4288 | 0.000317 |

