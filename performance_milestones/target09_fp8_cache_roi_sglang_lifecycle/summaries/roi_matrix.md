| row | persistent GiB/rank | graph headroom delta | equiv current pages/tokens | expected latency delta | quality/correctness risk | implementation scope |
| --- | --- | --- | --- | --- | --- | --- |
| current mini BF16 + additive indexer FP8 side | 2.302 | +0.000 GiB | 0.0 pages / 0 tokens | baseline | low; runtime-proven promoted path | none |
| current mini + SWA-only FP8 | 1.726 | +0.576 GiB | 32.0 pages / 8199 tokens | +0.016 ms/boundary if separated; needs fusion | medium quality/latency; correctness slice passed | replace SWA cache only; keep C4/C128/indexer/state |
| current mini + full source-aligned MLA/indexer FP8 | 1.614 | +0.688 GiB | 38.3 pages / 9794 tokens | unknown; likely worse until fused/integrated | high; C4/C128/indexer/prefix ownership not integrated | SWA+C4+C128 MLA FP8 and indexer replacement |
| SGLang lifecycle + BF16 | 1.127 | +1.176 GiB | 65.4 pages / 16734 tokens | near baseline or slight metadata cost | medium correctness; lifecycle not runtime-proven in mini | independent SWA pool, 16 tail pages, BF16 cache dtype |
| SGLang lifecycle + SWA-only FP8 | 1.055 | +1.248 GiB | 69.4 pages / 17759 tokens | estimated +0.006-0.012 ms/boundary if fused store + selected gather | medium-high; combines lifecycle and FP8 | lifecycle plus FP8 SWA tail pool (16 pages) |
| SGLang lifecycle + broader MLA/indexer FP8 | 0.942 | +1.360 GiB | 75.6 pages / 19354 tokens | unknown; attention-integrated dequant may be needed | highest; broad source layout plus ownership rewrite | lifecycle plus SWA/C4/C128/indexer FP8 replacement |
