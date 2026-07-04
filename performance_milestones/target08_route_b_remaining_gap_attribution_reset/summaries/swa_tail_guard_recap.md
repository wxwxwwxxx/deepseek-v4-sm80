# SWA Tail Guard Recap

08.26 did not reroute into SWA-tail work. The main `serving_mixed_112req_wave16` runs have `prefix_saved_prefill_tokens=0`, `prefix_evictions=0`, and identical decode graph replay/eager coverage (`441/0`) across phase1, Route B baseline, Route B direct C4, and full direct.

Prior TARGET 08.22/08.24 evidence already showed the SWA-tail exact-multiple guard as a small secondary effect, not the large Route B gap owner. In this reset, the dominant Route B direct C4 remaining gap is decode prepare metadata construction, especially component page tables, while SWA index construction is small in comparison.

Recommendation: do not return to TARGET 08.23 unless future prefix-hit or eviction scenarios show SWA retention/capacity loss. It is not the current `serving_mixed_112req_wave16` bottleneck.
