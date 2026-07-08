# MTP Debug Harnesses

Reusable DeepSeek V4 MTP correctness harnesses live here.

## Scripts

- `run_matrix.py`: runs baseline or MTP speculative decode matrices and records
  token ids, stats, optional state traces, and text outputs.
- `analyze_state_parity.py`: compares baseline and MTP matrix artifacts,
  including accepted-commit stats and state/KV bisection.  It includes the
  C128-aware planner rule that treats baseline legacy C128 raw locs and MTP
  online C128 chunk/bank0 locs as different coordinates for the same logical
  state when the trace proves that mapping.

## Output Policy

Do not write large results into this directory.  Use:

```text
performance_milestones/<target>/raw/
```

or `/tmp` for large transient files.

## Typical Usage

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  debug/mtp/run_matrix.py \
  --mode baseline \
  --output performance_milestones/<target>/raw/baseline_matrix.json \
  --page-size 256 --num-pages 16 --decode-len 8 --draft-len 2 \
  --max-running-req 4 --batch-sizes 1 2 4 5 6
```

```bash
python debug/mtp/analyze_state_parity.py \
  --baseline performance_milestones/<target>/raw/baseline_matrix.json \
  --mtp performance_milestones/<target>/raw/mtp_matrix.json \
  --output performance_milestones/<target>/raw/analysis.json \
  --batch-size 6
```
