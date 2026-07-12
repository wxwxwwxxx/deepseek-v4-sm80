from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch


def _load_batches(root: Path, rank: int) -> list[dict[str, Any]]:
    path = root / f"batches.rank{rank}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing prefix debug batch log: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(rows, key=lambda row: int(row["batch_index"]))


def _load_logits(root: Path, row: dict[str, Any]) -> torch.Tensor | None:
    rel = row.get("logits_path")
    if not rel:
        return None
    payload = torch.load(root / rel, map_location="cpu")
    return payload["logits"].float()


def _topk(logits: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()), dim=-1)
    return [
        {"token_id": int(token), "prob": float(prob)}
        for token, prob in zip(indices.tolist(), values.tolist())
    ]


def _step_label(uid: int, occurrence: int, phase: str) -> str:
    if occurrence == 0 and phase == "prefill":
        return f"uid{uid}:prefill_last"
    return f"uid{uid}:decode_step_{occurrence}"


def build_parity(args: argparse.Namespace) -> dict[str, Any]:
    pre_rows = _load_batches(args.prebuild_debug_dir, args.rank)
    rel_rows = _load_batches(args.release_debug_dir, args.rank)
    pre_by_index = {int(row["batch_index"]): row for row in pre_rows}
    rel_by_index = {int(row["batch_index"]): row for row in rel_rows}
    common = sorted(set(pre_by_index).intersection(rel_by_index))
    occurrences: dict[int, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    for batch_index in common:
        pre = pre_by_index[batch_index]
        rel = rel_by_index[batch_index]
        pre_logits = _load_logits(args.prebuild_debug_dir, pre)
        rel_logits = _load_logits(args.release_debug_dir, rel)
        pre_tokens = list(pre.get("sampled_token_ids", []))
        rel_tokens = list(rel.get("sampled_token_ids", []))
        reqs = pre.get("reqs", [])
        row_count = min(len(reqs), len(pre_tokens), len(rel_tokens))
        if pre_logits is not None and rel_logits is not None:
            row_count = min(row_count, pre_logits.shape[0], rel_logits.shape[0])
        for row_idx in range(row_count):
            req = reqs[row_idx]
            uid = int(req["uid"])
            occurrence = occurrences[uid]
            occurrences[uid] += 1
            phase = str(pre.get("phase", ""))
            label = _step_label(uid, occurrence, phase)
            record: dict[str, Any] = {
                "label": label,
                "batch_index": batch_index,
                "row": row_idx,
                "uid": uid,
                "phase": phase,
                "prebuild_forward_source": pre.get("forward_source"),
                "release_forward_source": rel.get("forward_source"),
                "prebuild_token": int(pre_tokens[row_idx]),
                "release_token": int(rel_tokens[row_idx]),
                "token_diverged": int(pre_tokens[row_idx]) != int(rel_tokens[row_idx]),
            }
            if pre_logits is not None and rel_logits is not None:
                delta = (pre_logits[row_idx] - rel_logits[row_idx]).abs()
                record.update(
                    {
                        "max_abs_diff": float(delta.max().item()),
                        "mean_abs_diff": float(delta.mean().item()),
                        "prebuild_topk": _topk(pre_logits[row_idx], args.topk),
                        "release_topk": _topk(rel_logits[row_idx], args.topk),
                    }
                )
            else:
                record.update(
                    {
                        "max_abs_diff": None,
                        "mean_abs_diff": None,
                        "prebuild_topk": [],
                        "release_topk": [],
                    }
                )
            if first_divergence is None and record["token_diverged"]:
                first_divergence = record
            records.append(record)
    return {
        "rank": args.rank,
        "prebuild_debug_dir": str(args.prebuild_debug_dir),
        "release_debug_dir": str(args.release_debug_dir),
        "topk": args.topk,
        "records": records,
        "first_token_divergence": first_divergence,
    }


def _fmt_topk(items: list[dict[str, Any]]) -> str:
    return ", ".join(f"{item['token_id']}:{item['prob']:.4g}" for item in items)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Logit Parity Ladder",
        "",
        f"Rank: `{payload['rank']}`",
        "",
    ]
    first = payload.get("first_token_divergence")
    if first is None:
        lines.append("First token divergence: none in recorded steps.")
    else:
        lines.append(
            "First token divergence: "
            f"`{first['label']}` prebuild=`{first['prebuild_token']}` "
            f"release=`{first['release_token']}`."
        )
    lines.extend(
        [
            "",
            "| Step | Phase | Sources | Tokens pre/release | Max abs diff | Mean abs diff | Prebuild top-k prob | Release top-k prob |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for record in payload["records"]:
        max_abs = record["max_abs_diff"]
        mean_abs = record["mean_abs_diff"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record['label']}`",
                    str(record["phase"]),
                    f"{record['prebuild_forward_source']}/{record['release_forward_source']}",
                    f"{record['prebuild_token']}/{record['release_token']}",
                    "n/a" if max_abs is None else f"{max_abs:.6g}",
                    "n/a" if mean_abs is None else f"{mean_abs:.6g}",
                    _fmt_topk(record["prebuild_topk"]),
                    _fmt_topk(record["release_topk"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuild-debug-dir", type=Path, required=True)
    parser.add_argument("--release-debug-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_parity(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)


if __name__ == "__main__":
    main()
