from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _files(root: Path, prefix: str) -> list[Path]:
    return sorted(root.glob(f"{prefix}*.jsonl"))


def _rank_from_row(row: dict[str, Any]) -> int:
    return int(row.get("rank", 0) or 0)


def _item_key(item: dict[str, Any]) -> tuple[int | None, str, int, int]:
    layer = item.get("layer_id")
    return (
        None if layer is None else int(layer),
        str(item.get("component") or item.get("attribute") or ""),
        int(item.get("start", item.get("data_ptr", 0)) or 0),
        int(item.get("bytes", 0) or 0),
    )


def _brief_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_id": item.get("layer_id"),
        "component": item.get("component") or item.get("attribute"),
        "bytes": int(item.get("bytes", 0) or 0),
        "start": int(item.get("start", item.get("data_ptr", 0)) or 0),
        "end": int(item.get("end", 0) or 0),
        "dtype": item.get("dtype"),
    }


def _load_freed_by_rank(root: Path) -> dict[int, list[dict[str, Any]]]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in _files(root, "marlin_wna16_freed_ranges_"):
        for row in _read_jsonl(path):
            if row.get("event") != "dsv4_marlin_wna16_freed_range":
                continue
            if not row.get("released", False):
                continue
            by_rank[_rank_from_row(row)].append(row)
    for rows in by_rank.values():
        rows.sort(key=lambda r: (float(r.get("time_s", 0.0)), int(r.get("start", 0) or 0)))
    return dict(by_rank)


def _load_guards_by_rank(root: Path) -> dict[int, list[dict[str, Any]]]:
    first_by_rank_index: dict[tuple[int, int], dict[str, Any]] = {}
    for path in _files(root, "marlin_wna16_release_guards_"):
        for row in _read_jsonl(path):
            if row.get("event") != "dsv4_marlin_wna16_release_guard_check":
                continue
            index = row.get("quarantine_index")
            if index is None:
                continue
            key = (_rank_from_row(row), int(index))
            previous = first_by_rank_index.get(key)
            if previous is None or float(row.get("time_s", 0.0)) < float(
                previous.get("time_s", 0.0)
            ):
                first_by_rank_index[key] = row
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (rank, _index), row in first_by_rank_index.items():
        by_rank[rank].append(row)
    for rows in by_rank.values():
        rows.sort(key=lambda r: int(r.get("quarantine_index", 0) or 0))
    return dict(by_rank)


def _load_owner_overlaps(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _files(root, "marlin_wna16_owner_ledger_"):
        for row in _read_jsonl(path):
            if row.get("event") != "dsv4_marlin_wna16_owner_allocation":
                continue
            if row.get("overlaps_freed_range", False):
                rows.append(row)
    return rows


def _summarize_guard_rank(
    *,
    freed: list[dict[str, Any]],
    guards: list[dict[str, Any]],
) -> dict[str, Any]:
    guarded_items = [
        row.get("source_released_item", {})
        for row in guards
        if isinstance(row.get("source_released_item"), dict)
    ]
    guard_keys = {_item_key(item) for item in guarded_items}
    release_order_guarded = [
        idx for idx, item in enumerate(freed) if _item_key(item) in guard_keys
    ]
    layer_components = Counter(
        (item.get("layer_id"), item.get("component") or item.get("attribute"))
        for item in guarded_items
    )
    layers = sorted({int(item["layer_id"]) for item in guarded_items if item.get("layer_id") is not None})
    bytes_by_layer = Counter()
    for item in guarded_items:
        if item.get("layer_id") is not None:
            bytes_by_layer[int(item["layer_id"])] += int(item.get("bytes", 0) or 0)
    return {
        "guard_count": len(guarded_items),
        "guard_bytes": int(sum(int(item.get("bytes", 0) or 0) for item in guarded_items)),
        "guard_layers": layers,
        "guard_release_order_indices": release_order_guarded,
        "guard_matches_first_n_release_items": release_order_guarded
        == list(range(len(release_order_guarded))),
        "layer_components": {
            f"layer{layer}.{component}": count
            for (layer, component), count in sorted(layer_components.items())
        },
        "bytes_by_layer": dict(sorted(bytes_by_layer.items())),
        "first_guarded_items": [_brief_item(item) for item in guarded_items[:8]],
    }


def _summarize_overlaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage = Counter(str(row.get("stage", "")) for row in rows)
    by_owner = Counter(str(row.get("owner", "")) for row in rows)
    by_freed = Counter()
    total_overlap_bytes = 0
    for row in rows:
        overlap = row.get("overlap_freed_range")
        if not isinstance(overlap, dict):
            continue
        by_freed[
            (
                overlap.get("layer_id"),
                overlap.get("component"),
            )
        ] += 1
        total_overlap_bytes += int(overlap.get("overlap_bytes", 0) or 0)
    return {
        "overlap_rows": len(rows),
        "sum_overlap_bytes_from_rows": int(total_overlap_bytes),
        "top_stages": by_stage.most_common(16),
        "top_owners": by_owner.most_common(24),
        "top_freed_items": [
            {"layer_id": layer, "component": component, "rows": count}
            for (layer, component), count in by_freed.most_common(24)
        ],
    }


def _markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Guard Range Census")
    lines.append("")
    lines.append("## Guarded Range Summary")
    lines.append("")
    lines.append("| Rank | Guard items | Guard GiB | Layers | Release-order prefix |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for rank, summary in sorted(result["guard_by_rank"].items(), key=lambda kv: int(kv[0])):
        layers = summary["guard_layers"]
        layer_text = f"{layers[0]}-{layers[-1]}" if layers else ""
        prefix = "yes" if summary["guard_matches_first_n_release_items"] else "no"
        lines.append(
            f"| {rank} | {summary['guard_count']} | "
            f"{summary['guard_bytes'] / float(1 << 30):.6f} | {layer_text} | {prefix} |"
        )
    lines.append("")
    lines.append("## Guarded Components")
    lines.append("")
    first_rank = next(iter(sorted(result["guard_by_rank"], key=int)), None)
    if first_rank is not None:
        summary = result["guard_by_rank"][first_rank]
        lines.append(f"Rank {first_rank} first guarded items:")
        lines.append("")
        lines.append("| Layer | Component | Bytes | Source start |")
        lines.append("| ---: | --- | ---: | ---: |")
        for item in summary["first_guarded_items"]:
            lines.append(
                f"| {item['layer_id']} | `{item['component']}` | "
                f"{item['bytes']} | {item['start']} |"
            )
    lines.append("")
    lines.append("## Owner Overlap Comparison")
    lines.append("")
    for label in ("unsafe_owner_overlaps", "guarded_owner_overlaps"):
        section = result.get(label, {})
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- rows: {section.get('overlap_rows', 0)}")
        lines.append(
            f"- summed overlap bytes from rows: {section.get('sum_overlap_bytes_from_rows', 0)}"
        )
        lines.append("- top owners:")
        for owner, count in section.get("top_owners", [])[:10]:
            lines.append(f"  - `{owner}`: {count}")
        lines.append("- top stages:")
        for stage, count in section.get("top_stages", [])[:8]:
            lines.append(f"  - `{stage}`: {count}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if result.get("all_ranks_guard_same_prefix", False):
        lines.append(
            "The 3.1875 GiB guard maps to the first 32 released ledger items on every "
            "rank in the analyzed run. Those items are layers 0-7 w13/w13_scale/w2/w2_scale."
        )
    else:
        lines.append(
            "The analyzed ranks do not all map the guard to the same release-order prefix."
        )
    lines.append(
        "This census proves release-order membership, but by itself does not prove that "
        "layers 0-7 are the semantic root."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guarded-dir", type=Path, required=True)
    parser.add_argument("--unsafe-dir", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    freed_by_rank = _load_freed_by_rank(args.guarded_dir)
    guards_by_rank = _load_guards_by_rank(args.guarded_dir)
    guard_by_rank: dict[str, Any] = {}
    for rank in sorted(set(freed_by_rank) | set(guards_by_rank)):
        guard_by_rank[str(rank)] = _summarize_guard_rank(
            freed=freed_by_rank.get(rank, []),
            guards=guards_by_rank.get(rank, []),
        )

    prefix_ok = bool(guard_by_rank) and all(
        summary["guard_matches_first_n_release_items"]
        and summary["guard_count"] == len(summary["guard_release_order_indices"])
        for summary in guard_by_rank.values()
    )
    result: dict[str, Any] = {
        "guarded_dir": str(args.guarded_dir),
        "unsafe_dir": str(args.unsafe_dir) if args.unsafe_dir else None,
        "guard_by_rank": guard_by_rank,
        "all_ranks_guard_same_prefix": prefix_ok,
        "guarded_owner_overlaps": _summarize_overlaps(_load_owner_overlaps(args.guarded_dir)),
    }
    if args.unsafe_dir:
        result["unsafe_owner_overlaps"] = _summarize_overlaps(_load_owner_overlaps(args.unsafe_dir))
    else:
        result["unsafe_owner_overlaps"] = {}

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
