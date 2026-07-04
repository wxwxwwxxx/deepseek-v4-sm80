from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"


PERF_REPORTS = {
    "prefix_off": RAW
    / "perf_prefix_off"
    / "reports"
    / "000_prefix_full_hit_257_bs4__dsv4_sm80_a100_victory.json",
    "phase1_prefix_on": RAW
    / "perf_phase1_prefix_on"
    / "reports"
    / "000_prefix_full_hit_257_bs4__dsv4_sm80_a100_victory.json",
    "route_b_graph": RAW
    / "perf_route_b_graph"
    / "reports"
    / "000_prefix_full_hit_257_bs4__dsv4_sm80_a100_victory.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _graph(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("graph_runner_case") or config.get("graph_runner") or {}


def _summarize_text_smoke(path: Path) -> dict[str, Any]:
    data = _load(path)
    variants = data.get("variants", [])
    variant = variants[0] if variants else {}
    cfg = variant.get("config", {})
    graph = _graph(cfg)
    sidecar_name = path.name.replace(".json", ".dsv4_sm80_a100_victory.json")
    sidecar = path.with_name(sidecar_name)
    outputs: list[str] = []
    if sidecar.exists():
        sidecar_data = _load(sidecar)
        for case in sidecar_data.get("cases", sidecar_data.get("outputs", [])):
            outputs.append(
                case.get("output") or case.get("generated_text") or case.get("text") or ""
            )
    return {
        "status": data.get("status"),
        "variant": variant.get("variant", {}).get("name"),
        "outputs": outputs,
        "captured_bs": graph.get("captured_bs"),
        "requested_bs": graph.get("requested_bs"),
        "replay_count": graph.get("replay_count"),
        "eager_decode_count": graph.get("eager_decode_count"),
        "capture_compressed_locs_in_graph": graph.get("capture_compressed_locs_in_graph"),
        "component_guarded_hook": graph.get(
            "capture_compressed_locs_in_graph_component_guarded"
        ),
        "error_type": (graph.get("error") or {}).get("exception_type"),
    }


def _summarize_perf(label: str, path: Path) -> dict[str, Any]:
    data = _load(path)
    metrics = data["metrics"]
    config = data["config"]
    graph = _graph(config)
    prefix = config.get("prefix_cache_metrics", {})
    component = prefix.get("dsv4_component_ownership") or {}
    return {
        "label": label,
        "status": data.get("status"),
        "elapsed_s": metrics.get("elapsed_s"),
        "ttft_s_mean": metrics.get("ttft_s_mean"),
        "topt_s_mean": metrics.get("topt_s_mean"),
        "prefill_tokens_per_s": metrics.get("prefill_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
        "e2e_output_tokens_per_s": metrics.get("end_to_end_output_tokens_per_s"),
        "e2e_total_tokens_per_s": metrics.get("end_to_end_total_tokens_per_s"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "actual_output_tokens": metrics.get("actual_output_tokens"),
        "graph_captured_bs": graph.get("captured_bs"),
        "graph_replay_count": graph.get("replay_count"),
        "eager_decode_count": graph.get("eager_decode_count"),
        "component_guarded_hook": graph.get(
            "capture_compressed_locs_in_graph_component_guarded"
        ),
        "prefix_match_requests": prefix.get("match_requests"),
        "prefix_hit_requests": prefix.get("hit_requests"),
        "saved_prefill_tokens": prefix.get("saved_prefill_tokens"),
        "retained_prefix_pages": prefix.get("retained_prefix_pages"),
        "live_full_pages": component.get("live_full_pages"),
        "live_c4_slots": component.get("live_c4_slots"),
        "live_c128_slots": component.get("live_c128_slots"),
        "live_c4_indexer_slots": component.get("live_c4_indexer_slots"),
        "live_c4_state_slots": component.get("live_c4_state_slots"),
        "live_c128_state_slots": component.get("live_c128_state_slots"),
        "live_c4_indexer_state_slots": component.get("live_c4_indexer_state_slots"),
    }


def _write_perf_tables(rows: list[dict[str, Any]]) -> None:
    keys = [
        "label",
        "status",
        "elapsed_s",
        "ttft_s_mean",
        "topt_s_mean",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "e2e_output_tokens_per_s",
        "e2e_total_tokens_per_s",
        "graph_replay_count",
        "eager_decode_count",
        "prefix_hit_requests",
        "saved_prefill_tokens",
        "retained_prefix_pages",
        "live_full_pages",
        "live_c4_slots",
        "live_c128_slots",
        "live_c4_indexer_slots",
        "live_c4_state_slots",
        "live_c128_state_slots",
        "live_c4_indexer_state_slots",
    ]
    csv_lines = [",".join(keys)]
    for row in rows:
        csv_lines.append(",".join("" if row.get(key) is None else str(row.get(key)) for key in keys))
    (SUMMARIES / "performance_ab.csv").write_text("\n".join(csv_lines) + "\n")

    md_lines = [
        "| mode | TTFT s | TPOT s | prefill tok/s | decode tok/s | output tok/s | graph replay/eager | hits/saved | live full | C4/C128/indexer | state C4/C128/indexer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {label} | {ttft_s_mean:.4f} | {topt_s_mean:.4f} | "
            "{prefill_tokens_per_s:.2f} | {decode_tokens_per_s:.2f} | "
            "{e2e_output_tokens_per_s:.2f} | {graph_replay_count}/{eager_decode_count} | "
            "{prefix_hit_requests}/{saved_prefill_tokens} | {live_full_pages} | "
            "{live_c4_slots}/{live_c128_slots}/{live_c4_indexer_slots} | "
            "{live_c4_state_slots}/{live_c128_state_slots}/{live_c4_indexer_state_slots} |".format(
                **{
                    key: (0 if row.get(key) is None and key != "label" else row.get(key))
                    for key in row
                }
            )
        )
    (SUMMARIES / "performance_ab.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    text = {
        "route_b_victory_graph": _summarize_text_smoke(
            RAW / "text_smoke_route_b_victory_graph.json"
        ),
        "route_b_fallback_graph_attempt": _summarize_text_smoke(
            RAW / "text_smoke_route_b_graph.json"
        ),
    }
    perf = [_summarize_perf(label, path) for label, path in PERF_REPORTS.items()]
    result = {"text_smoke": text, "performance_ab": perf}
    (SUMMARIES / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    _write_perf_tables(perf)
    print(json.dumps({"text_status": text["route_b_victory_graph"]["status"], "perf_rows": len(perf)}))


if __name__ == "__main__":
    main()
