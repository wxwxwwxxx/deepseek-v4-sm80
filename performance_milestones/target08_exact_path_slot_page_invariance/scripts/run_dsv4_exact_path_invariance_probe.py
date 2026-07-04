#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")
os.environ.setdefault("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE", "1")


@dataclass(frozen=True)
class PreludeBatch:
    name: str
    prompts: list[list[int]]
    labels: list[str]


@dataclass(frozen=True)
class InvarianceScenario:
    name: str
    description: str
    coverage: list[str]
    probe_prompts: list[list[int]]
    probe_labels: list[str]
    prelude_batches: list[PreludeBatch]


def _git_info() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
        except Exception as exc:
            return f"<error: {type(exc).__name__}: {exc}>"

    return {
        "rev": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}; "
            "launch with torchrun --standalone --nproc_per_node=8."
        )
    return tp_rank, tp_size, env_world_size


def _distributed_init_method(args: argparse.Namespace, tp_size: int) -> str | None:
    if args.distributed_init_method is not None:
        return args.distributed_init_method
    if tp_size > 1 and "MASTER_ADDR" in os.environ:
        return "env://"
    return None


def _dtype_from_name(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _random_tokens(
    rng: random.Random,
    length: int,
    *,
    vocab_size: int,
    token_id_range: int,
    marker: int,
) -> list[int]:
    low = 10 if vocab_size > 64 else 1
    high = min(max(low, int(token_id_range)), max(vocab_size - 1, low))
    tokens = [rng.randint(low, high) for _ in range(length)]
    if tokens:
        tokens[0] = min(high, low + marker)
    return tokens


def _prompt_digest(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(x)) for x in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _make_prompt_bank(
    *,
    vocab_size: int,
    token_id_range: int,
    seed: int,
    page_size: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {
        "target_257": _random_tokens(
            rng,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=11,
        ),
        "target_129": _random_tokens(
            rng,
            129,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=13,
        ),
        "dummy_257_a": _random_tokens(
            rng,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=31,
        ),
        "dummy_257_b": _random_tokens(
            rng,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=37,
        ),
        "dummy_513": _random_tokens(
            rng,
            page_size * 2 + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=41,
        ),
        "dummy_769": _random_tokens(
            rng,
            page_size * 3 + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=43,
        ),
    }


def _make_fillers(
    rng: random.Random,
    count: int,
    length: int,
    *,
    vocab_size: int,
    token_id_range: int,
    marker_base: int,
) -> list[list[int]]:
    return [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=marker_base + idx,
        )
        for idx in range(count)
    ]


def _scenario_identical_slots(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del rng, vocab_size, token_id_range, page_size
    target = bank["target_257"]
    return InvarianceScenario(
        name="identical_prompts_batch",
        description="Four identical 257-token prompts in one prefix-disabled batch.",
        coverage=[
            "prefix cache disabled identical prompts",
            "batch slot invariance",
            "page boundary 257",
        ],
        probe_prompts=[list(target) for _ in range(4)],
        probe_labels=[f"target_slot{i}" for i in range(4)],
        prelude_batches=[],
    )


def _scenario_single_target(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del rng, vocab_size, token_id_range, page_size
    return InvarianceScenario(
        name="single_target_alone",
        description="The 257-token target prompt run alone.",
        coverage=["single prompt alone baseline", "page boundary 257"],
        probe_prompts=[list(bank["target_257"])],
        probe_labels=["target"],
        prelude_batches=[],
    )


def _make_slot_scenario(slot: int) -> Callable[..., InvarianceScenario]:
    def build(
        bank: dict[str, list[int]],
        rng: random.Random,
        vocab_size: int,
        token_id_range: int,
        page_size: int,
    ) -> InvarianceScenario:
        fillers = _make_fillers(
            rng,
            3,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker_base=100 + slot * 10,
        )
        prompts = []
        labels = []
        filler_iter = iter(fillers)
        for idx in range(4):
            if idx == slot:
                prompts.append(list(bank["target_257"]))
                labels.append("target")
            else:
                prompts.append(next(filler_iter))
                labels.append(f"filler{idx}")
        return InvarianceScenario(
            name=f"target_in_batch_slot{slot}",
            description=f"The same 257-token target prompt placed in batch slot {slot}.",
            coverage=["one prompt alone vs slot 0/1/2/3", "batch slot invariance"],
            probe_prompts=prompts,
            probe_labels=labels,
            prelude_batches=[],
        )

    return build


def _make_table_row_scenario(dummy_count: int) -> Callable[..., InvarianceScenario]:
    def build(
        bank: dict[str, list[int]],
        rng: random.Random,
        vocab_size: int,
        token_id_range: int,
        page_size: int,
    ) -> InvarianceScenario:
        prompts = _make_fillers(
            rng,
            dummy_count,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker_base=200 + dummy_count * 10,
        )
        preludes = []
        if prompts:
            preludes.append(
                PreludeBatch(
                    name=f"table_row_churn_{dummy_count}",
                    prompts=prompts,
                    labels=[f"dummy{i}" for i in range(dummy_count)],
                )
            )
        return InvarianceScenario(
            name=f"target_table_row_after_{dummy_count}_dummy",
            description=(
                "The 257-token target prompt run alone after a completed dummy batch "
                f"of size {dummy_count}, perturbing the request-table free-list."
            ),
            coverage=["same prompt with different request-table rows"],
            probe_prompts=[list(bank["target_257"])],
            probe_labels=["target"],
            prelude_batches=preludes,
        )

    return build


def _make_page_location_scenario(kind: str) -> Callable[..., InvarianceScenario]:
    def build(
        bank: dict[str, list[int]],
        rng: random.Random,
        vocab_size: int,
        token_id_range: int,
        page_size: int,
    ) -> InvarianceScenario:
        del rng, vocab_size, token_id_range, page_size
        if kind == "none":
            preludes: list[PreludeBatch] = []
        elif kind == "one_page":
            preludes = [
                PreludeBatch(
                    name="page_churn_one_page",
                    prompts=[list(bank["dummy_257_a"]), list(bank["dummy_257_b"])],
                    labels=["dummy_a", "dummy_b"],
                )
            ]
        elif kind == "mixed_pages":
            preludes = [
                PreludeBatch(
                    name="page_churn_mixed_pages",
                    prompts=[list(bank["dummy_513"]), list(bank["dummy_769"])],
                    labels=["dummy_513", "dummy_769"],
                )
            ]
        else:
            raise ValueError(kind)
        return InvarianceScenario(
            name=f"target_physical_page_{kind}",
            description=(
                "The 257-token target prompt run after dummy allocations/frees to "
                f"perturb physical page location ({kind})."
            ),
            coverage=["same prompt with different physical page locations"],
            probe_prompts=[list(bank["target_257"])],
            probe_labels=["target"],
            prelude_batches=preludes,
        )

    return build


def _scenario_swa_boundary(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del bank, page_size
    prompts = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=300 + idx,
        )
        for idx, length in enumerate((127, 128, 129))
    ]
    return InvarianceScenario(
        name="swa_boundary_127_128_129_bs3",
        description="No-hit SWA boundary batch of size 3 for eager vs graph bucket 4 replay.",
        coverage=["no-hit SWA boundary lengths 127/128/129", "bs=3 graph bucket 4"],
        probe_prompts=prompts,
        probe_labels=["len127", "len128", "len129"],
        prelude_batches=[],
    )


def _scenario_page_boundary(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del bank
    prompts = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=330 + idx,
        )
        for idx, length in enumerate((255, 256, 257, 258))
    ]
    return InvarianceScenario(
        name="page_boundary_255_256_257_258",
        description="No-hit prompt lengths around the 256-token page boundary.",
        coverage=["page boundary 255/256/257/258"],
        probe_prompts=prompts,
        probe_labels=["len255", "len256", "len257", "len258"],
        prelude_batches=[],
    )


def _scenario_target_same_length_fillers(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    fillers = _make_fillers(
        rng,
        3,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker_base=390,
    )
    return InvarianceScenario(
        name="target_same_length_fillers",
        description="The 257-token target prompt batched with unrelated 257-token fillers.",
        coverage=["target prompt + filler prompts with same length"],
        probe_prompts=[list(bank["target_257"]), *fillers],
        probe_labels=["target", "filler_same0", "filler_same1", "filler_same2"],
        prelude_batches=[],
    )


def _scenario_target_c4_boundary_fillers(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del page_size
    lengths = (252, 253, 254)
    fillers = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=410 + idx,
        )
        for idx, length in enumerate(lengths)
    ]
    return InvarianceScenario(
        name="target_c4_boundary_fillers",
        description="The 257-token target prompt batched with fillers around C4 boundaries.",
        coverage=["target prompt + filler prompts crossing C4 boundaries"],
        probe_prompts=[list(bank["target_257"]), *fillers],
        probe_labels=["target", "len252", "len253", "len254"],
        prelude_batches=[],
    )


def _scenario_target_c128_boundary_fillers(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del page_size
    lengths = (127, 128, 129)
    fillers = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=430 + idx,
        )
        for idx, length in enumerate(lengths)
    ]
    return InvarianceScenario(
        name="target_c128_boundary_fillers",
        description="The 257-token target prompt batched with fillers around C128 boundaries.",
        coverage=["target prompt + filler prompts crossing C128 boundaries"],
        probe_prompts=[list(bank["target_257"]), *fillers],
        probe_labels=["target", "len127", "len128", "len129"],
        prelude_batches=[],
    )


def _scenario_target_swa_boundary_fillers(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del page_size
    lengths = (127, 128, 129)
    fillers = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=450 + idx,
        )
        for idx, length in enumerate(lengths)
    ]
    return InvarianceScenario(
        name="target_swa_boundary_fillers",
        description="The 257-token target prompt batched with fillers around the SWA boundary.",
        coverage=["target prompt + filler prompts around SWA 127/128/129"],
        probe_prompts=[list(bank["target_257"]), *fillers],
        probe_labels=["target", "len127", "len128", "len129"],
        prelude_batches=[],
    )


def _scenario_c4_c128_boundary(
    bank: dict[str, list[int]],
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
) -> InvarianceScenario:
    del bank, page_size
    lengths = (3, 4, 5, 127, 128, 129, 255, 256)
    prompts = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=360 + idx,
        )
        for idx, length in enumerate(lengths)
    ]
    return InvarianceScenario(
        name="c4_c128_boundary_lengths",
        description="No-hit prompt lengths around C4 and C128 multiples.",
        coverage=["C4 boundary around multiples of 4", "C128 boundary around multiples of 128"],
        probe_prompts=prompts,
        probe_labels=[f"len{length}" for length in lengths],
        prelude_batches=[],
    )


_SCENARIO_BUILDERS: dict[str, Callable[..., InvarianceScenario]] = {
    "identical_prompts_batch": _scenario_identical_slots,
    "single_target_alone": _scenario_single_target,
    "target_in_batch_slot0": _make_slot_scenario(0),
    "target_in_batch_slot1": _make_slot_scenario(1),
    "target_in_batch_slot2": _make_slot_scenario(2),
    "target_in_batch_slot3": _make_slot_scenario(3),
    "target_table_row_after_0_dummy": _make_table_row_scenario(0),
    "target_table_row_after_2_dummy": _make_table_row_scenario(2),
    "target_table_row_after_3_dummy": _make_table_row_scenario(3),
    "target_physical_page_none": _make_page_location_scenario("none"),
    "target_physical_page_one_page": _make_page_location_scenario("one_page"),
    "target_physical_page_mixed_pages": _make_page_location_scenario("mixed_pages"),
    "swa_boundary_127_128_129_bs3": _scenario_swa_boundary,
    "page_boundary_255_256_257_258": _scenario_page_boundary,
    "target_same_length_fillers": _scenario_target_same_length_fillers,
    "target_c4_boundary_fillers": _scenario_target_c4_boundary_fillers,
    "target_c128_boundary_fillers": _scenario_target_c128_boundary_fillers,
    "target_swa_boundary_fillers": _scenario_target_swa_boundary_fillers,
    "c4_c128_boundary_lengths": _scenario_c4_c128_boundary,
}
_SCENARIO_ORDER = list(_SCENARIO_BUILDERS)


def _build_scenarios(args: argparse.Namespace, vocab_size: int) -> list[InvarianceScenario]:
    names = args.scenarios or _SCENARIO_ORDER
    bank = _make_prompt_bank(
        vocab_size=vocab_size,
        token_id_range=args.token_id_range,
        seed=args.seed,
        page_size=args.page_size,
    )
    scenarios = []
    for index, name in enumerate(names):
        rng = random.Random(args.seed + 1009 * (index + 1))
        scenarios.append(
            _SCENARIO_BUILDERS[name](
                bank,
                rng,
                vocab_size,
                args.token_id_range,
                args.page_size,
            )
        )
    return scenarios


def _sampling(max_tokens: int):
    from minisgl.core import SamplingParams

    return SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)


def _set_debug_context(*, mode: str, scenario: str, stage: str) -> None:
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_MODE"] = mode
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_SCENARIO"] = scenario
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_STAGE"] = stage


def _prompt_report(prompts: list[list[int]], labels: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "row": idx,
            "label": label,
            "length": len(prompt),
            "sha256_16": _prompt_digest(prompt),
        }
        for idx, (prompt, label) in enumerate(zip(prompts, labels, strict=True))
    ]


def _run_generate(llm: Any, prompts: list[list[int]], max_tokens: int) -> list[dict[str, Any]]:
    import torch

    outputs = llm.generate(prompts, [_sampling(max_tokens) for _ in prompts])
    torch.cuda.synchronize(llm.device)
    return outputs


def _run_scenario(
    llm: Any,
    args: argparse.Namespace,
    scenario: InvarianceScenario,
) -> dict[str, Any]:
    prelude_outputs = []
    tic = time.perf_counter()
    for index, prelude in enumerate(scenario.prelude_batches):
        _set_debug_context(
            mode=args.mode,
            scenario=scenario.name,
            stage=f"prelude{index}:{prelude.name}",
        )
        outputs = _run_generate(llm, prelude.prompts, args.prelude_max_tokens)
        prelude_outputs.append(
            {
                "name": prelude.name,
                "prompts": _prompt_report(prelude.prompts, prelude.labels),
                "outputs": outputs,
            }
        )
        llm.sync_all_ranks()

    _set_debug_context(mode=args.mode, scenario=scenario.name, stage="probe")
    probe_outputs = _run_generate(llm, scenario.probe_prompts, args.probe_max_tokens)
    elapsed_s = time.perf_counter() - tic
    return {
        "name": scenario.name,
        "description": scenario.description,
        "coverage": scenario.coverage,
        "probe_prompts": _prompt_report(scenario.probe_prompts, scenario.probe_labels),
        "prelude_batches": [asdict(item) for item in scenario.prelude_batches],
        "prelude_outputs": prelude_outputs,
        "probe_outputs": probe_outputs,
        "elapsed_s": elapsed_s,
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from minisgl.distributed import DistributedInfo
    from minisgl.llm.llm import LLM

    rank, tp_size, _ = _tp_rank_size(args)
    distributed_init_method = _distributed_init_method(args, tp_size)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_trace"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_DIR"] = str(debug_dir)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_TOPK"] = str(args.topk)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_LOGITS"] = "1"
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_MODE"] = args.mode
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_ACTIVATIONS"] = "1" if args.capture_activations else "0"
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_ATTENTION_COMPONENTS"] = (
        "1" if args.debug_attention_components else "0"
    )
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_MAX_ACTIVATION_ROWS"] = str(args.max_activation_rows)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_ACTIVATIONS"] = (
        "1" if args.save_full_activations else "0"
    )

    kwargs: dict[str, Any] = {}
    if distributed_init_method is not None:
        kwargs["distributed_init_method"] = distributed_init_method

    allow_graph = args.mode == "graph"
    cuda_graph_bs = args.cuda_graph_bs if allow_graph else []
    dtype = _dtype_from_name(args.dtype)
    llm = None
    failures = 0
    try:
        load_tic = time.perf_counter()
        llm = LLM(
            args.model_path,
            dtype=dtype,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=args.max_running_req,
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            allow_dsv4_cuda_graph=allow_graph,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_capture_greedy_sample=False,
            enable_dsv4_radix_prefix_cache=False,
            **kwargs,
        )
        torch.cuda.synchronize(llm.device)
        load_s = time.perf_counter() - load_tic
        scenarios = _build_scenarios(args, llm.engine.sampler.vocab_size)
        scenario_reports = []
        for scenario in scenarios:
            scenario_reports.append(_run_scenario(llm, args, scenario))
            llm.sync_all_ranks()

        if rank == 0:
            payload = {
                "status": "pass",
                "mode": args.mode,
                "model_path": args.model_path,
                "git": _git_info(),
                "config": {
                    "tensor_parallel_size": tp_size,
                    "distributed_init_method": distributed_init_method,
                    "dtype": args.dtype,
                    "page_size": args.page_size,
                    "num_pages": args.num_pages,
                    "max_seq_len": args.max_seq_len,
                    "max_extend_tokens": args.max_extend_tokens,
                    "max_running_req": args.max_running_req,
                    "allow_dsv4_cuda_graph": allow_graph,
                    "cuda_graph_bs": cuda_graph_bs,
                    "enable_dsv4_radix_prefix_cache": False,
                    "probe_max_tokens": args.probe_max_tokens,
                    "prelude_max_tokens": args.prelude_max_tokens,
                    "token_id_range": args.token_id_range,
                    "seed": args.seed,
                    "capture_activations": bool(args.capture_activations),
                    "debug_attention_components": bool(args.debug_attention_components),
                    "max_activation_rows": args.max_activation_rows,
                    "save_full_activations": bool(args.save_full_activations),
                    "disable_toggles": os.environ.get(
                        "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES", ""
                    ),
                    "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
                    "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}),
                },
                "load_seconds": load_s,
                "scenarios": scenario_reports,
                "final_prefix_cache_metrics": llm.cache_manager.prefix_metrics_snapshot(),
            }
            with (output_dir / "run.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(json.dumps({"run_json": str(output_dir / "run.json"), "status": "pass"}))
    except BaseException as exc:
        failures += 1
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "fail",
                "mode": args.mode,
                "error": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(limit=40),
                },
                "git": _git_info(),
            }
            with (output_dir / "run.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            traceback.print_exc()
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                if rank == 0:
                    traceback.print_exc()
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TARGET 08.195 DSV4 exact-path invariance probe.")
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="*", choices=tuple(_SCENARIO_ORDER))
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-extend-tokens", type=int, default=20000)
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument("--cuda-graph-bs", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--prelude-max-tokens", type=int, default=1)
    parser.add_argument("--probe-max-tokens", type=int, default=2)
    parser.add_argument("--token-id-range", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=8195)
    parser.add_argument("--capture-activations", action="store_true")
    parser.add_argument("--debug-attention-components", action="store_true")
    parser.add_argument("--max-activation-rows", type=int, default=4)
    parser.add_argument("--save-full-activations", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for name in _SCENARIO_ORDER:
            print(name)
        raise SystemExit(0)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1")
    if args.max_seq_len <= 0 or args.max_extend_tokens <= 0 or args.max_running_req <= 0:
        parser.error("max sequence/running settings must be positive")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.prelude_max_tokens <= 0 or args.probe_max_tokens <= 1:
        parser.error("--prelude-max-tokens must be positive and --probe-max-tokens must be > 1")
    if args.max_activation_rows <= 0:
        parser.error("--max-activation-rows must be positive")
    args.cuda_graph_bs = sorted(set(args.cuda_graph_bs or []))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
