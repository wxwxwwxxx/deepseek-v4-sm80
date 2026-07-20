from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import resolve_engine_reasoning_token_ids
from minisgl.engine.graph import GraphCaptureBuffer
from minisgl.engine.graph_memory import estimate_dsv4_sm80_graph_memory
from minisgl.engine.sample import (
    BatchSamplingArgs,
    ReasoningSampler,
    Sampler,
    mask_reasoning_logits_,
)
from minisgl.llm.llm import LLM
from minisgl.reasoning import (
    ReasoningState,
    ReasoningTokenIds,
    advance_reasoning_state,
    effective_reasoning_state,
    initial_reasoning_state,
    resolve_reasoning_token_ids,
)
from minisgl.scheduler.prefill import PrefillAdder, PrefillManager
from minisgl.scheduler.utils import PendingReq
from minisgl.server.args import parse_args

IDS = ReasoningTokenIds(bos=0, eos=1, think_start=2, think_end=3)


def _req(uid: int, effort: str | None = None) -> Req:
    return Req(
        input_ids=torch.tensor([9], dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=object(),
        reasoning_effort=effort,
    )


def _masked(
    state: ReasoningState,
    dominant: int,
    *,
    current_input: int | None = None,
) -> torch.Tensor:
    logits = torch.arange(8, dtype=torch.float32).mul_(-0.1).unsqueeze(0)
    logits[0, dominant] = 100.0
    mask_reasoning_logits_(
        logits,
        torch.tensor([int(state)], dtype=torch.int32),
        IDS,
        current_input_ids=(
            torch.tensor([current_input], dtype=torch.int32) if current_input is not None else None
        ),
    )
    return logits


def _reference_sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    draws: int = 256,
) -> torch.Tensor:
    scaled = logits / temperature
    if top_k is not None:
        cutoff = torch.topk(scaled, top_k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
    probs = torch.softmax(scaled, dim=-1)
    if top_p is not None:
        sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(1, sorted_ids, sorted_probs)
        probs /= probs.sum(dim=-1, keepdim=True)
    torch.manual_seed(13)
    return torch.multinomial(probs[0], draws, replacement=True)


def test_initial_states_and_transition_are_request_local_and_one_way() -> None:
    assert initial_reasoning_state(None) == ReasoningState.CHAT
    assert initial_reasoning_state("high") == ReasoningState.THINKING
    assert initial_reasoning_state("max") == ReasoningState.THINKING
    with pytest.raises(ValueError, match="reasoning_effort"):
        initial_reasoning_state("medium")

    state = ReasoningState.THINKING
    state = advance_reasoning_state(state, 7, think_end_token_id=IDS.think_end)
    assert state == ReasoningState.THINKING
    state = advance_reasoning_state(state, IDS.think_end, think_end_token_id=IDS.think_end)
    assert state == ReasoningState.ANSWER
    assert (
        advance_reasoning_state(state, IDS.think_end, think_end_token_id=IDS.think_end)
        == ReasoningState.ANSWER
    )
    assert (
        advance_reasoning_state(
            ReasoningState.CHAT,
            IDS.think_end,
            think_end_token_id=IDS.think_end,
        )
        == ReasoningState.CHAT
    )


def test_overlap_close_input_is_answer_before_host_state_catches_up() -> None:
    assert (
        effective_reasoning_state(
            ReasoningState.THINKING,
            current_input_token_id=IDS.think_end,
            think_end_token_id=IDS.think_end,
        )
        == ReasoningState.ANSWER
    )
    logits = _masked(
        ReasoningState.THINKING,
        IDS.think_end,
        current_input=IDS.think_end,
    )
    assert torch.isneginf(logits[0, IDS.think_end])
    assert not torch.isneginf(logits[0, IDS.eos])


@pytest.mark.parametrize(
    ("state", "forbidden"),
    [
        (ReasoningState.CHAT, IDS.think_start),
        (ReasoningState.CHAT, IDS.think_end),
        (ReasoningState.THINKING, IDS.think_start),
        (ReasoningState.THINKING, IDS.eos),
        (ReasoningState.ANSWER, IDS.think_start),
        (ReasoningState.ANSWER, IDS.think_end),
    ],
)
def test_forbidden_top1_loses_greedy_and_stochastic_sampling(
    state: ReasoningState,
    forbidden: int,
) -> None:
    logits = _masked(state, forbidden)
    assert int(logits.argmax(dim=-1).item()) != forbidden
    for kwargs in ({}, {"top_k": 4}, {"top_p": 0.9}, {"top_k": 4, "top_p": 0.9}):
        samples = _reference_sample(logits, temperature=1.0, **kwargs)
        assert forbidden not in samples.tolist()


def test_think_end_is_legal_in_thinking_and_changes_the_next_step_mask() -> None:
    first = _masked(ReasoningState.THINKING, IDS.think_end)
    assert int(first.argmax(dim=-1).item()) == IDS.think_end
    second = _masked(
        ReasoningState.THINKING,
        IDS.think_end,
        current_input=IDS.think_end,
    )
    assert int(second.argmax(dim=-1).item()) != IDS.think_end


def test_mixed_batch_masks_rows_independently_without_mutating_states() -> None:
    states = torch.tensor(
        [ReasoningState.CHAT, ReasoningState.THINKING, ReasoningState.ANSWER],
        dtype=torch.int32,
    )
    original = states.clone()
    logits = torch.zeros((3, 8), dtype=torch.float32)
    logits[0, IDS.think_end] = 10
    logits[1, IDS.eos] = 10
    logits[2, IDS.think_end] = 10
    mask_reasoning_logits_(logits, states, IDS)
    assert logits.argmax(dim=-1).tolist() == [0, 0, 0]
    assert torch.equal(states, original)


def test_sampler_calls_the_same_mask_before_stochastic_backend(monkeypatch) -> None:
    req = _req(0, "high")
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids
    batch.reasoning_states = torch.tensor([ReasoningState.THINKING], dtype=torch.int32)
    observed = {}

    def fake_sample_impl(logits, temperatures, top_k, top_p):
        observed["logits"] = logits.clone()
        return torch.tensor([4])

    monkeypatch.setattr("minisgl.engine.sample.sample_impl", fake_sample_impl)
    sampler = ReasoningSampler(torch.device("cpu"), 8, IDS)
    result = sampler.sample(
        torch.zeros((1, 8)),
        BatchSamplingArgs(temperatures=torch.ones(1)),
        batch,
    )
    assert result.tolist() == [4]
    assert torch.isneginf(observed["logits"][0, IDS.think_start])
    assert torch.isneginf(observed["logits"][0, IDS.eos])


def test_external_sampler_ignores_graph_padding_input_rows(monkeypatch) -> None:
    reqs = [_req(0, None), _req(1, "high")]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs + [_req(99, None), _req(99, None)]
    batch.input_ids = torch.tensor(
        [7, IDS.think_end, IDS.think_end, IDS.think_end], dtype=torch.int32
    )
    batch.reasoning_states = torch.tensor(
        [ReasoningState.CHAT, ReasoningState.THINKING], dtype=torch.int32
    )
    observed = {}

    def fake_sample_impl(logits, temperatures, top_k, top_p):
        observed["logits"] = logits.clone()
        return torch.tensor([4, 4])

    monkeypatch.setattr("minisgl.engine.sample.sample_impl", fake_sample_impl)
    sampler = ReasoningSampler(torch.device("cpu"), 8, IDS)
    result = sampler.sample(
        torch.zeros((2, 8)),
        BatchSamplingArgs(temperatures=torch.ones(2)),
        batch,
    )
    assert result.tolist() == [4, 4]
    assert torch.isneginf(observed["logits"][0, IDS.think_end])
    # Real row 1 is the just-closed overlap row and must already be ANSWER.
    assert torch.isneginf(observed["logits"][1, IDS.think_end])
    assert not torch.isneginf(observed["logits"][1, IDS.eos])


def test_graph_padding_rows_are_separate_and_do_not_change_real_request_state() -> None:
    buffer = GraphCaptureBuffer.init(
        4,
        8,
        torch.device("cpu"),
        reasoning_sampler_contract_enabled=True,
    )
    assert buffer.reasoning_states is not None
    real_states = torch.tensor([ReasoningState.THINKING, ReasoningState.ANSWER], dtype=torch.int32)
    batch = SimpleNamespace(
        size=2,
        padded_size=4,
        input_ids=torch.tensor([5, 6, 0, 0], dtype=torch.int32),
        out_loc=torch.arange(4, dtype=torch.int32),
        positions=torch.arange(4, dtype=torch.int32),
        reasoning_states=real_states,
    )
    buffer.reasoning_states.fill_(ReasoningState.ANSWER)
    buffer.copy_from(batch)
    assert buffer.reasoning_states.tolist() == [
        ReasoningState.THINKING,
        ReasoningState.ANSWER,
        0,
        0,
    ]
    assert real_states.tolist() == [ReasoningState.THINKING, ReasoningState.ANSWER]


def test_effective_config_and_server_flag_select_production_or_oracle() -> None:
    base = {"model_path": "unused", "tp_info": DistributedInfo(0, 1)}
    assert not EngineConfig(**base).reasoning_sampler_contract_enabled
    assert EngineConfig(
        **base,
        enable_reasoning_sampler_contract=True,
    ).reasoning_sampler_contract_enabled
    assert not EngineConfig(
        **base,
        enable_reasoning_sampler_contract=False,
    ).reasoning_sampler_contract_enabled
    assert not EngineConfig(
        **base,
        dsv4_runtime_mode="fallback",
    ).reasoning_sampler_contract_enabled
    assert not EngineConfig(
        **base,
        dsv4_runtime_mode="fallback",
        enable_reasoning_sampler_contract=True,
    ).reasoning_sampler_contract_enabled

    args, _ = parse_args(["--model-path", "unused", "--enable-reasoning-sampler-contract"])
    assert args.enable_reasoning_sampler_contract
    assert args.reasoning_sampler_contract_enabled


def test_disabled_sampler_preserves_release_logits_and_creates_no_state_metadata(
    monkeypatch,
) -> None:
    req = _req(0, "high")
    req.sampling_params = SamplingParams(temperature=1.0, top_k=4, top_p=0.9)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids
    sampler = Sampler(torch.device("cpu"), 8)
    args = sampler.prepare(batch)
    assert not hasattr(batch, "reasoning_states")

    original = torch.zeros((1, 8), dtype=torch.float32)
    original[0, IDS.eos] = 100.0
    original[0, IDS.think_start] = 90.0
    observed = {}

    def fake_sample_impl(logits, temperatures, top_k, top_p):
        observed["logits"] = logits.clone()
        observed["top_k"] = top_k
        observed["top_p"] = top_p
        return logits.argmax(dim=-1)

    monkeypatch.setattr("minisgl.engine.sample.sample_impl", fake_sample_impl)
    logits = original.clone()
    result = sampler.sample(logits, args, batch)
    assert result.tolist() == [IDS.eos]
    assert torch.equal(logits, original)
    assert torch.equal(observed["logits"], original)
    assert observed["top_k"].tolist() == [4]
    assert observed["top_p"].tolist() == pytest.approx([0.9])


def test_disabled_graph_metadata_and_memory_match_release_census() -> None:
    bs, vocab = 4, 8
    buffer = GraphCaptureBuffer.init(
        bs,
        vocab,
        torch.device("cpu"),
        capture_greedy_sample=True,
        reasoning_sampler_contract_enabled=False,
    )
    assert buffer.reasoning_states is None
    release_nbytes = (3 * bs + 1 + bs * vocab + bs) * torch.int32.itemsize
    assert buffer.nbytes() == release_nbytes

    batch = SimpleNamespace(
        size=2,
        padded_size=bs,
        input_ids=torch.tensor([5, 6, 0, 0], dtype=torch.int32),
        out_loc=torch.arange(bs, dtype=torch.int32),
        positions=torch.arange(bs, dtype=torch.int32),
    )
    release_copy_bytes = bs * 3 * torch.int32.itemsize + torch.int32.itemsize
    assert buffer.copy_from(batch) == release_copy_bytes
    assert not hasattr(batch, "reasoning_states")

    disabled = estimate_dsv4_sm80_graph_memory(
        [1, 2, 4],
        metadata_width=8192,
        page_size=256,
        capture_greedy_sample=True,
        reasoning_sampler_contract_enabled=False,
    )
    enabled = estimate_dsv4_sm80_graph_memory(
        [1, 2, 4],
        metadata_width=8192,
        page_size=256,
        capture_greedy_sample=True,
        reasoning_sampler_contract_enabled=True,
    )
    assert enabled.metadata_allowance_bytes - disabled.metadata_allowance_bytes == bs * 4
    assert enabled.estimate_bytes - disabled.estimate_bytes == bs * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
def test_eager_and_cuda_graph_greedy_masks_have_mixed_overlap_parity() -> None:
    device = torch.device("cuda:0")
    states = torch.tensor(
        [ReasoningState.CHAT, ReasoningState.THINKING, ReasoningState.THINKING],
        dtype=torch.int32,
        device=device,
    )
    input_ids = torch.tensor([7, 7, IDS.think_end], dtype=torch.int32, device=device)
    source = torch.zeros((3, 8), dtype=torch.float32, device=device)
    source[0, IDS.think_end] = 20
    source[1, IDS.eos] = 20
    source[2, IDS.think_end] = 20

    eager_logits = source.clone()
    mask_reasoning_logits_(
        eager_logits,
        states,
        IDS,
        current_input_ids=input_ids,
    )
    eager = eager_logits.argmax(dim=-1)

    graph_logits = torch.empty_like(source)
    graph_out = torch.empty(3, dtype=torch.int64, device=device)
    graph = torch.cuda.CUDAGraph()
    graph_logits.copy_(source)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        graph_logits.copy_(source)
        mask_reasoning_logits_(
            graph_logits,
            states,
            IDS,
            current_input_ids=input_ids,
        )
        torch.argmax(graph_logits, dim=-1, out=graph_out)
    graph.replay()
    torch.cuda.synchronize(device)
    assert graph_out.tolist() == eager.tolist()
    assert graph_out.tolist() == [0, 0, 0]


def _dynamic_graph_batch(
    states: list[ReasoningState],
    input_ids: list[int],
    *,
    padded_size: int,
    device: torch.device,
) -> SimpleNamespace:
    size = len(states)
    return SimpleNamespace(
        size=size,
        padded_size=padded_size,
        input_ids=torch.tensor(
            input_ids + [0] * (padded_size - size),
            dtype=torch.int32,
            device=device,
        ),
        out_loc=torch.arange(padded_size, dtype=torch.int32, device=device),
        positions=torch.arange(padded_size, dtype=torch.int32, device=device),
        reasoning_states=torch.tensor(states, dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_enabled_graph_buffer_replay_reads_dynamic_reasoning_metadata() -> None:
    device = torch.device("cuda:0")
    padded_size, vocab = 8, 8
    buffer = GraphCaptureBuffer.init(
        padded_size,
        vocab,
        device,
        capture_greedy_sample=True,
        reasoning_sampler_contract_enabled=True,
    )
    assert buffer.reasoning_states is not None
    assert buffer.next_tokens is not None
    state_ptr = buffer.reasoning_states.data_ptr()
    input_ptr = buffer.input_ids.data_ptr()

    source = torch.zeros((padded_size, vocab), dtype=torch.float32, device=device)
    source[:, IDS.think_end] = 100.0
    source[:, IDS.eos] = 90.0
    source[:, 4] = 80.0

    first = _dynamic_graph_batch(
        [
            ReasoningState.CHAT,
            ReasoningState.THINKING,
            ReasoningState.ANSWER,
            ReasoningState.THINKING,
        ],
        [7, 7, 7, IDS.think_end],
        padded_size=padded_size,
        device=device,
    )
    buffer.copy_from(first)

    graph = torch.cuda.CUDAGraph()
    buffer.logits.copy_(source)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        buffer.logits.copy_(source)
        mask_reasoning_logits_(
            buffer.logits,
            buffer.reasoning_states,
            IDS,
            current_input_ids=buffer.input_ids,
        )
        buffer.next_tokens[:] = torch.argmax(buffer.logits, dim=-1).to(torch.int32)

    def eager_reference(batch: SimpleNamespace) -> torch.Tensor:
        logits = source[: batch.size].clone()
        mask_reasoning_logits_(
            logits,
            batch.reasoning_states,
            IDS,
            current_input_ids=batch.input_ids[: batch.size],
        )
        return logits.argmax(dim=-1)

    graph.replay()
    torch.cuda.synchronize(device)
    assert buffer.next_tokens[: first.size].tolist() == eager_reference(first).tolist()
    assert buffer.next_tokens[: first.size].tolist() == [IDS.eos, IDS.think_end, IDS.eos, IDS.eos]
    assert buffer.reasoning_states[first.size :].eq(ReasoningState.CHAT).all()

    second = _dynamic_graph_batch(
        [
            ReasoningState.THINKING,
            ReasoningState.CHAT,
            ReasoningState.THINKING,
            ReasoningState.ANSWER,
        ],
        [7, 7, IDS.think_end, 7],
        padded_size=padded_size,
        device=device,
    )
    buffer.copy_from(second)
    assert buffer.reasoning_states.data_ptr() == state_ptr
    assert buffer.input_ids.data_ptr() == input_ptr
    graph.replay()
    torch.cuda.synchronize(device)
    assert buffer.next_tokens[: second.size].tolist() == eager_reference(second).tolist()
    assert buffer.next_tokens[: second.size].tolist() == [IDS.think_end, IDS.eos, IDS.eos, IDS.eos]
    assert buffer.reasoning_states[second.size :].eq(ReasoningState.CHAT).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_disabled_graph_buffer_replay_preserves_raw_top1_without_reasoning_metadata() -> None:
    device = torch.device("cuda:0")
    padded_size, vocab = 8, 8
    buffer = GraphCaptureBuffer.init(
        padded_size,
        vocab,
        device,
        capture_greedy_sample=True,
        reasoning_sampler_contract_enabled=False,
    )
    assert buffer.reasoning_states is None
    assert buffer.next_tokens is not None
    source = torch.zeros((padded_size, vocab), dtype=torch.float32, device=device)
    source[:, IDS.think_end] = 100.0

    def raw_batch(first_token: int) -> SimpleNamespace:
        return SimpleNamespace(
            size=4,
            padded_size=padded_size,
            input_ids=torch.tensor(
                [first_token, 7, IDS.think_end, 7, 0, 0, 0, 0],
                dtype=torch.int32,
                device=device,
            ),
            out_loc=torch.arange(padded_size, dtype=torch.int32, device=device),
            positions=torch.arange(padded_size, dtype=torch.int32, device=device),
        )

    first = raw_batch(7)
    buffer.copy_from(first)
    graph = torch.cuda.CUDAGraph()
    buffer.logits.copy_(source)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        buffer.logits.copy_(source)
        buffer.next_tokens[:] = torch.argmax(buffer.logits, dim=-1).to(torch.int32)
    graph.replay()
    torch.cuda.synchronize(device)
    assert buffer.next_tokens[: first.size].tolist() == [IDS.think_end] * first.size

    second = raw_batch(IDS.think_end)
    buffer.copy_from(second)
    graph.replay()
    torch.cuda.synchronize(device)
    assert buffer.next_tokens[: second.size].tolist() == [IDS.think_end] * second.size
    assert buffer.reasoning_states is None


def test_abort_slot_reuse_chunked_prefill_and_radix_hit_keep_request_owned_state() -> None:
    pending = PrefillManager(
        cache_manager=SimpleNamespace(),
        table_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
    )
    user = SimpleNamespace(
        uid=11,
        input_ids=torch.tensor([4, 5], dtype=torch.int32),
        sampling_params=SamplingParams(),
        reasoning_effort="max",
    )
    pending.add_one_req(user)
    assert pending.pending_list[0].reasoning_effort == "max"

    chunk_adder = PrefillAdder(
        token_budget=1,
        reserved_size=0,
        cache_manager=SimpleNamespace(),
        table_manager=SimpleNamespace(
            token_pool=torch.empty((1, 8), dtype=torch.int32)
        ),
    )
    chunk = chunk_adder._add_one_req(
        pending_req=PendingReq(
            uid=12,
            input_ids=torch.tensor([4, 5, 6], dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=2),
            reasoning_effort="high",
        ),
        cache_handle=object(),
        table_idx=0,
        cached_len=0,
    )
    assert chunk.reasoning_effort == "high"
    assert chunk.reasoning_state == ReasoningState.THINKING
    assert pending.abort_req(11) is None
    assert not pending.pending_list

    # Reusing the same table slot creates a fresh request state; a radix cache
    # hit only changes cached_len and never derives protocol state from tokens.
    old = _req(0, "high")
    old.observe_generated_token(IDS.think_end, think_end_token_id=IDS.think_end)
    reused = _req(0, None)
    radix_hit = Req(
        input_ids=torch.tensor([4, 5, 6], dtype=torch.int32),
        table_idx=1,
        cached_len=2,
        output_len=4,
        uid=12,
        sampling_params=SamplingParams(),
        cache_handle=object(),
        reasoning_effort="high",
    )
    assert old.reasoning_state == ReasoningState.ANSWER
    assert reused.reasoning_state == ReasoningState.CHAT
    assert radix_hit.reasoning_state == ReasoningState.THINKING


class _FakeTokenizer:
    bos_token = "<bos>"
    bos_token_id = 10
    eos_token = "<eos>"
    eos_token_id = 11
    mapping = {"<bos>": 10, "<eos>": 11, "<think>": 12, "</think>": 13}

    def encode(self, marker, add_special_tokens=False):
        assert not add_special_tokens
        return [self.mapping[marker]]

    def convert_tokens_to_ids(self, marker):
        return self.mapping[marker]


def test_token_ids_are_resolved_from_tokenizer_and_mismatch_fails_clearly() -> None:
    assert resolve_reasoning_token_ids(_FakeTokenizer()) == ReasoningTokenIds(10, 11, 12, 13)
    broken = _FakeTokenizer()
    broken.mapping = {**broken.mapping, "</think>": 12}
    with pytest.raises(RuntimeError, match="must be distinct"):
        resolve_reasoning_token_ids(broken)


def test_engine_skips_extra_tokenizer_load_for_disabled_and_fallback(monkeypatch) -> None:
    calls = []

    def fake_load_tokenizer(model_path):
        calls.append(model_path)
        return _FakeTokenizer()

    monkeypatch.setattr("minisgl.engine.engine.load_tokenizer", fake_load_tokenizer)
    base = {"model_path": "checkpoint", "tp_info": DistributedInfo(0, 1)}
    disabled = EngineConfig(**base)
    fallback = EngineConfig(
        **base,
        dsv4_runtime_mode="fallback",
    )
    enabled = EngineConfig(**base, enable_reasoning_sampler_contract=True)

    assert resolve_engine_reasoning_token_ids(disabled) is None
    assert resolve_engine_reasoning_token_ids(fallback) is None
    assert calls == []
    assert resolve_engine_reasoning_token_ids(enabled) == ReasoningTokenIds(10, 11, 12, 13)
    assert calls == ["checkpoint"]


@pytest.mark.parametrize("effort", ["high", "max"])
def test_offline_thinking_prompt_requires_official_formatter_boundary(effort) -> None:
    llm = object.__new__(LLM)
    llm.reasoning_sampler_contract_enabled = True
    llm.engine = SimpleNamespace(reasoning_token_ids=IDS)

    with pytest.raises(ValueError, match="official DeepSeek V4 thinking formatter"):
        llm.generate(
            [[9]],
            SamplingParams(max_tokens=1),
            reasoning_effort=effort,
        )
    llm._validate_offline_reasoning_prompt([9, IDS.think_start], effort)


def test_offline_prompt_validation_is_absent_in_disabled_oracle_mode() -> None:
    llm = object.__new__(LLM)
    llm.reasoning_sampler_contract_enabled = False
    llm.engine = SimpleNamespace(reasoning_token_ids=None)
    llm._validate_offline_reasoning_prompt([9], "high")


def test_saved_multiple_think_failure_pattern_is_structurally_prevented() -> None:
    state = ReasoningState.THINKING
    state = advance_reasoning_state(state, IDS.think_end, think_end_token_id=IDS.think_end)
    assert state == ReasoningState.ANSWER
    logits = _masked(state, IDS.think_end)
    assert torch.isneginf(logits[0, IDS.think_end])


def test_saved_chat_answer_with_think_failure_pattern_is_structurally_prevented() -> None:
    logits = _masked(ReasoningState.CHAT, IDS.think_end)
    assert torch.isneginf(logits[0, IDS.think_end])
