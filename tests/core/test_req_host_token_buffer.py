from __future__ import annotations

import pytest
import torch
from minisgl.core import Req, SamplingParams


def _make_req(*, output_len: int = 4) -> Req:
    return Req(
        input_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=object(),
    )


def test_append_host_reuses_fixed_capacity_storage() -> None:
    req = _make_req()

    req.append_host(torch.tensor([21], dtype=torch.int32))
    storage_ptr = req.input_ids.untyped_storage().data_ptr()
    prefix_view = req.input_ids[:3]

    for token in (22, 23, 24):
        req.append_host(torch.tensor([token], dtype=torch.int32))
        assert req.input_ids.untyped_storage().data_ptr() == storage_ptr

    assert req.input_ids.tolist() == [11, 12, 13, 21, 22, 23, 24]
    assert prefix_view.tolist() == [11, 12, 13]
    assert req.input_ids.untyped_storage().nbytes() == req.max_device_len * 4
    assert not req.can_commit_token


def test_append_host_validates_token_contract_and_capacity() -> None:
    req = _make_req(output_len=1)

    with pytest.raises(ValueError, match="exactly one token"):
        req.append_host(torch.tensor([21, 22], dtype=torch.int32))
    with pytest.raises(ValueError, match="CPU token dtype"):
        req.append_host(torch.tensor([21], dtype=torch.int64))

    req.append_host(torch.tensor([21], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="no remaining host token capacity"):
        req.append_host(torch.tensor([22], dtype=torch.int32))
