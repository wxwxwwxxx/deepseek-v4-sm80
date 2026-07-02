from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
DSV4_DIRECT_COPY_NVTX_ENV = "MINISGL_DSV4_PROFILE_DIRECT_COPY_NVTX"


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


def dsv4_direct_copy_nvtx_enabled() -> bool:
    return os.environ.get(DSV4_DIRECT_COPY_NVTX_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def dsv4_direct_copy_nvtx(name: str, **tensors: "torch.Tensor"):
    if not dsv4_direct_copy_nvtx_enabled():
        return nullcontext()
    try:
        import torch
    except Exception:
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    suffixes = []
    for key, tensor in tensors.items():
        shape = "x".join(str(int(dim)) for dim in tensor.shape)
        suffixes.append(f"{key}={shape}:{tensor.dtype}")
    label = f"dsv4.direct_copy.{name}"
    if suffixes:
        label = f"{label}|{';'.join(suffixes)}"
    return torch.cuda.nvtx.range(label)
