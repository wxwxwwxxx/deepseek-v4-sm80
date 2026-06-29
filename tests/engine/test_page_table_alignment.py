from __future__ import annotations

from minisgl.engine.engine import _align_up


def test_page_table_alignment_covers_page_sized_tail_writes():
    assert _align_up(16, max(32, 256)) == 256
    assert _align_up(4096, max(32, 256)) == 4096
    assert _align_up(4097, max(32, 256)) == 4352
