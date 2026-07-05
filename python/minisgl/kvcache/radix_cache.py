from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.kvcache.deepseek_v4_pool import DSV4ComponentPageHandles, DSV4SWAPageHandles
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
        self._dsv4_component_pages: DSV4ComponentPageHandles | None = None
        self._dsv4_swa_pages: DSV4SWAPageHandles | None = None

    def set_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        dsv4_component_pages: DSV4ComponentPageHandles | None = None,
        dsv4_swa_pages: DSV4SWAPageHandles | None = None,
    ) -> None:
        assert len(key) == len(value)
        if dsv4_component_pages is not None and dsv4_component_pages.length != len(key):
            raise ValueError(
                "Radix DSV4 component handle length mismatch: "
                f"key={len(key)}, components={dsv4_component_pages.length}"
            )
        if dsv4_swa_pages is not None and dsv4_swa_pages.length != len(key):
            raise ValueError(
                "Radix DSV4 SWA handle length mismatch: "
                f"key={len(key)}, swa={dsv4_swa_pages.length}"
            )
        self._key = key
        self._value = value
        self._length = len(key)
        self._dsv4_component_pages = dsv4_component_pages
        self._dsv4_swa_pages = dsv4_swa_pages

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        parent_components = (
            None
            if self._dsv4_component_pages is None
            else self._dsv4_component_pages.slice_tokens(0, pos)
        )
        child_components = (
            None
            if self._dsv4_component_pages is None
            else self._dsv4_component_pages.slice_tokens(pos, self.length)
        )
        parent_swa = (
            None
            if self._dsv4_swa_pages is None
            else self._dsv4_swa_pages.slice_tokens(0, pos)
        )
        child_swa = (
            None
            if self._dsv4_swa_pages is None
            else self._dsv4_swa_pages.slice_tokens(pos, self.length)
        )
        new_node.set_key_value(
            self._key[:pos],
            self._value[:pos],
            parent_components,
            parent_swa,
        )
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:], child_components, child_swa)
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)

    def get_dsv4_component_pages(self) -> DSV4ComponentPageHandles | None:
        node = self.node
        handles: List[DSV4ComponentPageHandles] = []
        while not node.is_root():
            if node._dsv4_component_pages is not None:
                handles.append(node._dsv4_component_pages)
            node = node.parent
        handles.reverse()
        return DSV4ComponentPageHandles.concat(handles)

    def get_dsv4_swa_pages(self) -> DSV4SWAPageHandles | None:
        node = self.node
        handles: List[DSV4SWAPageHandles] = []
        while not node.is_root():
            if node._dsv4_swa_pages is not None:
                handles.append(node._dsv4_swa_pages)
            node = node.parent
        handles.reverse()
        return DSV4SWAPageHandles.concat(handles)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected
        self.dsv4_component_evict_callback: Callable[[DSV4ComponentPageHandles | None], None] | None
        self.dsv4_component_evict_callback = None
        self.dsv4_component_ownership_enabled = False
        self.dsv4_swa_independent_lifecycle_enabled = False
        self.dsv4_swa_evict_callback: Callable[[DSV4SWAPageHandles | None, bool], None] | None
        self.dsv4_swa_evict_callback = None

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        if self.dsv4_component_ownership_enabled:
            node, prefix_len = self._dsv4_safe_match_boundary(node, prefix_len)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        dsv4_component_pages: DSV4ComponentPageHandles | None = None,
        dsv4_swa_pages: DSV4SWAPageHandles | None = None,
        dsv4_component_pages_builder: Callable[
            [int, int], DSV4ComponentPageHandles | None
        ]
        | None = None,
        dsv4_swa_pages_builder: Callable[[int, int], DSV4SWAPageHandles | None]
        | None = None,
    ) -> InsertResult:
        if dsv4_component_pages is not None and dsv4_component_pages_builder is not None:
            raise ValueError(
                "Pass either dsv4_component_pages or dsv4_component_pages_builder, not both"
            )
        if dsv4_swa_pages is not None and dsv4_swa_pages_builder is not None:
            raise ValueError("Pass either dsv4_swa_pages or dsv4_swa_pages_builder, not both")
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        if dsv4_component_pages is not None:
            dsv4_component_pages = dsv4_component_pages.slice_tokens(0, insert_len)
        if dsv4_swa_pages is not None:
            dsv4_swa_pages = dsv4_swa_pages.slice_tokens(0, insert_len)
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            if dsv4_component_pages_builder is not None:
                new_components = dsv4_component_pages_builder(prefix_len, insert_len)
                if (
                    new_components is not None
                    and new_components.length != insert_len - prefix_len
                ):
                    raise ValueError(
                        "Radix DSV4 component builder returned length mismatch: "
                        f"new_segment={insert_len - prefix_len}, "
                        f"components={new_components.length}"
                    )
            else:
                new_components = (
                    None
                    if dsv4_component_pages is None
                    else dsv4_component_pages.slice_tokens(prefix_len, insert_len)
                )
            if dsv4_swa_pages_builder is not None:
                new_swa = dsv4_swa_pages_builder(prefix_len, insert_len)
                if new_swa is not None and new_swa.length != insert_len - prefix_len:
                    raise ValueError(
                        "Radix DSV4 SWA builder returned length mismatch: "
                        f"new_segment={insert_len - prefix_len}, swa={new_swa.length}"
                    )
            else:
                new_swa = (
                    None
                    if dsv4_swa_pages is None
                    else dsv4_swa_pages.slice_tokens(prefix_len, insert_len)
                )
            new_node.set_key_value(
                input_ids[prefix_len:],
                indices[prefix_len:].clone(),
                new_components,
                new_swa,
            )
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            if self.dsv4_component_evict_callback is not None:
                self.dsv4_component_evict_callback(node._dsv4_component_pages)
            if self.dsv4_swa_evict_callback is not None:
                self.dsv4_swa_evict_callback(node._dsv4_swa_pages, False)
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        evictable_size = 0
        protected_size = 0
        stack: List[RadixTreeNode] = [self.root_node]
        visited: set[int] = set()

        while stack:
            node = stack.pop()
            if node.uuid in visited:
                raise RuntimeError(f"Radix cache cycle detected at node {node.uuid}")
            visited.add(node.uuid)
            if node.is_root():
                if node.ref_count <= 0:
                    raise RuntimeError("Radix cache root must stay protected")
            else:
                if node.length != len(node.value):
                    raise RuntimeError(
                        f"Radix cache node {node.uuid} length mismatch: "
                        f"length={node.length}, value={len(node.value)}"
                    )
                if (
                    node._dsv4_component_pages is not None
                    and node._dsv4_component_pages.length != node.length
                ):
                    raise RuntimeError(
                        f"Radix cache node {node.uuid} DSV4 component length mismatch: "
                        f"node={node.length}, components={node._dsv4_component_pages.length}"
                    )
                if node._dsv4_swa_pages is not None and node._dsv4_swa_pages.length != node.length:
                    raise RuntimeError(
                        f"Radix cache node {node.uuid} DSV4 SWA length mismatch: "
                        f"node={node.length}, swa={node._dsv4_swa_pages.length}"
                    )
                if node.ref_count < 0:
                    raise RuntimeError(f"Radix cache node {node.uuid} has negative ref_count")
                if node.ref_count == 0:
                    evictable_size += node.length
                else:
                    protected_size += node.length
            for child in node.children.values():
                if child._parent is not node:
                    raise RuntimeError(
                        f"Radix cache child {child.uuid} has wrong parent for node {node.uuid}"
                    )
                stack.append(child)

        if evictable_size != self.evictable_size or protected_size != self.protected_size:
            raise RuntimeError(
                "Radix cache size accounting mismatch: "
                f"evictable={self.evictable_size}/{evictable_size}, "
                f"protected={self.protected_size}/{protected_size}"
            )

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def release_dsv4_full_head(
        self,
        handle: BaseCacheHandle,
        *,
        tail_tokens: int,
    ) -> torch.Tensor:
        if not self.dsv4_component_ownership_enabled:
            return self.empty_tensor
        assert isinstance(handle, RadixCacheHandle)
        path = self._path_from_root(handle.node)
        if not path:
            return self.empty_tensor
        tail_tokens = int(tail_tokens)
        tail_tokens = 0 if tail_tokens <= 0 else align_down(max(tail_tokens, self.page_size), self.page_size)
        total_len = sum(node.length for node in path)
        releasable_until = align_down(max(total_len - tail_tokens, 0), self.page_size)
        if releasable_until <= 0:
            return self.empty_tensor
        released_chunks: list[torch.Tensor] = []
        offset = 0
        for node in list(path):
            node_start = offset
            node_end = offset + node.length
            offset = node_end
            release_end = min(node_end, releasable_until) - node_start
            release_end = align_down(max(release_end, 0), self.page_size)
            if release_end <= 0:
                continue
            target = node
            if release_end < node.length:
                target = node.split_at(release_end)
            head = target.value[:release_end]
            if head.numel() == 0:
                continue
            valid_pages = head.view(-1, self.page_size)[:, 0] >= 0
            if not bool(torch.any(valid_pages)):
                continue
            released_chunks.append(head.view(-1, self.page_size)[valid_pages].reshape(-1).clone())
            head.view(-1, self.page_size)[valid_pages] = -1
        if not released_chunks:
            return self.empty_tensor
        return torch.cat(released_chunks)

    def release_dsv4_swa_out_of_window(
        self,
        handle: BaseCacheHandle,
        *,
        tail_tokens: int,
    ) -> int:
        if not self.dsv4_swa_independent_lifecycle_enabled:
            return 0
        assert isinstance(handle, RadixCacheHandle)
        path = self._path_from_root(handle.node)
        if not path:
            return 0
        tail_tokens = int(tail_tokens)
        tail_tokens = 0 if tail_tokens <= 0 else align_down(max(tail_tokens, self.page_size), self.page_size)
        total_len = sum(node.length for node in path)
        tombstone_until = align_down(max(total_len - tail_tokens, 0), self.page_size)
        if tombstone_until <= 0:
            return 0
        tombstoned_pages = 0
        offset = 0
        for node in list(path):
            node_start = offset
            node_end = offset + node.length
            offset = node_end
            release_end = min(node_end, tombstone_until) - node_start
            release_end = align_down(max(release_end, 0), self.page_size)
            if release_end <= 0 or node._dsv4_swa_pages is None:
                continue
            target = node
            if release_end < node.length:
                target = node.split_at(release_end)
            assert target._dsv4_swa_pages is not None
            updated, released = target._dsv4_swa_pages.tombstone_tokens(0, release_end)
            target._dsv4_swa_pages = updated
            tombstoned_pages += released.live_pages
            if self.dsv4_swa_evict_callback is not None:
                self.dsv4_swa_evict_callback(released, True)
        return tombstoned_pages

    def release_dsv4_evictable_swa_pages(self, pages: int) -> int:
        if not self.dsv4_swa_independent_lifecycle_enabled:
            return 0
        pages = int(pages)
        if pages <= 0:
            return 0
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        released_pages = 0
        while released_pages < pages and leave_nodes:
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            if node._dsv4_swa_pages is not None and node._dsv4_swa_pages.live_pages > 0:
                updated, released = node._dsv4_swa_pages.tombstone_tokens(0, node.length)
                node._dsv4_swa_pages = updated
                released_pages += released.live_pages
                if self.dsv4_swa_evict_callback is not None:
                    self.dsv4_swa_evict_callback(released, True)
            parent = node.parent
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leave_nodes, parent)
        return released_pages

    def _path_from_root(self, node: RadixTreeNode) -> list[RadixTreeNode]:
        path: list[RadixTreeNode] = []
        while not node.is_root():
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    @property
    def dsv4_evictable_live_full_tokens(self) -> int:
        if not self.dsv4_component_ownership_enabled:
            return self.evictable_size
        total = 0
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if not node.is_root() and node.ref_count == 0:
                total += int(torch.count_nonzero(node.value >= 0).item())
            stack.extend(node.children.values())
        return total

    @property
    def dsv4_evictable_component_tokens(self) -> int:
        if not self.dsv4_component_ownership_enabled:
            return self.evictable_size
        total = 0
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if (
                not node.is_root()
                and node.ref_count == 0
                and node._dsv4_component_pages is not None
            ):
                total += node.length
            stack.extend(node.children.values())
        return total

    @property
    def dsv4_evictable_swa_tokens(self) -> int:
        if not self.dsv4_swa_independent_lifecycle_enabled:
            return self.evictable_size
        return self._dsv4_swa_tokens_with_ref(ref_count_zero=True)

    @property
    def dsv4_protected_swa_tokens(self) -> int:
        if not self.dsv4_swa_independent_lifecycle_enabled:
            return self.protected_size
        return self._dsv4_swa_tokens_with_ref(ref_count_zero=False)

    def _dsv4_swa_tokens_with_ref(self, *, ref_count_zero: bool) -> int:
        total = 0
        stack: List[RadixTreeNode] = [self.root_node]
        while stack:
            node = stack.pop()
            if not node.is_root() and node._dsv4_swa_pages is not None:
                if (node.ref_count == 0) is ref_count_zero:
                    total += node._dsv4_swa_pages.live_pages * self.page_size
            stack.extend(node.children.values())
        return total

    def _dsv4_safe_match_boundary(
        self,
        node: RadixTreeNode,
        prefix_len: int,
    ) -> tuple[RadixTreeNode, int]:
        while not node.is_root():
            if (
                self._dsv4_node_has_live_tail(node)
                and self._dsv4_path_has_state_or_live_tail(node)
            ):
                return node, prefix_len
            prefix_len -= node.length
            node = node.parent
        return node, 0

    def _dsv4_path_has_state_or_live_tail(self, node: RadixTreeNode) -> bool:
        cur = node
        while not cur.is_root():
            if not (
                self._dsv4_node_has_independent_state(cur)
                or self._dsv4_node_has_live_tail(cur)
            ):
                return False
            cur = cur.parent
        return True

    def _dsv4_node_has_independent_state(self, node: RadixTreeNode) -> bool:
        handles = node._dsv4_component_pages
        return bool(handles is not None and handles.has_required_state_pages)

    def _dsv4_node_has_live_tail(self, node: RadixTreeNode) -> bool:
        if self.dsv4_swa_independent_lifecycle_enabled:
            return bool(node._dsv4_swa_pages is not None and node._dsv4_swa_pages.has_live_tail)
        if node.length <= 0 or len(node.value) < self.page_size:
            return False
        tail = node.value[-self.page_size :]
        return bool(torch.all(tail >= 0).item())

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
