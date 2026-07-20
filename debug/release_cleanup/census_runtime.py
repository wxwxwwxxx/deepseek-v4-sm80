#!/usr/bin/env python3
"""Build source-level evidence for the DSV4 release-cleanup census.

The helper deliberately lives outside :mod:`minisgl`.  It never imports the
production package and it does not mutate runtime state.  Its output is raw
evidence: release-default values, environment readers, Python call/attribute
edges, Triton ``kernel[grid](...)`` launches, custom-op/JIT registrations, and
native include/source ownership.  Policy is applied by milestone-specific
consumers rather than this read-only collector.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ENV_RE = re.compile(r"MINISGL_(?:DSV4|PYNCCL)_[A-Z0-9_]+")
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")
SOURCE_ROOTS = ("python", "benchmark", "tests")
SOURCE_SUFFIXES = {".py", ".cu", ".cpp", ".h", ".cuh", ".hpp"}
OPERATOR_FILES = (
    "python/minisgl/models/deepseek_v4.py",
    "python/minisgl/attention/deepseek_v4.py",
    "python/minisgl/kernel/deepseek_v4.py",
    "python/minisgl/kernel/triton/deepseek_v4.py",
    "python/minisgl/kernel/triton/fused_moe.py",
    "python/minisgl/kernel/marlin_wna16.py",
    "python/minisgl/kernel/moe_impl.py",
    "python/minisgl/kernel/pynccl.py",
)
RELEASE_DEFAULT_FILE = "python/minisgl/engine/engine.py"
NATIVE_ROOT = "python/minisgl/kernel/csrc"


def git_lines(root: Path, *args: str) -> list[str]:
    return subprocess.check_output(["git", *args], cwd=root, text=True).splitlines()


def tracked_files(root: Path) -> list[Path]:
    return [root / name for name in git_lines(root, "ls-files")]


def source_files(root: Path) -> list[Path]:
    return [
        path
        for path in tracked_files(root)
        if path.suffix in SOURCE_SUFFIXES and path.relative_to(root).parts[0] in SOURCE_ROOTS
    ]


def _safe_expr(node: ast.AST, names: dict[str, Any] | None = None) -> Any:
    """Evaluate the small literal/arithmetic subset used by Engine defaults."""

    names = names or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
        return ast.literal_eval(node)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"str", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _safe_expr(node.args[0], names)
        return str(value) if node.func.id == "str" else int(value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_expr(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _safe_expr(node.left, names)
        right = _safe_expr(node.right, names)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(f"unsupported literal expression: {ast.unparse(node)}")


def release_defaults(root: Path) -> dict[str, Any]:
    path = root / RELEASE_DEFAULT_FILE
    tree = ast.parse(path.read_text(), filename=RELEASE_DEFAULT_FILE)
    names: dict[str, Any] = {}
    result: dict[str, Any] = {}
    wanted = {
        "_DSV4_SM80_RELEASE_DEFAULT_ENV",
        "_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES",
        "_DSV4_SM80_DEFAULT_RECIPE",
        "_DSV4_SM80_DEFAULT_MAX_EXTEND_TOKENS",
        "_DSV4_SM80_RECIPES",
    }
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                value = _safe_expr(value_node, names)
            except (ValueError, TypeError, KeyError):
                continue
            names[target.id] = value
            if target.id in wanted:
                result[target.id] = value
    missing = sorted(wanted - result.keys())
    if missing:
        raise RuntimeError(f"could not resolve release defaults: {missing}")
    return result


def env_census(root: Path, files: Iterable[Path]) -> list[dict[str, object]]:
    occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    assignments: dict[str, list[dict[str, object]]] = defaultdict(list)
    readers: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in files:
        rel = str(path.relative_to(root))
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        for line_no, line in enumerate(lines, 1):
            for name in sorted(set(ENV_RE.findall(line))):
                # Prefix strings such as ``"MINISGL_DSV4_SM80_"`` are search
                # namespaces, not environment variables.
                if name.endswith("_"):
                    continue
                item = {"path": rel, "line": line_no, "text": line.strip()}
                occurrences[name].append(item)
                if re.search(rf"(?:^|\W)[A-Z0-9_]+\s*=\s*[\"']{re.escape(name)}[\"']", line):
                    assignments[name].append(item)

        # Resolve constant-backed readers.  Production commonly declares the
        # spelling once and later calls ``os.environ.get(CONSTANT, default)``
        # or a typed helper.  Same-line matching loses both the reader and its
        # default, especially for multi-line declarations.
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            continue
        names: dict[str, Any] = {}
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                try:
                    value = _safe_expr(node.value, names)
                except (ValueError, TypeError, KeyError):
                    continue
                names[target.id] = value
                if isinstance(value, str) and ENV_RE.fullmatch(value):
                    item = {
                        "path": rel,
                        "line": node.lineno,
                        "text": " ".join(
                            part.strip()
                            for part in lines[node.lineno - 1 : (node.end_lineno or node.lineno)]
                        ),
                    }
                    assignments[value].append(item)
                    occurrences[value].append(item)

        reader_suffixes = {
            "environ.get",
            "getenv",
            "dsv4_env_flag",
            "dsv4_sm80_triton_enabled",
            "dsv4_env_value",
            "env_truthy",
            "_env",
            "_env_bytes",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            api = expr_name(node.func)
            if not any(api == suffix or api.endswith(f".{suffix}") for suffix in reader_suffixes):
                continue
            try:
                name = _safe_expr(node.args[0], names)
            except (ValueError, TypeError, KeyError):
                continue
            if not isinstance(name, str) or not ENV_RE.fullmatch(name):
                continue
            default: Any = None
            if len(node.args) > 1:
                try:
                    default = _safe_expr(node.args[1], names)
                except (ValueError, TypeError, KeyError):
                    default = ast.unparse(node.args[1])
            item = {
                "path": rel,
                "line": node.lineno,
                "text": lines[node.lineno - 1].strip(),
                "reader_api": api,
                "default": default,
            }
            readers[name].append(item)
            occurrences[name].append(item)

    def dedupe(items: list[dict[str, object]]) -> list[dict[str, object]]:
        return list({(item["path"], item["line"], item["text"]): item for item in items}.values())

    result = []
    for name in sorted(occurrences):
        all_occurrences = dedupe(occurrences[name])
        runtime = [item for item in all_occurrences if item["path"].startswith("python/")]
        runtime_readers = [item for item in readers[name] if item["path"].startswith("python/")]
        result.append(
            {
                "name": name,
                "definitions": dedupe(assignments[name]),
                "runtime_readers": dedupe(runtime_readers),
                "runtime_occurrences": runtime,
                "test_benchmark_occurrence_count": sum(
                    not item["path"].startswith("python/") for item in all_occurrences
                ),
            }
        )
    return result


def expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return expr_name(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


class PythonEvidenceVisitor(ast.NodeVisitor):
    """Collect definitions plus calls without losing their lexical owner."""

    def __init__(self, path: str):
        self.path = path
        self.stack: list[str] = []
        self.node_stack: list[dict[str, object]] = []
        self.items: list[dict[str, object]] = []
        self.launches: list[dict[str, object]] = []
        self.dynamic_edges: list[dict[str, object]] = []

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.stack, node.name])
        decorators = [ast.unparse(item) for item in node.decorator_list]
        item: dict[str, object] = {
            "path": self.path,
            "line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "qualname": qualname,
            "symbol": node.name,
            "kind": "method" if self.stack else "function",
            "is_private": node.name.startswith("_"),
            "decorators": decorators,
            "is_property": any(value.endswith("property") for value in decorators),
            "is_triton_jit": any("triton.jit" in value for value in decorators),
            "calls": [],
            "attribute_references": [],
        }
        self.items.append(item)
        self.stack.append(node.name)
        self.node_stack.append(item)
        for child in node.body:
            self.visit(child)
        self.node_stack.pop()
        self.stack.pop()

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join([*self.stack, node.name])
        decorators = [ast.unparse(item) for item in node.decorator_list]
        item: dict[str, object] = {
            "path": self.path,
            "line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "qualname": qualname,
            "symbol": node.name,
            "kind": "class",
            "is_private": node.name.startswith("_"),
            "decorators": decorators,
            "is_property": False,
            "is_triton_jit": False,
            "calls": [],
            "attribute_references": [],
        }
        self.items.append(item)
        self.stack.append(node.name)
        self.node_stack.append(item)
        for base in node.bases:
            self.visit(base)
        for child in node.body:
            self.visit(child)
        self.node_stack.pop()
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        owner = self.node_stack[-1] if self.node_stack else None
        launch_style = "ordinary"
        grid = None
        target_node = node.func
        if isinstance(node.func, ast.Subscript):
            launch_style = "triton_grid"
            target_node = node.func.value
            grid = ast.unparse(node.func.slice)
        target = expr_name(target_node)
        edge = {
            "path": self.path,
            "line": node.lineno,
            "owner": None if owner is None else owner["qualname"],
            "target": target,
            "target_symbol": target.rsplit(".", 1)[-1],
            "syntax": launch_style,
            "grid": grid,
        }
        if owner is not None:
            owner["calls"].append(edge)
        if launch_style == "triton_grid":
            self.launches.append(edge)
        if target in {"import_module", "load_jit", "load", "load_inline"} or target.endswith(
            (".import_module", ".load_jit", ".load_inline")
        ):
            self.dynamic_edges.append(edge)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.node_stack and isinstance(node.ctx, ast.Load):
            refs = self.node_stack[-1]["attribute_references"]
            assert isinstance(refs, list)
            refs.append(
                {
                    "path": self.path,
                    "line": node.lineno,
                    "name": expr_name(node),
                    "symbol": node.attr,
                }
            )
        self.generic_visit(node)


def python_evidence(root: Path, files: Iterable[Path]) -> dict[str, object]:
    source_paths = {str(path.relative_to(root)): path for path in files if path.suffix == ".py"}
    definitions: list[dict[str, object]] = []
    launches: list[dict[str, object]] = []
    dynamic_edges: list[dict[str, object]] = []
    for rel in OPERATOR_FILES:
        path = root / rel
        visitor = PythonEvidenceVisitor(rel)
        visitor.visit(ast.parse(path.read_text(), filename=rel))
        definitions.extend(visitor.items)
        launches.extend(visitor.launches)
        dynamic_edges.extend(visitor.dynamic_edges)

    # Resolve ordinary calls, properties, aliases, and tests by final symbol.
    sites_by_symbol: dict[str, list[dict[str, object]]] = defaultdict(list)
    attributes_by_symbol: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rel, path in source_paths.items():
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target_node = node.func.value if isinstance(node.func, ast.Subscript) else node.func
                target = expr_name(target_node)
                sites_by_symbol[target.rsplit(".", 1)[-1]].append(
                    {"path": rel, "line": node.lineno, "target": target}
                )
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                attributes_by_symbol[node.attr].append(
                    {"path": rel, "line": node.lineno, "target": expr_name(node)}
                )
    for item in definitions:
        symbol = str(item["symbol"])
        call_sites = sites_by_symbol[symbol]
        attribute_sites = attributes_by_symbol[symbol]
        item["resolved_call_sites"] = call_sites
        item["resolved_attribute_sites"] = attribute_sites
        item["runtime_call_site_count"] = sum(
            site["path"].startswith("python/")
            and not (site["path"] == item["path"] and site["line"] == item["line"])
            for site in call_sites
        )
        item["test_call_site_count"] = sum(site["path"].startswith("tests/") for site in call_sites)
        item["runtime_attribute_site_count"] = sum(
            site["path"].startswith("python/") for site in attribute_sites
        )
        # Deduplicate noisy attribute references kept on the owner itself.
        refs = item.pop("attribute_references")
        item["attribute_references"] = list(
            {(ref["path"], ref["line"], ref["name"]): ref for ref in refs}.values()
        )
    return {
        "definitions": definitions,
        "triton_grid_launches": launches,
        "dynamic_loader_edges": dynamic_edges,
    }


def native_evidence(root: Path) -> dict[str, object]:
    rel_paths = git_lines(root, "ls-files", NATIVE_ROOT)
    entries: list[dict[str, object]] = []
    registrations: list[dict[str, object]] = []
    registration_tokens = (
        "TORCH_LIBRARY",
        "TORCH_LIBRARY_IMPL",
        "REGISTER_EXTENSION",
        "TVM_FFI",
        "TVM_REGISTER",
        ".def(",
        ".impl(",
    )
    for rel in rel_paths:
        path = root / rel
        text = path.read_text(errors="replace")
        includes = [
            match.group(1)
            for line in text.splitlines()
            if (match := INCLUDE_RE.match(line)) is not None
        ]
        entries.append(
            {
                "path": rel,
                "suffix": path.suffix,
                "includes": sorted(set(includes)),
                "is_translation_unit": path.suffix in {".cu", ".cpp"},
            }
        )
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(token in line for token in registration_tokens):
                registrations.append({"path": rel, "line": line_no, "text": line.strip()})

    python_registrations: list[dict[str, object]] = []
    for rel in git_lines(root, "ls-files", "python/minisgl/kernel/*.py"):
        path = root / rel
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if any(
                token in line
                for token in (
                    "load_jit(",
                    "load_aot(",
                    "cpp_extension import load",
                    "register_object(",
                    "torch.ops",
                    ".glob(",
                )
            ):
                python_registrations.append({"path": rel, "line": line_no, "text": line.strip()})
    return {
        "sources": entries,
        "native_registrations": registrations,
        "python_loader_registrations": python_registrations,
    }


def build_census(root: Path) -> dict[str, object]:
    root = root.resolve()
    files = source_files(root)
    python = python_evidence(root, files)
    native = native_evidence(root)
    return {
        "schema_version": 2,
        "method": (
            "tracked-source AST/regex census with typed release-default values, "
            "ordinary/property calls, Triton grid launches, dynamic loaders, "
            "custom registrations, and native include edges"
        ),
        "release_defaults": release_defaults(root),
        "env": env_census(root, files),
        "callables": python["definitions"],
        "triton_grid_launches": python["triton_grid_launches"],
        "dynamic_loader_edges": python["dynamic_loader_edges"],
        "native_sources": native["sources"],
        "native_registrations": native["native_registrations"],
        "python_loader_registrations": native["python_loader_registrations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_census(args.root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
