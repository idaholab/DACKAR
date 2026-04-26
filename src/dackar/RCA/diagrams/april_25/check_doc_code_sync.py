#!/usr/bin/env python3
"""
Check Markdown doc anchors against the DACKAR RCA Python tree.

 Parses HTML comments of the form:
   <!-- @code: relative/path.py | ClassName.method -->
   <!-- @code: relative/path.py | _function_name -->
   <!-- @doc: ... | @reviewed: YYYY-MM-DD -->
   <!-- @schema: relative/path.json -->

 Paths are resolved under src/dackar/RCA/ (the RCA package root).
 Non-file anchors, e.g. (deployment) kg query layer, are reported as skip.
 Placeholder text like <path> in the doc (not a real file) is ignored.

Usage (DACKAR repo root):
  python src/dackar/RCA/diagrams/april_25/check_doc_code_sync.py
  python src/dackar/RCA/diagrams/april_25/check_doc_code_sync.py --stale
  (from this directory): python check_doc_code_sync.py
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

# Default relative to repository root: src/dackar/RCA
RCA_SUBPATH = Path("src") / "dackar" / "RCA"

# To closing `-->` (content may include `>` in generics or placeholders)
RE_CODE = re.compile(
    r"<!--\s*@code:\s*([\s\S]*?)--\s*>",
    re.IGNORECASE,
)
RE_SCHEMA = re.compile(
    r"<!--\s*@schema:\s*([\s\S]*?)--\s*>",
    re.IGNORECASE,
)
RE_REVIEWED_INLINE = re.compile(
    r"@reviewed:\s*(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
RE_PLACEHOLDER = re.compile(
    r"<path>|<class|YYYY-MM|placeholder",
    re.IGNORECASE,
)


@dataclass
class CodeAnchor:
    line: int
    raw: str
    code_path: str
    symbol: str
    full_comment: str

    @property
    def is_file_anchor(self) -> bool:
        p = (self.code_path or "").strip()
        if not p or p.startswith("(") or p.startswith("<"):
            return False
        if " " in p or "<" in p or ">" in p or RE_PLACEHOLDER.search(p):
            return False
        if p.endswith(".py") or p.endswith(".json"):
            if ".." in p or p.startswith(("/", "\\")):
                return False
            if any(c in p for c in "\t\n\r"):
                return False
            return True
        return False


def _parse_code_body(body: str) -> Tuple[str, str]:
    rest = body.strip()
    if "|" not in rest:
        return rest.strip(), ""
    left, right = rest.split("|", 1)
    right = right.strip()
    if right.lower().startswith("@status:"):
        return left.strip(), ""
    return left.strip(), right.strip()


def find_code_anchors(text: str) -> List[CodeAnchor]:
    out: List[CodeAnchor] = []
    for m in RE_CODE.finditer(text):
        line = text[: m.start()].count("\n") + 1
        full = m.group(0)
        path_part, sym = _parse_code_body(m.group(1))
        out.append(
            CodeAnchor(
                line=line,
                raw=m.group(1).strip(),
                code_path=path_part,
                symbol=sym.strip(),
                full_comment=full,
            )
        )
    return out


def find_review_dates(text: str) -> List[date]:
    dates: List[date] = []
    for m in RE_REVIEWED_INLINE.finditer(text):
        try:
            y, month, d = m.group(1).split("-")
            dates.append(date(int(y), int(month), int(d)))
        except ValueError:
            pass
    return dates


def _iter_classes_functions(
    body: List[ast.stmt],
) -> Generator[Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef], None, None]:
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _class_methods(
    cls: ast.ClassDef,
) -> Generator[Union[ast.FunctionDef, ast.AsyncFunctionDef], None, None]:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def symbol_in_module(tree: ast.Module, symbol: str) -> bool:
    if not symbol or symbol in ("@status: environment-specific",):
        return True
    parts = [p for p in symbol.split(".") if p]

    if len(parts) == 1:
        name = parts[0]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return True
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for m in _class_methods(node):
                    if m.name == name:
                        return True
        return False

    if len(parts) == 2:
        cls_name, mem = parts[0], parts[1]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for m in _class_methods(node):
                    if m.name == mem:
                        return True
                return False
        return False

    if len(parts) == 3:
        a, b, c = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == a:
                for inner in node.body:
                    if isinstance(inner, ast.ClassDef) and inner.name == b:
                        for m in _class_methods(inner):
                            if m.name == c:
                                return True
        return False

    return False


def check_python_symbol(py_path: Path, symbol: str) -> Tuple[bool, str]:
    if not symbol:
        return True, "no symbol; file presence only"
    try:
        source = py_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"read error: {e}"
    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    if symbol_in_module(tree, symbol):
        return True, f"symbol `{symbol}` found"
    return False, f"symbol `{symbol}` not found in {py_path.name}"


def _git_file_last_commit_iso(repo: Path, rel_file: Path) -> Optional[str]:
    try:
        cp = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "log",
                "-1",
                "--format=%cI",
                "--",
                str(rel_file).replace("\\", "/"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            return None
        return cp.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_git_iso(s: str) -> Optional[datetime]:
    s = s.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _relative_to_safe(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path


def _unwrap_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def resolve_paths(
    repo_root: Path,
    rca_root: Path,
    anchor: CodeAnchor,
) -> Tuple[Optional[Path], Optional[Path], str]:
    if not anchor.is_file_anchor:
        return None, None, "non-file anchor (documentation placeholder or illustration)"

    rel = Path(anchor.code_path.replace("\\", "/").lstrip("/"))
    c1 = rca_root / rel
    c2 = repo_root / rel
    if c1.is_file():
        abs_p = c1.resolve()
        return abs_p, _relative_to_safe(abs_p, repo_root), "resolved from RCA root"
    if c2.is_file():
        abs_p = c2.resolve()
        return abs_p, _relative_to_safe(abs_p, repo_root), "resolved from repo root"
    return None, rel, f"missing file: {c1}"


def run_checks(
    repo_root: Path,
    rca_root: Path,
    md_path: Path,
    *,
    check_stale: bool,
    quiet: bool = False,
) -> Tuple[int, int]:
    """Returns (exit_code, warning_count). exit_code: 0 ok, 1 warnings-only if strict, 2 errors."""
    text = md_path.read_text(encoding="utf-8")
    anchors = find_code_anchors(text)
    review_dates = find_review_dates(text)
    max_review = max(review_dates) if review_dates else None

    errors: List[str] = []
    warnings: List[str] = []
    oks: List[str] = []

    for m in RE_SCHEMA.finditer(text):
        line = text[: m.start()].count("\n") + 1
        raw = m.group(1).strip()
        rel_s = _unwrap_quotes(raw).replace("\\", "/").lstrip("/")
        if not rel_s or "<" in rel_s or ">" in rel_s or " " in rel_s:
            oks.append(
                f"  {md_path.name}:{line} @schema skip (illustration / placeholder): {raw[:50]!r}..."
            )
            continue
        rel = Path(rel_s)
        if not rel.suffix == ".json":
            warnings.append(
                f"  {md_path.name}:{line} @schema non-.json path {rel!r} - not checked on disk"
            )
            continue
        try:
            p = (rca_root / rel).resolve() if not rel.is_absolute() else rel.resolve()
        except OSError as e:
            errors.append(f"  {md_path.name}:{line} @schema bad path: {e}")
            continue
        if p.is_file():
            oks.append(f"  {md_path.name}:{line} @schema OK {rel}")
        else:
            errors.append(f"  {md_path.name}:{line} @schema missing: {p}")

    for a in anchors:
        abs_p, rel_to_repo, note = resolve_paths(repo_root, rca_root, a)
        if abs_p is None:
            oks.append(
                f"  {md_path.name}:{a.line} @code skip - {a.code_path!r} ({note})"
            )
            continue
        if abs_p.suffix == ".json":
            oks.append(
                f"  {md_path.name}:{a.line} @code OK (schema file) {rel_to_repo}"
            )
            continue
        if abs_p.suffix != ".py":
            warnings.append(
                f"  {md_path.name}:{a.line} @code unknown suffix {abs_p.suffix!r} for {rel_to_repo}"
            )
            continue

        ok, msg = check_python_symbol(abs_p, a.symbol)
        if ok:
            oks.append(
                f"  {md_path.name}:{a.line} @code OK {rel_to_repo} | {a.symbol!r} - {msg}"
            )
            if check_stale and max_review is not None and rel_to_repo is not None:
                rpath = rel_to_repo if isinstance(rel_to_repo, Path) else Path(rel_to_repo)
                iso = _git_file_last_commit_iso(repo_root, rpath)
                gdt = _parse_git_iso(iso) if iso else None
                if gdt and gdt.date() > max_review:
                    warnings.append(
                        f"  STALE: {rpath} last commit {gdt.date()} > doc @reviewed max {max_review} "
                        f"(line {a.line} in {md_path.name})"
                    )
        else:
            errors.append(
                f"  {md_path.name}:{a.line} @code FAIL {rel_to_repo} | {a.symbol!r} - {msg}"
            )

    if oks and not quiet:
        print("OK / skip:")
        for x in oks:
            print(x)
        print()
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)

    wcount = len(warnings)
    if errors:
        return 2, wcount
    return 0, wcount


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / ".git").is_dir() or (p / ".git").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.parent


def _default_markdown_path(script_path: Path, rca_root: Path) -> Path:
    colocated = script_path.parent / "rca_workflow_reference_guide_april_25.md"
    if colocated.is_file():
        return colocated
    return rca_root / "diagrams" / "april_25" / "rca_workflow_reference_guide_april_25.md"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify @code / @doc anchors in Markdown against RCA code."
    )
    ap.add_argument(
        "--markdown",
        type=Path,
        action="append",
        help="Markdown file to scan (repeatable). Default: co-located reference guide in april_25.",
    )
    ap.add_argument(
        "--rca-root",
        type=Path,
        help="Path to src/dackar/RCA. Default: <repo>/src/dackar/RCA",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        help="Git repository root. Default: walk up from this script to find .git",
    )
    ap.add_argument(
        "--stale",
        action="store_true",
        help="Warn when a referenced file's last git commit is after the max @reviewed date.",
    )
    ap.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Exit 1 if any warning was printed (e.g. STALE).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print problems (no OK / skip list).",
    )
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = (args.repo_root or _find_repo_root(script_path)).resolve()
    rca_root = (args.rca_root or (repo_root / RCA_SUBPATH)).resolve()

    if not rca_root.is_dir():
        print(f"ERROR: RCA root not found: {rca_root}", file=sys.stderr)
        return 2

    default_md = _default_markdown_path(script_path, rca_root)
    md_list = list(args.markdown) if args.markdown else [default_md]

    exit_code = 0
    total_warnings = 0
    for md in md_list:
        mp = Path(md).resolve()
        if not mp.is_file():
            print(f"ERROR: markdown not found: {mp}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"Scanning {_relative_to_safe(mp, repo_root)}")
        code, wcount = run_checks(
            repo_root,
            rca_root,
            mp,
            check_stale=args.stale,
            quiet=args.quiet,
        )
        total_warnings += wcount
        if code == 2:
            exit_code = 2
        elif code > exit_code:
            exit_code = code
    if args.strict_warnings and total_warnings > 0 and exit_code < 2:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
