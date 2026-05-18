"""Quick production-quality sweep: AST parse all .py files, find undefined names.

Run with::

    python scripts/quality_check.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    issues: list[str] = []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return [f"SYNTAX: {exc.lineno}:{exc.offset}: {exc.msg}"]

    # Collect module-level imports + function-local imports.
    module_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                module_imports.add((n.asname or n.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                module_imports.add(n.asname or n.name)

    # Trivial check: any bare-name "torch.xxx" use where torch isn't imported at top level.
    suspicious = {"torch"}  # heaviest offender in this codebase
    for sus in suspicious:
        if sus in module_imports:
            continue
        # Walk and find Name(id=sus) used outside of any FunctionDef that imports it.
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == sus:
                # Check parents — naive: scan all FunctionDefs and see if `sus` is in their local imports.
                # If not, flag.
                covered = False
                for fn in ast.walk(tree):
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not (fn.lineno <= node.lineno <= (fn.end_lineno or 1e9)):
                            continue
                        local_imports: set[str] = set()
                        for sub in ast.walk(fn):
                            if isinstance(sub, ast.Import):
                                for n in sub.names:
                                    local_imports.add((n.asname or n.name).split(".")[0])
                            elif isinstance(sub, ast.ImportFrom):
                                for n in sub.names:
                                    local_imports.add(n.asname or n.name)
                        if sus in local_imports:
                            covered = True
                            break
                if not covered:
                    issues.append(f"L{node.lineno}: uses `{sus}.{node.attr}` but `{sus}` not imported at module level or in enclosing function")
    return issues


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    targets = list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")) + list((root / "app").rglob("*.py"))
    targets = [p for p in targets if "__pycache__" not in p.parts]
    print(f"Scanning {len(targets)} python files...")
    bad = 0
    for p in sorted(targets):
        issues = check_file(p)
        if issues:
            bad += 1
            rel = p.relative_to(root)
            print(f"\n[{rel}]")
            for it in issues[:10]:
                print(f"  {it}")
    print(f"\nDone. Files with issues: {bad}/{len(targets)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
