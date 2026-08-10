#!/usr/bin/env python3
"""
Fix incorrect .data_read<T>() assignments to non-const pointers.
Pattern:  Type* [CT_RESTRICT] var = expr.data_read<T>();
Replace:  Type* [CT_RESTRICT] var = expr.data_write<T>();

Also fixes direct indexed writes that were mis-classified as reads:
    expr.data_read<T>()[i] = ...  ->  expr.data_write<T>()[i] = ...
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = [
    "src/kernels/CPU-BASIC",
    "src/kernels/CPU-SIMD",
    "src/kernels/AMX",
    "src/kernels/MPS",
    "src/AutoGrad",
    "src/tests",
    "include",
]
SKIP_DIRS = {"backup_threadpool", ".git", "build", "build-llama", "scripts"}


def collect_files() -> list[Path]:
    files = []
    for target in TARGETS:
        tpath = REPO / target
        if tpath.is_file():
            files.append(tpath)
        elif tpath.is_dir():
            for p in tpath.rglob("*"):
                if p.suffix in {".cpp", ".h", ".hpp", ".mm"}:
                    if any(part in SKIP_DIRS for part in p.parts):
                        continue
                    files.append(p)
    return sorted(set(files))


def fix_line(line: str) -> str:
    # Direct indexed write: X.data_read<T>()[i] = ...
    line = re.sub(
        r"(\.data_read\s*<[^>]+>\s*\(\)\s*\[[^\]]+\])\s*=",
        lambda m: m.group(1).replace("data_read", "data_write") + " =",
        line,
    )

    # Non-const pointer declaration with .data_read<T>() RHS.
    # Capture:  Type* [restrict] var = <expr>.data_read<T>();
    # Reject if the declaration is const-qualified.
    def repl(m):
        decl = m.group(1)
        # If 'const' appears before the '*' in the declaration, leave it as data_read.
        star_pos = decl.find("*")
        prefix = decl[:star_pos] if star_pos != -1 else decl
        if re.search(r"\bconst\b", prefix):
            return m.group(0)
        return f"{decl}{m.group(2)}.data_write<{m.group(3)}>()"

    line = re.sub(
        r"(\b(?:const\s+)?\w+\s*\*(?:\s+\w+)?\s*\w+\s*=\s*)((?:[^;])*?)\.data_read\s*<([^>]+)>\s*\(\)",
        repl,
        line,
    )
    return line


def fix_file(fpath: Path) -> bool:
    text = fpath.read_text(encoding="utf-8")
    new_lines = [fix_line(line) for line in text.splitlines()]
    new_text = "\n".join(new_lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"
    if new_text != text:
        fpath.write_text(new_text, encoding="utf-8")
        return True
    return False


def main():
    fixed = []
    for fpath in collect_files():
        if fix_file(fpath):
            fixed.append(str(fpath.relative_to(REPO)))
    print("Fixed files:")
    for p in fixed:
        print(f"  {p}")


if __name__ == "__main__":
    main()
