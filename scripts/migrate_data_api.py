#!/usr/bin/env python3
"""
Mechanical migration: Tensor::data<T>() -> Tensor::data_read<T>() / data_write<T>().

Rules:
- Storage::data<T>() (e.g. _storage.data<T>()) is NOT touched.
- const T*  p = X.data<T>()  ->  X.data_read<T>()
- T*        p = X.data<T>()  ->  X.data_write<T>()
- Direct element access:
    X.data<T>()[...] on LHS of assignment / memset target -> data_write
    otherwise -> data_read
- Other call sites are left for manual review (logged).

Run from repo root:
    python3 scripts/migrate_data_api.py
"""

import re
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parent.parent

# Files/directories to process.  We focus on core library + tests + examples.
TARGETS = [
    "include/Tensor.h",
    "src/Tensor.cpp",
    "src/AutoGrad",
    "src/kernels/CPU-BASIC",
    "src/kernels/CPU-SIMD",
    "src/kernels/MPS",
    "src/kernels/AMX",
    "src/tests",
    "mnist",
    "test_strides.cpp",
]

# Skip backup/old code that is not part of active build.
SKIP_DIRS = {
    "backup_threadpool",
    ".git",
    "build",
    "build-llama",
    "scripts",
}

DATA_RE = re.compile(r"(?<!_storage)\.data\s*<([^>]+)>\s*\(\)")


def should_process(path: Path) -> bool:
    if path.suffix not in {".cpp", ".h", ".hpp", ".mm"}:
        return False
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return True


def is_storage_data(expr: str) -> bool:
    return "_storage.data" in expr or "storage().data" in expr


def classify_line(line: str) -> Optional[str]:
    """
    Return 'read', 'write', or None for a line containing .data<T>().
    None means the script cannot decide automatically.
    """
    # Variable declaration: const T* [CT_RESTRICT] p = X.data<T>()
    if re.search(r"\bconst\s+\w+\s*\*(?:\s+\w+)?\s*\w+\s*=\s*[^;]*?\.data\s*<", line):
        return "read"
    # Variable declaration: T* [CT_RESTRICT] p = X.data<T>()
    if re.search(r"\b\w+\s*\*(?:\s+\w+)?\s*\w+\s*=\s*[^;]*?\.data\s*<", line):
        return "write"

    # Direct indexed access X.data<T>()[i] = ...  -> write
    if re.search(r"\.data\s*<[^>]+>\s*\(\)\s*\[[^\]]+\]\s*=", line):
        return "write"
    # std::memset(X.data<T>(), ...) -> write
    if re.search(r"std::memset\s*\(\s*[^,]*\.data\s*<", line):
        return "write"

    # Direct indexed read: X.data<T>()[i] used anywhere except LHS -> read
    if re.search(r"\.data\s*<[^>]+>\s*\(\)\s*\[[^\]]+\]", line):
        return "read"

    # std::copy / std::memset destination -> write
    if re.search(r"std::(?:copy|memset)\s*\(\s*[^,]*\.data\s*<", line):
        return "write"

    # Any remaining bare .data<T>() call site is treated as read by default.
    # This covers EXPECT_EQ, near, printf, std::cout, function arguments, etc.
    if re.search(r"\.data\s*<[^>]+>\s*\(\)", line):
        return "read"

    return None


def migrate_line(line: str) -> Tuple[str, bool]:
    """
    Migrate a single line. Returns (new_line, modified).
    If classification fails, logs and returns original line.
    """
    if not DATA_RE.search(line):
        return line, False
    if is_storage_data(line):
        return line, False

    kind = classify_line(line)
    if kind is None:
        return line, False

    repl = ".data_read<\\1>()" if kind == "read" else ".data_write<\\1>()"
    new_line = DATA_RE.sub(repl, line)
    return new_line, new_line != line


def collect_files() -> list[Path]:
    files = []
    for target in TARGETS:
        tpath = REPO / target
        if tpath.is_file():
            files.append(tpath)
        elif tpath.is_dir():
            for p in tpath.rglob("*"):
                if should_process(p):
                    files.append(p)
    return sorted(set(files))


def main():
    files = collect_files()
    modified_files = []
    uncertain = []

    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        new_lines = []
        changed = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            new_line, line_changed = migrate_line(line)
            new_lines.append(new_line)
            if line_changed:
                changed = True
            elif DATA_RE.search(line) and not is_storage_data(line) and classify_line(line) is None:
                uncertain.append(f"{fpath}:{lineno}: {line.strip()}")
        if changed:
            fpath.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
            modified_files.append(str(fpath.relative_to(REPO)))

    print("Modified files:")
    for p in modified_files:
        print(f"  {p}")

    if uncertain:
        print("\nUncertain .data<T>() calls (manual review needed):")
        for u in uncertain:
            print(f"  {u}")


if __name__ == "__main__":
    main()
