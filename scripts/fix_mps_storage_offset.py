#!/usr/bin/env python3
"""Batch-fix MPS kernel dispatch to pass storage base pointer + byte offset.

Root cause: MPSAllocator::bufferMap keys are MTLBuffer.contents base pointers.
Tensor::data<float>() returns _storage.data<float>() + _storage_offset, so
MPS_getBuffer() fails for views (storage_offset > 0). Even if it succeeded,
[encoder setBuffer:... offset:0] would read from the buffer base instead of
the view's offset.

Fix: Use t.storage().data<float>() (base pointer) for buffer lookup and pass
byte_offset = storage_offset() * dtypeSize(dtype()) to setBuffer:offset:.
"""

import re
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "src" / "kernels" / "MPS" / "MPS_kernel_dispatch.mm"

if not FILE.exists():
    print(f"ERROR: {FILE} not found", file=sys.stderr)
    sys.exit(1)

content = FILE.read_text()
lines = content.splitlines(keepends=True)

# Pass 1: collect prefixes for which we will generate _offset variables.
prefixes = set()

buf_re = re.compile(
    r'^(\s*)id<MTLBuffer>\s+(\w+)_buffer\s*=\s*MPS_getBuffer\('
    r'const_cast<void\*>\(static_cast<const void\*>\((\w+)\.data<float>\(\)\)\)\);'
)
ptr_re = re.compile(
    r'^(\s*)void\*\s+(\w+)_ptr\s*=\s*const_cast<void\*>\('
    r'static_cast<const void\*>\((\w+)\.data<float>\(\)\)\);'
)

for line in lines:
    m = buf_re.match(line)
    if m:
        prefixes.add(m.group(2))
    m = ptr_re.match(line)
    if m:
        prefixes.add(m.group(2))

print(f"[INFO] Collected {len(prefixes)} buffer/ptr prefixes to patch: {sorted(prefixes)}")

# Pass 2: rewrite lines.
new_lines = []
setbuf_re = re.compile(
    r'^(\s*)\[encoder\s+setBuffer:(\w+)_buffer\s+offset:0\s+atIndex:(\d+)\];'
)

for line in lines:
    m = buf_re.match(line)
    if m:
        indent, prefix, tensor = m.groups()
        new_lines.append(
            f"{indent}size_t {prefix}_offset = {tensor}.storage_offset() * dtypeSize({tensor}.dtype());\n"
        )
        new_lines.append(
            f"{indent}id<MTLBuffer> {prefix}_buffer = MPS_getBuffer("
            f"const_cast<void*>(static_cast<const void*>({tensor}.storage().data<float>())));\n"
        )
        continue

    m = ptr_re.match(line)
    if m:
        indent, prefix, tensor = m.groups()
        new_lines.append(
            f"{indent}size_t {prefix}_offset = {tensor}.storage_offset() * dtypeSize({tensor}.dtype());\n"
        )
        new_lines.append(
            f"{indent}void* {prefix}_ptr = const_cast<void*>("
            f"static_cast<const void*>({tensor}.storage().data<float>()));\n"
        )
        continue

    m = setbuf_re.match(line)
    if m and m.group(2) in prefixes:
        indent, prefix, index = m.groups()
        new_lines.append(
            f"{indent}[encoder setBuffer:{prefix}_buffer offset:{prefix}_offset atIndex:{index}];\n"
        )
        continue

    new_lines.append(line)

FILE.write_text(''.join(new_lines))
print(f"[INFO] Wrote patched {FILE}")
