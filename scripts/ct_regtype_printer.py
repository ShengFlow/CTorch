"""
GDB pretty-printer for ct::tl::vec::x86::RegType<ElemType, N>

Makes GDB display SIMD register types (RegType / WrapperType) as arrays
of their logical element type, e.g.:

    RegType<int16_t, 16>  -->  [0, 1, -1, 42, ...]   (16 elements)
    RegType<float32_t, 8> -->  [1.0, 2.0, 3.5, ...]  (8 elements)

Supports all element types defined by TL_DEFINE_MMREG:
  bfloat16_t, float16_t, float32_t, float64_t,
  int8_t, uint8_t, int16_t, uint16_t,
  int32_t, uint32_t, int64_t, uint64_t

Usage:
  This module is auto-loaded by GDB via the <executable>-gdb.py mechanism.
  It is registered through CMake: see CMakeLists.txt add_multiarch_executable().
"""

import re
import gdb
import gdb.printing

# ---------------------------------------------------------------------------
# Element type registry
# ---------------------------------------------------------------------------
# Map the C++ typedef name (as it appears in the template parameter) to
#   (gdb_type_name, size_in_bytes, is_float)
#
# These names must match what GDB prints in template arguments. For types
# that are namespace-qualified in the source (e.g. ct::int16_t), GDB may
# print either the short form or the fully-qualified form depending on
# typedef stripping.  We handle both.

_ELEM_TABLE = {
    # ---- C++ typedef name ----  (GDB lookup name)       bytes  float?
    'bfloat16_t':              ('unsigned short',           2,  False),
    'ct::bfloat16_t':          ('unsigned short',           2,  False),
    'float16_t':               ('unsigned short',           2,  False),
    'ct::float16_t':           ('unsigned short',           2,  False),
    'float32_t':               ('float',                    4,  True),
    'ct::float32_t':           ('float',                    4,  True),
    'float64_t':               ('double',                   8,  True),
    'ct::float64_t':           ('double',                   8,  True),
    'int8_t':                  ('signed char',              1,  False),
    'ct::int8_t':              ('signed char',              1,  False),
    'uint8_t':                 ('unsigned char',            1,  False),
    'ct::uint8_t':             ('unsigned char',            1,  False),
    'int16_t':                 ('short',                    2,  False),
    'ct::int16_t':             ('short',                    2,  False),
    'uint16_t':                ('unsigned short',           2,  False),
    'ct::uint16_t':            ('unsigned short',           2,  False),
    'int32_t':                 ('int',                      4,  False),
    'ct::int32_t':             ('int',                      4,  False),
    'uint32_t':                ('unsigned int',             4,  False),
    'ct::uint32_t':            ('unsigned int',             4,  False),
    'int64_t':                 ('long long',                8,  False),
    'ct::int64_t':             ('long long',                8,  False),
    'uint64_t':                ('unsigned long long',       8,  False),
    'ct::uint64_t':            ('unsigned long long',       8,  False),
}

# Inverse map: gdb_type_name -> (c_typedef_name, bytes, is_float) for display
_GDB_TO_TYPEDEF = {
    'unsigned short':  ('bfloat16_t', 2, False),
    'float':           ('float32_t',  4, True),
    'double':          ('float64_t',  8, True),
    'signed char':     ('int8_t',     1, False),
    'unsigned char':   ('uint8_t',    1, False),
    'short':           ('int16_t',    2, False),
    'unsigned short':  ('uint16_t',   2, False),
    'int':             ('int32_t',    4, False),
    'unsigned int':    ('uint32_t',   4, False),
    'long long':       ('int64_t',    8, False),
    'unsigned long long': ('uint64_t', 8, False),
}


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------
class RegTypePrinter:
    """
    Pretty-printer for ct::tl::vec::x86::RegType<ElemType, N>.

    The template parameter N is the number of logical elements.
    We extract the element type name from the template args and cast the
    underlying intrinsic vector ('v') to an array of that element type.
    """

    def __init__(self, val):
        self._val = val
        self._elem_gdb_type = None
        self._is_float = False
        self._nelems = 0
        self._tag_name = None   # e.g. "int16_t"
        self._init()

    # ------------------------------------------------------------------
    def _init(self):
        tag = self._val.type.strip_typedefs().tag
        if not tag:
            return

        # Match RegType<..., N>  where ... is the element type name
        m = re.search(r'RegType\s*<\s*(.+?)\s*,\s*(\d+)\s*>', tag)
        if not m:
            return

        elem_name = m.group(1).strip()
        self._nelems = int(m.group(2))

        # Resolve element type
        info = _ELEM_TABLE.get(elem_name)
        if info is None:
            # Try stripping 'ct::' if present (or adding it if not)
            if elem_name.startswith('ct::'):
                info = _ELEM_TABLE.get(elem_name[4:])
            else:
                info = _ELEM_TABLE.get('ct::' + elem_name)

        if info is None:
            # Last resort: try gdb.lookup_type directly
            try:
                t = gdb.lookup_type(elem_name)
                self._elem_gdb_type = t
                self._is_float = (t.code == gdb.TYPE_CODE_FLT)
                self._tag_name = elem_name
            except gdb.error:
                pass
            return

        gdb_name, _, is_float = info
        try:
            self._elem_gdb_type = gdb.lookup_type(gdb_name)
            self._is_float = is_float
            self._tag_name = gdb_name
        except gdb.error:
            pass

    # ------------------------------------------------------------------
    def _get_v(self):
        """Access the 'v' member (may be in base class WrapperType)."""
        try:
            return self._val['v']
        except gdb.error:
            pass
        # Fallback: walk base classes
        try:
            typ = self._val.type.strip_typedefs()
            for field in typ.fields():
                if field.is_base_class:
                    try:
                        return self._val.cast(field.type)['v']
                    except gdb.error:
                        continue
        except gdb.error:
            pass
        return None

    # ------------------------------------------------------------------
    def _cast_to_array(self, v_val):
        """Cast the intrinsic vector to an element-type array."""
        return v_val.cast(self._elem_gdb_type.array(self._nelems))

    # ------------------------------------------------------------------
    def _format_elem(self, elem_val):
        if self._is_float:
            return f'{float(elem_val):.6g}'
        else:
            return str(int(elem_val))

    # ------------------------------------------------------------------
    def to_string(self):
        if self._elem_gdb_type is None or self._nelems == 0:
            return str(self._val)

        v = self._get_v()
        if v is None:
            return str(self._val)

        try:
            arr = self._cast_to_array(v)
            # Summary: show first few elements, truncate if too long
            if self._nelems <= 8:
                parts = [self._format_elem(arr[i]) for i in range(self._nelems)]
            else:
                parts = [self._format_elem(arr[i]) for i in range(6)]
                parts.append('...')
                parts.append(self._format_elem(arr[self._nelems - 1]))
            return '[' + ', '.join(parts) + ']'
        except Exception as e:
            return f'<RegType printer error: {e}>'

    # ------------------------------------------------------------------
    def children(self):
        if self._elem_gdb_type is None or self._nelems == 0:
            return

        v = self._get_v()
        if v is None:
            return

        try:
            arr = self._cast_to_array(v)
            for i in range(self._nelems):
                yield (f'[{i}]', arr[i])
        except gdb.error:
            yield ('raw', v)

    # ------------------------------------------------------------------
    def display_hint(self):
        return 'array'


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_ctorch_printers(objfile=None):
    """Register all CTorch x86 RegType pretty-printers."""
    pp = gdb.printing.RegexpCollectionPrettyPrinter('CTorch')

    # Match ct::tl::vec::x86::RegType<anything, number>
    # The tag may or may not include namespace qualifiers on the element type
    pp.add_printer(
        'RegType',
        r'^ct::tl::vec::x86::RegType<.*,\s*\d+\s*>$',
        RegTypePrinter,
    )
    # Also match WrapperType directly (e.g. when viewed through base pointer)
    pp.add_printer(
        'WrapperType',
        r'^ct::tl::vec::x86::WrapperType<.*>$',
        RegTypePrinter,  # same printer — it just reads 'v'
    )

    gdb.printing.register_pretty_printer(objfile or gdb.current_objfile(), pp)


# Auto-register when this module is imported by the <exe>-gdb.py loader
register_ctorch_printers()
