"""std140 / std430 uniform-block layout, padding and packing exercise.

A uniform field is a mapping with exactly these logical keys::

    {"name": "view", "type": "mat4", "count": 1}

``name`` is unique inside one structure.  ``type`` is one of the base type
names below, or a list of field mappings for an inline nested structure.
``count`` is a positive element count; ``1`` means a single value and a value
greater than ``1`` means an array.  ``count`` may be omitted and defaults to
``1``.

Nested structure example::

    {
        "name": "light",
        "type": [
            {"name": "position", "type": "vec3", "count": 1},
            {"name": "energy", "type": "float", "count": 1},
        ],
        "count": 1,
    }

:func:`layout_struct` returns::

    {
        "offsets": {"field name": byte offset, ...},  # top-level fields only
        "size": int,    # used bytes, ending at the last field's storage
        "stride": int,  # bytes per structure instance, including tail padding
    }

The standard-layout rules used by this exercise are:

* Base sizes: ``float`` and ``int`` are 4 bytes, ``vec2`` 8, ``vec3`` 12,
  ``vec4`` 16, ``mat3`` 48 and ``mat4`` 64.  A ``mat3`` is three 16-byte
  columns and a ``mat4`` is four 16-byte columns.
* A ``vec3`` has a base alignment of 16, so it is stored in the same space as
  a ``vec4`` even though only 12 bytes are meaningful.
* ``mat3`` and ``mat4`` are treated as arrays of column vectors.  Every
  column starts on a 16-byte boundary: offsets 0, 16, 32 for ``mat3`` and
  0, 16, 32, 48 for ``mat4``.
* ``std140``: scalar alignment is 4, ``vec2`` 8, and ``vec3``/``vec4`` 16.
  A structure's alignment is 16, so its stride is a multiple of 16.
* ``std430``: scalar/vector/matrix base alignments are the same as
  ``std140`` (including a 16-byte ``vec3`` alignment), but a structure is
  aligned to its largest member alignment instead of always 16.  Thus
  ``[float, float]`` has stride 8 under ``std430`` and 16 under ``std140``.
* Arrays always use an element stride rounded up to a multiple of 16 under
  both rules (the exercise's explicit convention).  An array field itself
  starts on a 16-byte boundary.  The last array element's slot is fully
  counted, including its tail padding.
* An inline nested structure occupies its own ``stride``; when nested fields
  are arrays, that stride is again rounded up to 16.

All public operations report contract violations with :class:`LayoutError`.
The module uses only the Python standard library and never reads the network,
the clock or random state.
"""

import struct
from collections.abc import Mapping, Sequence


class LayoutError(Exception):
    """Raised for an invalid uniform layout, type, count, offset or rule."""


_BASE_SIZES = {
    "float": 4,
    "int": 4,
    "vec2": 8,
    "vec3": 12,
    "vec4": 16,
    "mat3": 48,
    "mat4": 64,
}

_BASE_ALIGNMENTS = {
    "float": 4,
    "int": 4,
    "vec2": 8,
    "vec3": 16,
    "vec4": 16,
    "mat3": 16,
    "mat4": 16,
}

_RULES = ("std140", "std430")

_VECTOR_COMPONENTS = {"vec2": 2, "vec3": 3, "vec4": 4}
_MATRIX_COLUMNS = {"mat3": 3, "mat4": 4}


def _validate_rule(rule):
    if rule not in _RULES:
        raise LayoutError(f"unknown layout rule: {rule!r}")


def _round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _field_info(field, rule):
    """Validate one field and return its placement facts.

    Returns a dict with ``name``, ``count``, ``align`` (base alignment of the
    field as a whole), ``size`` (bytes consumed, including array tail
    padding), ``elem_stride`` (per-element stride for arrays, else ``None``)
    and ``nested`` (the nested field list for inline structures, else
    ``None``).
    """
    if not isinstance(field, Mapping):
        raise LayoutError(f"field must be a mapping, got {type(field).__name__}")

    name = field.get("name")
    if not isinstance(name, str) or not name:
        raise LayoutError(f"field name must be a non-empty string, got {name!r}")

    count = field.get("count", 1)
    if not _is_positive_int(count):
        raise LayoutError(f"field {name!r}: count must be a positive integer, got {count!r}")

    ftype = field.get("type")
    info = {"name": name, "count": count, "nested": None, "elem_stride": None}

    if isinstance(ftype, str):
        if ftype not in _BASE_SIZES:
            raise LayoutError(f"field {name!r}: unknown type {ftype!r}")
        if count == 1:
            info["align"] = _BASE_ALIGNMENTS[ftype]
            info["size"] = _BASE_SIZES[ftype]
        else:
            info["align"] = 16
            info["elem_stride"] = _round_up(_BASE_SIZES[ftype], 16)
            info["size"] = info["elem_stride"] * count
        info["base_type"] = ftype
    elif isinstance(ftype, Sequence) and not isinstance(ftype, (str, bytes, bytearray)):
        if not ftype:
            raise LayoutError(f"field {name!r}: nested structure needs at least one field")
        _, _, sub_stride, sub_align = _layout(list(ftype), rule)
        info["nested"] = list(ftype)
        if count == 1:
            info["align"] = sub_align
            info["size"] = sub_stride
        else:
            info["align"] = 16
            info["elem_stride"] = _round_up(sub_stride, 16)
            info["size"] = info["elem_stride"] * count
        info["base_type"] = None
    else:
        raise LayoutError(f"field {name!r}: unknown type {ftype!r}")

    return info


def _layout(fields, rule):
    """Core layout engine: returns (offsets, size, stride, struct_alignment)."""
    _validate_rule(rule)
    if not isinstance(fields, (list, tuple)):
        raise LayoutError(f"fields must be a list, got {type(fields).__name__}")

    offsets = {}
    member_alignments = []
    cursor = 0
    for field in fields:
        info = _field_info(field, rule)
        name = info["name"]
        if name in offsets:
            raise LayoutError(f"duplicate field name: {name!r}")
        offset = _round_up(cursor, info["align"])
        offsets[name] = offset
        cursor = offset + info["size"]
        member_alignments.append(info["align"])

    size = cursor
    if not member_alignments:
        return offsets, 0, 0, 0

    if rule == "std140":
        struct_alignment = 16
    else:  # std430: align to the largest member alignment
        struct_alignment = max(member_alignments)
    stride = _round_up(size, struct_alignment)
    return offsets, size, stride, struct_alignment


def type_size(type_name: str) -> int:
    """Return the storage size in bytes of one base *type_name*.

    The rule-independent base sizes are::

        float  4      int    4
        vec2   8      vec3   12      vec4   16
        mat3   48     mat4   64

    ``mat3`` is 48 because it is three 16-byte columns; its three meaningful
    ``vec3`` payloads are 12 bytes each and the remaining 4 bytes of every
    column are padding.  ``vec3`` itself is 12 bytes here: its 16-byte
    alignment is a placement rule, not its payload size.

    Raises:
        LayoutError: if *type_name* is not one of the seven base types.
    """
    if not isinstance(type_name, str) or type_name not in _BASE_SIZES:
        raise LayoutError(f"unknown base type: {type_name!r}")
    return _BASE_SIZES[type_name]


def base_alignment(type_name: str, rule: str) -> int:
    """Return the base alignment in bytes of one base type for *rule*.

    ``rule`` must be ``"std140"`` or ``"std430"``.

    ``std140`` alignments are::

        float  4      int    4
        vec2   8      vec3   16      vec4   16
        mat3   16     mat4   16

    ``std430`` keeps the same scalar, vector and matrix column alignments:
    in particular ``vec3`` is still 16-byte aligned, and ``mat3``/``mat4``
    are aligned as arrays of 16-byte column vectors.  The two rules differ
    for aggregate structure alignment (see :func:`layout_struct`), not for
    the base types here.

    Raises:
        LayoutError: for an unknown type or an unknown rule.
    """
    _validate_rule(rule)
    if not isinstance(type_name, str) or type_name not in _BASE_ALIGNMENTS:
        raise LayoutError(f"unknown base type: {type_name!r}")
    return _BASE_ALIGNMENTS[type_name]


def layout_struct(fields: list[dict], rule: str) -> dict:
    """Lay out *fields* in declaration order and return offsets/size/stride.

    The result is::

        {
            "offsets": {field_name: byte_offset, ...},  # top-level fields
            "size": int,    # end of the last field's storage, no tail padding
            "stride": int,  # size rounded up to the structure alignment
        }

    Each field is aligned to its own base alignment.  A scalar/vector/matrix
    field consumes ``type_size(type_name)`` bytes.  A ``count > 1`` array
    field starts on a 16-byte boundary, uses
    :func:`array_stride` bytes per element, and consumes
    ``count * element_stride`` bytes in total, including the final element's
    padding.  A field whose ``type`` is a list is an inline nested structure:
    it is recursively laid out, placed at the nested structure's alignment
    (16 under ``std140``; the largest member alignment under ``std430``), and
    consumes the nested ``stride``.  An array of nested structures uses a
    nested element stride rounded up to 16.

    Structure alignment is 16 under ``std140``.  Under ``std430`` it is the
    largest member alignment (arrays count as 16-byte aligned), so a
    two-float structure has stride 8 rather than 16.  The structure
    ``stride`` is ``0`` for an empty field list.  This means a single
    ``float`` under ``std140`` has ``size == 4`` and ``stride == 16``.

    Field order is preserved in ``offsets``.  Field names must be unique,
    counts must be positive integers, types must be known (or a non-empty
    list of valid nested fields), and ``rule`` must be known.

    Raises:
        LayoutError: for a malformed field list, duplicate name, unknown type
            or unknown rule.
    """
    offsets, size, stride, _ = _layout(fields, rule)
    return {"offsets": offsets, "size": size, "stride": stride}


def array_stride(type_name: str, count: int, rule: str) -> int:
    """Return the per-element byte stride for an array of *type_name*.

    The element stride is ``type_size(type_name)`` rounded up to a multiple
    of 16.  Therefore a ``float`` array and a ``vec3`` array both have a
    16-byte stride, while ``mat3``/``mat4`` arrays have 48/64-byte strides.
    The rule does not change array stride in this exercise: both ``std140``
    and ``std430`` require a multiple of 16.

    *count* must be a positive integer and is validated, but the returned
    value is the stride of one element, not the total array size.  Multiply
    by *count* when total storage is needed.

    Raises:
        LayoutError: for an unknown type/rule or a non-positive count.
    """
    _validate_rule(rule)
    size = type_size(type_name)
    if not _is_positive_int(count):
        raise LayoutError(f"array count must be a positive integer, got {count!r}")
    return _round_up(size, 16)


def pad_bytes(fields: list[dict], rule: str) -> list[int]:
    """Return the number of padding bytes immediately before each field.

    The list has exactly one non-negative integer per top-level field, in
    declaration order.  The first field's entry is normally 0, but a field
    may be preceded by padding when the previous field's payload ends before
    the next field's required alignment.  For example, a ``vec3`` after a
    ``float`` starts at offset 16, so its preceding padding is 12 bytes.

    Tail padding after the last field is **not** included; that space is
    represented by the difference between ``size`` and ``stride`` in
    :func:`layout_struct`.  Padding internal to a nested structure is
    likewise not reported in this top-level list.  An empty field list
    returns ``[]``.

    Raises:
        LayoutError: for malformed fields or an unknown rule.
    """
    _validate_rule(rule)
    if not isinstance(fields, (list, tuple)):
        raise LayoutError(f"fields must be a list, got {type(fields).__name__}")

    pads = []
    seen = set()
    cursor = 0
    for field in fields:
        info = _field_info(field, rule)
        name = info["name"]
        if name in seen:
            raise LayoutError(f"duplicate field name: {name!r}")
        seen.add(name)
        offset = _round_up(cursor, info["align"])
        pads.append(offset - cursor)
        cursor = offset + info["size"]
    return pads


def validate_layout(fields: list[dict], rule: str) -> list[str]:
    """Return human-readable layout problems; an empty list means valid.

    The canonical layout is computed with :func:`layout_struct`.  Fields may
    carry two optional diagnostic keys used only by this function:

    * ``"offset"``: a proposed byte offset.  It must be non-negative, aligned
      to the field's base alignment, equal to the canonical offset, and its
      storage must end no later than the canonical structure ``stride``.
    * ``"stride"``: for a ``count > 1`` array only, a proposed element stride.
      It must be positive, a multiple of 16, and equal to
      :func:`array_stride`.

    Reported problems cover misplaced/misaligned offsets, overlap with an
    earlier field, storage beyond the structure stride (out of bounds), and
    array strides that are not multiples of 16.  A field without diagnostic
    keys contributes no problem.  The function does not mutate *fields*.

    Structural errors such as an unknown field type or a bad count are
    reported as problem strings rather than raised.  An unknown rule is a
    programming error and raises :class:`LayoutError`.
    """
    _validate_rule(rule)
    if not isinstance(fields, (list, tuple)):
        return [f"fields must be a list, got {type(fields).__name__}"]

    infos = []
    structural_problems = []
    seen = set()
    for field in fields:
        try:
            info = _field_info(field, rule)
            if info["name"] in seen:
                raise LayoutError(f"duplicate field name: {info['name']!r}")
            seen.add(info["name"])
            infos.append(info)
        except LayoutError as exc:
            structural_problems.append(str(exc))
            infos.append(None)

    if structural_problems:
        return structural_problems

    offsets, _, stride, _ = _layout(list(fields), rule)

    problems = []
    placed = []  # (start, end, name) intervals from diagnostic offsets
    for field, info in zip(fields, infos):
        name = info["name"]

        if "offset" in field:
            proposed = field["offset"]
            if not isinstance(proposed, int) or isinstance(proposed, bool) or proposed < 0:
                problems.append(
                    f"field {name}: offset {proposed!r} is not a non-negative integer"
                )
            else:
                if proposed % info["align"] != 0:
                    problems.append(
                        f"field {name}: offset {proposed} is not aligned to "
                        f"{info['align']} bytes"
                    )
                canonical = offsets[name]
                if proposed != canonical:
                    problems.append(
                        f"field {name}: offset {proposed} does not match the "
                        f"canonical offset {canonical}"
                    )
                end = proposed + info["size"]
                if end > stride:
                    problems.append(
                        f"field {name}: storage [{proposed}, {end}) is out of bounds; "
                        f"exceeds structure stride {stride}"
                    )
                for start, prev_end, prev_name in placed:
                    if proposed < prev_end and start < end:
                        problems.append(
                            f"field {name}: offset {proposed} overlaps field "
                            f"{prev_name} at [{start}, {prev_end})"
                        )
                placed.append((proposed, end, name))

        if "stride" in field and info["count"] > 1:
            proposed_stride = field["stride"]
            expected = info["elem_stride"]
            if (
                not isinstance(proposed_stride, int)
                or isinstance(proposed_stride, bool)
                or proposed_stride <= 0
            ):
                problems.append(
                    f"field {name}: array stride {proposed_stride!r} is not a "
                    f"positive integer"
                )
            else:
                if proposed_stride % 16 != 0:
                    problems.append(
                        f"field {name}: array stride {proposed_stride} is not a "
                        f"multiple of 16"
                    )
                if proposed_stride != expected:
                    problems.append(
                        f"field {name}: array stride {proposed_stride} does not "
                        f"match the canonical stride {expected}"
                    )

    return problems


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_sequence(value, length, what):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LayoutError(f"{what} must be a sequence of {length} numbers")
    if len(value) != length:
        raise LayoutError(
            f"{what} must have exactly {length} elements, got {len(value)}"
        )
    return list(value)


def _pack_base(buf, offset, base_type, value, what):
    try:
        if base_type == "float":
            if not _is_number(value):
                raise LayoutError(f"{what} must be a number, got {value!r}")
            struct.pack_into("<f", buf, offset, float(value))
        elif base_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise LayoutError(f"{what} must be an integer, got {value!r}")
            struct.pack_into("<i", buf, offset, value)
        elif base_type in _VECTOR_COMPONENTS:
            components = _as_sequence(
                value, _VECTOR_COMPONENTS[base_type], what
            )
            for component in components:
                if not _is_number(component):
                    raise LayoutError(
                        f"{what} components must be numbers, got {component!r}"
                    )
            struct.pack_into(
                f"<{len(components)}f", buf, offset,
                *(float(c) for c in components),
            )
        elif base_type in _MATRIX_COLUMNS:
            columns = _MATRIX_COLUMNS[base_type]
            flat = _as_sequence(value, columns * columns, what)
            for component in flat:
                if not _is_number(component):
                    raise LayoutError(
                        f"{what} components must be numbers, got {component!r}"
                    )
            # Column-major: each column occupies one 16-byte slot.
            for column in range(columns):
                start = column * columns
                struct.pack_into(
                    f"<{columns}f",
                    buf,
                    offset + 16 * column,
                    *(float(c) for c in flat[start:start + columns]),
                )
        else:  # pragma: no cover - guarded by field validation
            raise LayoutError(f"unknown base type: {base_type!r}")
    except struct.error as exc:
        raise LayoutError(f"{what}: cannot pack {value!r}: {exc}") from exc


def _pack_field(buf, offset, field, value, rule):
    name = field["name"]
    ftype = field["type"]
    count = field.get("count", 1)
    what = f"field {name}"

    if isinstance(ftype, str):
        if count == 1:
            _pack_base(buf, offset, ftype, value, what)
        else:
            elements = _as_sequence(value, count, what)
            elem_stride = _round_up(_BASE_SIZES[ftype], 16)
            for index, element in enumerate(elements):
                _pack_base(
                    buf, offset + index * elem_stride, ftype, element,
                    f"{what}[{index}]",
                )
    else:
        nested = list(ftype)
        _, _, sub_stride, _ = _layout(nested, rule)
        if count == 1:
            if not isinstance(value, Mapping):
                raise LayoutError(f"{what} must be a mapping of nested values")
            _pack_struct(buf, offset, nested, value, rule, what)
        else:
            elements = _as_sequence(value, count, what)
            elem_stride = _round_up(sub_stride, 16)
            for index, element in enumerate(elements):
                if not isinstance(element, Mapping):
                    raise LayoutError(f"{what}[{index}] must be a mapping")
                _pack_struct(
                    buf, offset + index * elem_stride, nested, element,
                    rule, f"{what}[{index}]",
                )


def _pack_struct(buf, base, fields, values, rule, what):
    offsets, _, _, _ = _layout(fields, rule)
    for field in fields:
        name = field["name"]
        if name not in values:
            raise LayoutError(f"{what}: missing value for field {name}")
        _pack_field(buf, base + offsets[name], field, values[name], rule)


def pack_values(fields: list[dict], values: dict, rule: str) -> bytes:
    """Pack *values* according to the layout and return exactly ``stride`` bytes.

    ``values`` maps each top-level field name to:

    * a number for ``float``/``int``;
    * a length-2/3/4 sequence for ``vec2``/``vec3``/``vec4``;
    * a flat length-9/16 sequence in column-major order for
      ``mat3``/``mat4`` (for example, the first three entries are column 0);
    * a length-``count`` sequence of element values for an array;
    * a mapping for an inline nested structure (or a sequence of mappings for
      an array of nested structures).

    The returned buffer is the full structure stride, initialized to zero.
    Padding before fields, between array elements, after the last array
    element, and inside matrix/vector slots stays zero.  Floats use
    little-endian IEEE-754 ``<f`` and ints use little-endian signed 32-bit
    ``<i``.  Missing or malformed values raise :class:`LayoutError`; the
    function never writes outside its zero-filled buffer.

    Raises:
        LayoutError: for malformed fields/values or an unknown rule.
    """
    _validate_rule(rule)
    if not isinstance(values, Mapping):
        raise LayoutError(f"values must be a mapping, got {type(values).__name__}")

    offsets, _, stride, _ = _layout(fields, rule)
    buf = bytearray(stride)
    for field in fields:
        name = field["name"]
        if name not in values:
            raise LayoutError(f"missing value for field {name}")
        _pack_field(buf, offsets[name], field, values[name], rule)
    return bytes(buf)


def layout_batch(blocks: list[dict], rule: str) -> dict:
    """Lay out several independent uniform blocks as one all-or-nothing batch.

    ``blocks`` is a sequence of mappings::

        {"name": "per_frame", "fields": [field, ...]}

    Block names must be non-empty strings and unique.  Every block is laid
    out independently, so offsets and stride never leak from one block into
    another.  The result preserves input order::

        {
            "order": ["per_frame", ...],
            "layouts": {"per_frame": layout_dict, ...},
        }

    The operation is transactional: if any block is malformed or cannot be
    laid out, raise :class:`LayoutError` and do not return or mutate partial
    results.  The input sequence and its mappings are never modified.  An
    empty batch returns ``{"order": [], "layouts": {}}``.

    Raises:
        LayoutError: if any block is invalid, a name is duplicated, or
            ``rule`` is unknown.
    """
    _validate_rule(rule)
    if not isinstance(blocks, (list, tuple)):
        raise LayoutError(f"blocks must be a list, got {type(blocks).__name__}")

    order = []
    layouts = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise LayoutError(f"block must be a mapping, got {type(block).__name__}")
        name = block.get("name")
        if not isinstance(name, str) or not name:
            raise LayoutError(f"block name must be a non-empty string, got {name!r}")
        if name in layouts:
            raise LayoutError(f"duplicate block name: {name!r}")
        if "fields" not in block:
            raise LayoutError(f"block {name!r} is missing its 'fields' list")
        # Each block is laid out from a clean cursor: no shared offset state.
        layouts[name] = layout_struct(block["fields"], rule)
        order.append(name)

    return {"order": order, "layouts": layouts}
