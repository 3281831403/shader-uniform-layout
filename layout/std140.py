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


class LayoutError(Exception):
    """Raised for an invalid uniform layout, type, count, offset or rule."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
