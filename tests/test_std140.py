"""Initial red test suite for the std140/std430 uniform-layout exercise.

Each test calls at least one public stub in ``layout.std140`` and expresses
one user-visible contract.  The suite collects cleanly while all fifteen
tests fail with ``NotImplementedError``.  Do not edit or weaken these tests.
"""

import struct

import pytest

from layout.std140 import (
    LayoutError,
    array_stride,
    base_alignment,
    layout_batch,
    layout_struct,
    pack_values,
    pad_bytes,
    type_size,
    validate_layout,
)


def field(name, type_name, count=1, **extra):
    """Build a field using the documented ``name``/``type``/``count`` keys."""
    item = {"name": name, "type": type_name, "count": count}
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# One regression test per documented defect.
# ---------------------------------------------------------------------------


def test_defect_01_vec3_uses_16_byte_alignment():
    # Defect 1: treating vec3 as 12-byte aligned shifts tint/alpha down.
    fields = [
        field("scale", "float"),
        field("tint", "vec3"),
        field("alpha", "float"),
    ]

    layout = layout_struct(fields, "std140")

    assert base_alignment("vec3", "std140") == 16
    assert layout["offsets"] == {"scale": 0, "tint": 16, "alpha": 28}
    assert layout["size"] == 32
    assert layout["stride"] == 32
    assert pad_bytes(fields, "std140") == [0, 12, 0]


def test_defect_02_mat4_columns_are_16_byte_aligned():
    # Defect 2: a mat4 aligned to 4 starts at offset 4 and its columns drift.
    fields = [
        field("weight", "float"),
        field("transform", "mat4"),
        field("bias", "float"),
    ]
    values = {
        "weight": 0.25,
        "transform": tuple(range(1, 17)),
        "bias": 0.5,
    }

    layout = layout_struct(fields, "std140")
    assert base_alignment("mat4", "std140") == 16
    assert layout["offsets"] == {"weight": 0, "transform": 16, "bias": 80}

    data = pack_values(fields, values, "std140")
    assert struct.unpack_from("<4f", data, 16) == (1.0, 2.0, 3.0, 4.0)
    assert struct.unpack_from("<4f", data, 32) == (5.0, 6.0, 7.0, 8.0)
    assert struct.unpack_from("<4f", data, 48) == (9.0, 10.0, 11.0, 12.0)
    assert struct.unpack_from("<4f", data, 64) == (13.0, 14.0, 15.0, 16.0)


def test_defect_03_array_stride_is_a_multiple_of_16():
    # Defect 3: using element size as stride makes weights[1] start at offset 4.
    assert array_stride("float", 4, "std140") == 16
    assert array_stride("vec3", 3, "std140") == 16
    assert array_stride("mat4", 2, "std430") == 64

    fields = [field("weights", "float", 4), field("tail", "float")]
    layout = layout_struct(fields, "std140")
    assert layout["offsets"] == {"weights": 0, "tail": 64}
    assert layout["size"] == 68
    assert layout["stride"] == 80

    data = pack_values(
        fields,
        {"weights": [1.25, 2.5, 3.75, 5.0], "tail": 9.5},
        "std140",
    )
    assert struct.unpack_from("<f", data, 16)[0] == 2.5
    assert struct.unpack_from("<f", data, 48)[0] == 5.0
    assert struct.unpack_from("<f", data, 64)[0] == 9.5


def test_defect_04_std430_does_not_reuse_std140_stride():
    # Defect 4: std430 must not force a two-float structure to stride 16.
    fields = [field("x", "float"), field("y", "float")]

    std140 = layout_struct(fields, "std140")
    std430 = layout_struct(fields, "std430")

    assert std140["offsets"] == std430["offsets"] == {"x": 0, "y": 4}
    assert std140["size"] == std430["size"] == 8
    assert std140["stride"] == 16
    assert std430["stride"] == 8
    assert pad_bytes(fields, "std140") == [0, 0]
    assert pad_bytes(fields, "std430") == [0, 0]


def test_defect_05_nested_struct_layout_recurses():
    # Defect 5: an inline struct must get its own offsets and tail padding.
    inner = [field("energy", "float"), field("position", "vec3")]
    outer = [
        field("id", "int"),
        field("light", inner),
        field("exposure", "float"),
    ]

    layout = layout_struct(outer, "std140")
    assert layout["offsets"] == {"id": 0, "light": 16, "exposure": 48}
    assert layout["size"] == 52
    assert layout["stride"] == 64

    values = {
        "id": 7,
        "light": {"energy": 1.5, "position": (2.5, 3.5, 4.5)},
        "exposure": 9.5,
    }
    data = pack_values(outer, values, "std140")
    assert struct.unpack_from("<f", data, 16)[0] == 1.5
    assert struct.unpack_from("<3f", data, 32) == (2.5, 3.5, 4.5)
    assert struct.unpack_from("<f", data, 48)[0] == 9.5


def test_defect_06_pack_values_keeps_padding_gaps():
    # Defect 6: packing fields back-to-back puts the vec3 at offset 4.
    fields = [field("scale", "float"), field("tint", "vec3")]
    data = pack_values(fields, {"scale": 1.25, "tint": (2.0, 3.0, 4.0)}, "std140")

    assert len(data) == 32
    assert struct.unpack_from("<f", data, 0)[0] == 1.25
    assert data[4:16] == b"\x00" * 12
    assert struct.unpack_from("<3f", data, 16) == (2.0, 3.0, 4.0)
    assert data[28:32] == b"\x00" * 4


def test_defect_07_int_and_float_use_distinct_packers():
    # Defect 7: both are 4-byte scalars with alignment 4, but an int must be
    # encoded as an integer rather than reused as a float payload.
    fields = [field("count", "int"), field("scale", "float"), field("bias", "int")]

    assert type_size("int") == type_size("float") == 4
    assert base_alignment("int", "std140") == 4
    assert base_alignment("float", "std140") == 4

    layout = layout_struct(fields, "std140")
    assert layout["offsets"] == {"count": 0, "scale": 4, "bias": 8}

    data = pack_values(fields, {"count": 7, "scale": 1.5, "bias": -3}, "std140")
    assert struct.unpack_from("<i", data, 0)[0] == 7
    assert struct.unpack_from("<f", data, 4)[0] == 1.5
    assert struct.unpack_from("<i", data, 8)[0] == -3


def test_defect_08_validate_reports_bad_offsets_and_bounds():
    # Defect 8: a misaligned offset, storage past the stride, and a bad array
    # stride must all be reported instead of silently writing out of bounds.
    fields = [
        field("pos", "vec3", offset=4),
        field("tail", "float", offset=64),
        field("weights", "float", 2, offset=80, stride=4),
    ]

    issues = validate_layout(fields, "std140")

    assert issues
    assert any("pos" in issue and "offset" in issue for issue in issues)
    assert any(
        "tail" in issue and ("out of bounds" in issue or "exceeds" in issue)
        for issue in issues
    )
    assert any(
        "weights" in issue and "multiple of 16" in issue for issue in issues
    )
    assert validate_layout(
        [field("color", "vec3"), field("energy", "float")], "std140"
    ) == []


def test_defect_09_layout_batch_isolates_blocks_and_rolls_back():
    # Defect 9: each block owns its offset table; one bad block aborts all.
    blocks = [
        {"name": "per_frame", "fields": [field("time", "float"), field("camera", "vec3")]},
        {"name": "per_object", "fields": [field("scale", "float")]},
    ]

    result = layout_batch(blocks, "std140")
    assert result["layouts"]["per_frame"]["offsets"] == {"time": 0, "camera": 16}
    assert result["layouts"]["per_frame"]["stride"] == 32
    assert result["layouts"]["per_object"]["offsets"] == {"scale": 0}
    assert result["layouts"]["per_object"]["stride"] == 16

    bad_blocks = [
        {"name": "ok", "fields": [field("value", "float")]},
        {"name": "broken", "fields": [field("value", "vec9")]},
    ]
    before = [
        {"name": "ok", "fields": [field("value", "float")]},
        {"name": "broken", "fields": [field("value", "vec9")]},
    ]

    with pytest.raises(LayoutError):
        layout_batch(bad_blocks, "std140")

    assert bad_blocks == before


def test_defect_10_pack_values_is_little_endian():
    # Defect 10: big-endian bytes decode to a completely different float.
    fields = [field("value", "float")]
    data = pack_values(fields, {"value": 1.5}, "std140")

    assert data[:4] == b"\x00\x00\xc0\x3f"
    assert data[:4] == struct.pack("<f", 1.5)
    assert data[:4] == struct.pack(">f", 1.5)[::-1]
    assert struct.unpack("<f", data[:4])[0] == 1.5


# ---------------------------------------------------------------------------
# Five additional contract tests.
# ---------------------------------------------------------------------------


def test_combo_single_float_size_4_stride_16():
    fields = [field("value", "float")]
    layout = layout_struct(fields, "std140")

    assert layout == {
        "offsets": {"value": 0},
        "size": 4,
        "stride": 16,
    }
    assert pad_bytes(fields, "std140") == [0]


def test_combo_empty_fields_return_empty_layout():
    empty = {"offsets": {}, "size": 0, "stride": 0}

    assert layout_struct([], "std140") == empty
    assert layout_struct([], "std430") == empty
    assert pad_bytes([], "std140") == []
    assert validate_layout([], "std140") == []


def test_combo_unknown_rule_raises_layout_error():
    calls = [
        lambda: layout_struct([], "std999"),
        lambda: base_alignment("vec3", "std999"),
        lambda: array_stride("float", 2, "std999"),
        lambda: pad_bytes([], "std999"),
        lambda: validate_layout([], "std999"),
        lambda: pack_values([], {}, "std999"),
        lambda: layout_batch([], "std999"),
    ]

    for call in calls:
        with pytest.raises(LayoutError):
            call()


def test_combo_pack_values_round_trips_with_struct_unpack():
    fields = [field("time", "float"), field("tint", "vec3"), field("model", "mat4")]
    model = tuple(float(index) for index in range(16))
    values = {"time": 1.25, "tint": (0.5, 1.5, 2.5), "model": model}

    data = pack_values(fields, values, "std140")
    assert len(data) == 96
    assert struct.unpack_from("<f", data, 0)[0] == 1.25
    assert struct.unpack_from("<3f", data, 16) == (0.5, 1.5, 2.5)

    columns = []
    for offset in (32, 48, 64, 80):
        columns.extend(struct.unpack_from("<4f", data, offset))
    assert tuple(columns) == model


def test_combo_layout_batch_success_keeps_order():
    names = ["frame", "material", "object"]
    blocks = [
        {"name": "frame", "fields": [field("time", "float")]},
        {"name": "material", "fields": [field("albedo", "vec4")]},
        {"name": "object", "fields": [field("model", "mat4")]},
    ]

    result = layout_batch(blocks, "std140")

    assert result["order"] == names
    assert list(result["layouts"]) == names
    assert [result["layouts"][name]["stride"] for name in names] == [16, 16, 64]
