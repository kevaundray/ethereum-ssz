import pytest
from dataclasses import dataclass
from typing import Annotated, List, List as TypingList, Tuple, Union

from ethereum_types.bytes import Bytes, Bytes4, Bytes32, Bytes48, Bytes96
from ethereum_types.numeric import U8, U16, U32, U64, U256, Uint

from ethereum_ssz import ssz
from ethereum_ssz.ssz import _is_fixed_size, _fixed_size_of, MaxLength, With
from ethereum_ssz.exceptions import DecodingError, EncodingError, SSZException


# Task 2: Exception tests

def test_decoding_error_chain() -> None:
    inner = DecodingError("invalid uint")
    outer = DecodingError("cannot decode field `slot`")
    outer.__cause__ = inner
    result = str(outer)
    assert "cannot decode field `slot`" in result
    assert "because invalid uint" in result


def test_encoding_error_is_ssz_exception() -> None:
    with pytest.raises(SSZException):
        raise EncodingError("unsupported type")


def test_decoding_error_is_ssz_exception() -> None:
    with pytest.raises(SSZException):
        raise DecodingError("bad data")


# Task 3: Bool tests

def test_encode_bool_true() -> None:
    assert ssz.encode(True) == b"\x01"


def test_encode_bool_false() -> None:
    assert ssz.encode(False) == b"\x00"


def test_decode_to_bool_true() -> None:
    assert ssz.decode_to(bool, b"\x01") is True


def test_decode_to_bool_false() -> None:
    assert ssz.decode_to(bool, b"\x00") is False


def test_decode_to_bool_invalid() -> None:
    with pytest.raises(DecodingError, match="invalid boolean"):
        ssz.decode_to(bool, b"\x02")


def test_decode_to_bool_empty() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(bool, b"")


def test_decode_to_bool_too_long() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(bool, b"\x01\x00")


# Task 4: Uint tests

def test_encode_uint8() -> None:
    assert ssz.encode(U8(0)) == b"\x00"
    assert ssz.encode(U8(255)) == b"\xff"


def test_encode_uint16() -> None:
    assert ssz.encode(U16(256)) == b"\x00\x01"


def test_encode_uint32() -> None:
    assert ssz.encode(U32(1)) == b"\x01\x00\x00\x00"


def test_encode_uint64() -> None:
    assert ssz.encode(U64(256)) == b"\x00\x01\x00\x00\x00\x00\x00\x00"


def test_encode_uint256() -> None:
    result = ssz.encode(U256(1))
    assert len(result) == 32
    assert result[0] == 1
    assert result[1:] == b"\x00" * 31


def test_decode_to_uint8() -> None:
    assert ssz.decode_to(U8, b"\xff") == U8(255)


def test_decode_to_uint64() -> None:
    data = b"\x00\x01\x00\x00\x00\x00\x00\x00"
    assert ssz.decode_to(U64, data) == U64(256)


def test_decode_to_uint64_wrong_length() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(U64, b"\x00\x01")


def test_decode_to_uint256() -> None:
    data = b"\x01" + b"\x00" * 31
    assert ssz.decode_to(U256, data) == U256(1)


def test_round_trip_uint() -> None:
    for val in [U8(0), U8(255), U64(12345678), U256(2**200)]:
        cls = type(val)
        encoded = ssz.encode(val)
        decoded = ssz.decode_to(cls, encoded)
        assert decoded == val


# Task 5: Bytes tests

def test_encode_fixed_bytes() -> None:
    value = Bytes4(b"\x01\x02\x03\x04")
    assert ssz.encode(value) == b"\x01\x02\x03\x04"


def test_encode_bytes32() -> None:
    value = Bytes32(b"\xab" * 32)
    assert ssz.encode(value) == b"\xab" * 32


def test_encode_variable_bytes() -> None:
    assert ssz.encode(b"\x01\x02\x03") == b"\x01\x02\x03"
    assert ssz.encode(b"") == b""


def test_decode_to_fixed_bytes() -> None:
    data = b"\x01\x02\x03\x04"
    result = ssz.decode_to(Bytes4, data)
    assert isinstance(result, Bytes4)
    assert result == Bytes4(b"\x01\x02\x03\x04")


def test_decode_to_bytes32() -> None:
    data = b"\xab" * 32
    result = ssz.decode_to(Bytes32, data)
    assert isinstance(result, Bytes32)


def test_decode_to_fixed_bytes_wrong_length() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(Bytes32, b"\x00" * 31)


def test_decode_to_variable_bytes() -> None:
    result = ssz.decode_to(bytes, b"\x01\x02\x03")
    assert result == b"\x01\x02\x03"


def test_decode_to_variable_bytes_empty() -> None:
    result = ssz.decode_to(bytes, b"")
    assert result == b""


# Task 6: Type utility tests

def test_is_fixed_size_bool() -> None:
    assert _is_fixed_size(bool) is True


def test_is_fixed_size_uint() -> None:
    assert _is_fixed_size(U64) is True


def test_is_fixed_size_fixed_bytes() -> None:
    assert _is_fixed_size(Bytes32) is True


def test_is_fixed_size_variable_bytes() -> None:
    assert _is_fixed_size(bytes) is False


def test_is_fixed_size_list() -> None:
    assert _is_fixed_size(list[U64]) is False


def test_fixed_size_of_bool() -> None:
    assert _fixed_size_of(bool) == 1


def test_fixed_size_of_uint64() -> None:
    assert _fixed_size_of(U64) == 8


def test_fixed_size_of_uint256() -> None:
    assert _fixed_size_of(U256) == 32


def test_fixed_size_of_bytes32() -> None:
    assert _fixed_size_of(Bytes32) == 32


def test_fixed_size_of_bytes4() -> None:
    assert _fixed_size_of(Bytes4) == 4


# Task 7: Container encoding tests

@dataclass
class Point:
    x: U64
    y: U64


@dataclass
class Signed:
    message: Bytes32
    signature: Bytes96


@dataclass
class WithBool:
    flag: bool
    value: U64


@dataclass
class Variable:
    fixed_val: U32
    data: bytes


@dataclass
class TwoVariable:
    a: bytes
    b: bytes
    fixed: U32


def test_encode_container_fixed() -> None:
    p = Point(U64(1), U64(2))
    expected = (
        b"\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x02\x00\x00\x00\x00\x00\x00\x00"
    )
    assert ssz.encode(p) == expected


def test_encode_container_fixed_bytes() -> None:
    msg = Bytes32(b"\xab" * 32)
    sig = Bytes96(b"\xcd" * 96)
    s = Signed(msg, sig)
    assert ssz.encode(s) == b"\xab" * 32 + b"\xcd" * 96


def test_encode_container_with_bool() -> None:
    w = WithBool(True, U64(42))
    expected = b"\x01" + b"\x2a\x00\x00\x00\x00\x00\x00\x00"
    assert ssz.encode(w) == expected


def test_encode_container_variable() -> None:
    v = Variable(U32(1), b"\xaa\xbb")
    # Fixed part: U32(1) = 4 bytes + offset(4 bytes) = 8 bytes
    # Offset points to byte 8
    expected = (
        b"\x01\x00\x00\x00"
        b"\x08\x00\x00\x00"
        b"\xaa\xbb"
    )
    assert ssz.encode(v) == expected


def test_encode_container_two_variable() -> None:
    t = TwoVariable(b"\x01\x02", b"\x03", U32(99))
    # Fixed: offset_a(4) + offset_b(4) + U32(4) = 12 bytes
    # offset_a = 12, offset_b = 14
    expected = (
        b"\x0c\x00\x00\x00"
        b"\x0e\x00\x00\x00"
        b"\x63\x00\x00\x00"
        b"\x01\x02"
        b"\x03"
    )
    assert ssz.encode(t) == expected


# Task 8: Container decoding tests

def test_decode_container_fixed() -> None:
    data = (
        b"\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x02\x00\x00\x00\x00\x00\x00\x00"
    )
    result = ssz.decode_to(Point, data)
    assert result == Point(U64(1), U64(2))


def test_decode_container_variable() -> None:
    data = (
        b"\x01\x00\x00\x00"
        b"\x08\x00\x00\x00"
        b"\xaa\xbb"
    )
    result = ssz.decode_to(Variable, data)
    assert result == Variable(U32(1), b"\xaa\xbb")


def test_decode_container_two_variable() -> None:
    data = (
        b"\x0c\x00\x00\x00"
        b"\x0e\x00\x00\x00"
        b"\x63\x00\x00\x00"
        b"\x01\x02"
        b"\x03"
    )
    result = ssz.decode_to(TwoVariable, data)
    assert result == TwoVariable(b"\x01\x02", b"\x03", U32(99))


def test_round_trip_container_fixed() -> None:
    original = Point(U64(12345), U64(67890))
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(Point, encoded)
    assert decoded == original


def test_round_trip_container_variable() -> None:
    original = Variable(U32(42), b"\x01\x02\x03\x04\x05")
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(Variable, encoded)
    assert decoded == original


def test_decode_container_wrong_size() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(Point, b"\x01\x00")


@dataclass
class Nested:
    header: Point
    value: U32


def test_round_trip_nested_container() -> None:
    original = Nested(Point(U64(1), U64(2)), U32(3))
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(Nested, encoded)
    assert decoded == original


@dataclass
class Empty:
    pass


def test_encode_empty_container() -> None:
    assert ssz.encode(Empty()) == b""


def test_decode_empty_container() -> None:
    result = ssz.decode_to(Empty, b"")
    assert result == Empty()


def test_decode_empty_container_extra_data() -> None:
    with pytest.raises(DecodingError):
        ssz.decode_to(Empty, b"\x00")


# Task 9: List tests

@dataclass
class WithList:
    values: list[U64]

def test_encode_container_with_list() -> None:
    w = WithList([U64(1), U64(2), U64(3)])
    expected = (
        b"\x04\x00\x00\x00"
        + b"\x01\x00\x00\x00\x00\x00\x00\x00"
        + b"\x02\x00\x00\x00\x00\x00\x00\x00"
        + b"\x03\x00\x00\x00\x00\x00\x00\x00"
    )
    assert ssz.encode(w) == expected

def test_decode_container_with_list() -> None:
    data = (
        b"\x04\x00\x00\x00"
        + b"\x01\x00\x00\x00\x00\x00\x00\x00"
        + b"\x02\x00\x00\x00\x00\x00\x00\x00"
        + b"\x03\x00\x00\x00\x00\x00\x00\x00"
    )
    result = ssz.decode_to(WithList, data)
    assert result == WithList([U64(1), U64(2), U64(3)])

def test_round_trip_empty_list() -> None:
    original = WithList([])
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithList, encoded)
    assert decoded == original

@dataclass
class WithVariableList:
    items: list[bytes]

def test_encode_variable_element_list() -> None:
    w = WithVariableList([b"\x01\x02", b"\x03"])
    container_offset = b"\x04\x00\x00\x00"
    list_data = (
        b"\x08\x00\x00\x00"
        b"\x0a\x00\x00\x00"
        b"\x01\x02"
        b"\x03"
    )
    assert ssz.encode(w) == container_offset + list_data

def test_round_trip_variable_element_list() -> None:
    original = WithVariableList([b"\x01\x02", b"\x03\x04\x05"])
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithVariableList, encoded)
    assert decoded == original

# Task 9: Tuple tests

@dataclass
class WithFixedTuple:
    point: Tuple[U64, U32]

def test_encode_fixed_tuple() -> None:
    w = WithFixedTuple((U64(1), U32(2)))
    expected = (
        b"\x01\x00\x00\x00\x00\x00\x00\x00"
        b"\x02\x00\x00\x00"
    )
    assert ssz.encode(w) == expected

def test_round_trip_fixed_tuple() -> None:
    original = WithFixedTuple((U64(100), U32(200)))
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithFixedTuple, encoded)
    assert decoded == original

@dataclass
class WithHomogeneousTuple:
    values: Tuple[U64, ...]

def test_encode_homogeneous_tuple() -> None:
    w = WithHomogeneousTuple((U64(1), U64(2)))
    expected = (
        b"\x04\x00\x00\x00"
        + b"\x01\x00\x00\x00\x00\x00\x00\x00"
        + b"\x02\x00\x00\x00\x00\x00\x00\x00"
    )
    assert ssz.encode(w) == expected

def test_round_trip_homogeneous_tuple() -> None:
    original = WithHomogeneousTuple((U64(10), U64(20), U64(30)))
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithHomogeneousTuple, encoded)
    assert decoded == original

def test_decode_list_wrong_element_size() -> None:
    @dataclass
    class BadList:
        items: list[U64]
    data = b"\x04\x00\x00\x00" + b"\x00" * 7
    with pytest.raises(DecodingError):
        ssz.decode_to(BadList, data)

# Task 10: MaxLength tests

@dataclass
class WithMaxLength:
    data: Annotated[list[U64], MaxLength(1024)]

def test_encode_with_max_length() -> None:
    w = WithMaxLength([U64(1), U64(2)])
    expected = (
        b"\x04\x00\x00\x00"
        + b"\x01\x00\x00\x00\x00\x00\x00\x00"
        + b"\x02\x00\x00\x00\x00\x00\x00\x00"
    )
    assert ssz.encode(w) == expected

def test_round_trip_with_max_length() -> None:
    original = WithMaxLength([U64(10), U64(20)])
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithMaxLength, encoded)
    assert decoded == original

@dataclass
class WithAnnotatedBytes:
    extra_data: Annotated[bytes, MaxLength(32)]

def test_round_trip_annotated_bytes() -> None:
    original = WithAnnotatedBytes(b"\xab\xcd\xef")
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithAnnotatedBytes, encoded)
    assert decoded == original

# Task 10: With (custom codec) tests

def decode_custom(data: bytes) -> U64:
    return U64.from_le_bytes(data)

@dataclass
class WithCustomCodec:
    value: Annotated[U64, With(decode_custom)]

def test_decode_with_custom_codec() -> None:
    data = ssz.encode(WithCustomCodec(U64(42)))
    result = ssz.decode_to(WithCustomCodec, data)
    assert result == WithCustomCodec(U64(42))

@dataclass
class WithUnrelatedAnnotated:
    foo: Annotated[U64, "ignore me!"]

def test_round_trip_unrelated_annotated() -> None:
    original = WithUnrelatedAnnotated(U64(99))
    encoded = ssz.encode(original)
    decoded = ssz.decode_to(WithUnrelatedAnnotated, encoded)
    assert decoded == original

@dataclass
class WithTwoWith:
    foo: Annotated[U64, With(decode_custom), With(decode_custom)]

def test_decode_two_with_raises() -> None:
    with pytest.raises(DecodingError, match="multiple ssz.With"):
        ssz.decode_to(WithTwoWith, b"\x00" * 8)

# Task 10: Union tests

@dataclass
class WithUnion:
    value: Union[Bytes4, bool]

def test_decode_union_left() -> None:
    # Bytes4 matches 4-byte data, bool doesn't (needs 1 byte)
    w = WithUnion(Bytes4(b"\x01\x02\x03\x04"))
    data = ssz.encode(w)
    result = ssz.decode_to(WithUnion, data)
    assert result.value == Bytes4(b"\x01\x02\x03\x04")

def test_decode_union_right() -> None:
    w = WithUnion(True)
    data = ssz.encode(w)
    result = ssz.decode_to(WithUnion, data)
    assert result.value is True

def test_decode_union_no_match() -> None:
    @dataclass
    class BadUnion:
        value: Union[Bytes4, Bytes32]
    # 5 bytes matches neither Bytes4 (needs 4) nor Bytes32 (needs 32)
    # We need to construct raw data that would trigger this in a container
    # Container: the value field. For a fixed-size union... hmm.
    # Actually unions of different fixed sizes are tricky.
    # Let's test with a simpler scenario:
    with pytest.raises(DecodingError, match="no matching union variant"):
        ssz._decode_union(Union[Bytes4, Bytes32], b"\x00" * 5)


# ======================================================================
# Task 11: Integration tests (public API imports)
# ======================================================================

from ethereum_ssz import SSZ, MaxLength, With, decode_to, encode


@dataclass
class BeaconBlockHeader:
    slot: U64
    proposer_index: U64
    parent_root: Bytes32
    state_root: Bytes32
    body_root: Bytes32


def test_beacon_block_header_round_trip() -> None:
    header = BeaconBlockHeader(
        slot=U64(100),
        proposer_index=U64(42),
        parent_root=Bytes32(b"\x01" * 32),
        state_root=Bytes32(b"\x02" * 32),
        body_root=Bytes32(b"\x03" * 32),
    )
    encoded = encode(header)
    assert len(encoded) == 112  # 2*8 + 3*32
    decoded = decode_to(BeaconBlockHeader, encoded)
    assert decoded == header


@dataclass
class Validator:
    pubkey: Bytes48
    withdrawal_credentials: Bytes32
    effective_balance: U64
    slashed: bool
    activation_epoch: U64
    exit_epoch: U64


@dataclass
class SimpleState:
    genesis_time: U64
    slot: U64
    validators: Annotated[list[Validator], MaxLength(1099511627776)]


def test_simple_state_round_trip() -> None:
    v1 = Validator(
        pubkey=Bytes48(b"\xaa" * 48),
        withdrawal_credentials=Bytes32(b"\xbb" * 32),
        effective_balance=U64(32000000000),
        slashed=False,
        activation_epoch=U64(0),
        exit_epoch=U64(2**64 - 1),
    )
    v2 = Validator(
        pubkey=Bytes48(b"\xcc" * 48),
        withdrawal_credentials=Bytes32(b"\xdd" * 32),
        effective_balance=U64(32000000000),
        slashed=True,
        activation_epoch=U64(100),
        exit_epoch=U64(200),
    )
    state = SimpleState(
        genesis_time=U64(1606824023),
        slot=U64(1000000),
        validators=[v1, v2],
    )
    encoded = encode(state)
    decoded = decode_to(SimpleState, encoded)
    assert decoded == state


def test_ssz_protocol() -> None:
    header = BeaconBlockHeader(
        slot=U64(0),
        proposer_index=U64(0),
        parent_root=Bytes32(b"\x00" * 32),
        state_root=Bytes32(b"\x00" * 32),
        body_root=Bytes32(b"\x00" * 32),
    )
    assert isinstance(header, SSZ)


# ======================================================================
# Task 12: Error handling edge cases
# ======================================================================

from typing import Any, cast


def test_encode_unsupported_type() -> None:
    with pytest.raises(EncodingError, match="unsupported type"):
        ssz.encode(cast(Any, 123))


def test_encode_unsupported_str() -> None:
    with pytest.raises(EncodingError, match="unsupported type"):
        ssz.encode(cast(Any, "hello"))


def test_chained_decoding_error() -> None:
    @dataclass
    class Inner:
        value: U64

    @dataclass
    class Outer:
        inner: Inner
        flag: bool

    # Inner needs 8 bytes, bool needs 1 = 9 total for Outer
    # Provide 9 bytes but with inner containing only invalid-length
    # data for the nested U64 — actually the fixed-size decoding
    # will slice exactly 8 bytes for Inner, which then decodes fine.
    # Instead: provide correct-size data but make the Inner's U64
    # wrong by making Inner have a sub-container that fails.
    # Simpler approach: Outer is 9 bytes. Give it exactly 9 bytes
    # but with bool = 0x02 (invalid).
    data = b"\x00" * 8 + b"\x02"
    with pytest.raises(DecodingError, match="cannot decode field"):
        ssz.decode_to(Outer, data)


# ======================================================================
# Issue 3: Additional coverage tests
# ======================================================================


def test_decoding_error_non_decoding_cause() -> None:
    """DecodingError with a non-DecodingError __cause__ formats with 'because'."""
    inner = ValueError("bad value")
    outer = DecodingError("cannot decode field `slot`")
    outer.__cause__ = inner
    result = str(outer)
    assert "cannot decode field `slot`" in result
    assert "because bad value" in result


def test_decode_union_multiple_matching_variants() -> None:
    """Union[bool, U8] with 1-byte data matches both variants."""
    with pytest.raises(DecodingError, match="multiple matching union variants"):
        ssz._decode_union(Union[bool, U8], b"\x01")


def test_decode_unsupported_type() -> None:
    """_decode_value raises DecodingError for unsupported type."""
    with pytest.raises(DecodingError, match="unsupported type for SSZ decoding"):
        ssz._decode_value(float, b"\x00")


def test_decode_annotation_unsupported() -> None:
    """_decode_annotation raises DecodingError for unsupported annotations."""
    # dict[str, str] has origin=dict, which is not Union/tuple/list
    with pytest.raises(DecodingError, match="unsupported annotation"):
        ssz._decode_annotation(dict[str, str], b"\x00")


def test_is_fixed_size_fixed_tuple() -> None:
    """Tuple of fixed types is fixed-size."""
    from typing import Tuple
    assert _is_fixed_size(Tuple[U64, U32]) is True


def test_is_fixed_size_bare_tuple() -> None:
    """Bare Tuple (no args) is not fixed-size."""
    from typing import Tuple
    assert _is_fixed_size(Tuple) is False


def test_is_fixed_size_unknown_type() -> None:
    """_is_fixed_size returns False for unknown types."""
    assert _is_fixed_size(float) is False


def test_is_fixed_size_union() -> None:
    """Union types are not fixed-size."""
    assert _is_fixed_size(Union[U64, bool]) is False


def test_fixed_size_of_raises_for_unsupported() -> None:
    """_fixed_size_of raises EncodingError for unsupported types."""
    with pytest.raises(EncodingError, match="cannot determine fixed size"):
        _fixed_size_of(float)


def test_encode_value_unsupported_type_hint() -> None:
    """_encode_value raises EncodingError for unsupported type hints."""
    with pytest.raises(EncodingError, match="unsupported type hint"):
        ssz._encode_value(3.14, float)


def test_decode_container_offset_before_fixed_part() -> None:
    """Offset pointing into the fixed part raises DecodingError."""
    # TwoVariable has: a(bytes, offset=4), b(bytes, offset=4), fixed(U32, 4)
    # fixed_part_size = 4 + 4 + 4 = 12
    # Craft data where first offset is 2 (< 12)
    import struct
    data = (
        struct.pack("<I", 2)     # offset for a = 2 (invalid: < 12)
        + struct.pack("<I", 12)  # offset for b = 12
        + struct.pack("<I", 99)  # fixed = 99
    )
    with pytest.raises(DecodingError, match="before the end of the fixed part"):
        ssz.decode_to(TwoVariable, data)


def test_decode_container_offset_beyond_data() -> None:
    """Offset beyond data length raises DecodingError."""
    import struct
    data = (
        struct.pack("<I", 12)    # offset for a = 12
        + struct.pack("<I", 999) # offset for b = 999 (beyond data)
        + struct.pack("<I", 99)  # fixed = 99
        + b"\x01\x02"           # variable data for a
    )
    with pytest.raises(DecodingError, match="beyond the data length"):
        ssz.decode_to(TwoVariable, data)


def test_decode_container_offsets_not_monotonic() -> None:
    """Non-monotonically increasing offsets raise DecodingError."""
    import struct
    data = (
        struct.pack("<I", 14)   # offset for a = 14
        + struct.pack("<I", 12) # offset for b = 12 (< 14, not monotonic)
        + struct.pack("<I", 99) # fixed = 99
        + b"\x01\x02"          # some variable data
        + b"\x03"
    )
    with pytest.raises(DecodingError, match="less than the previous offset"):
        ssz.decode_to(TwoVariable, data)


def test_decode_sequence_offset_beyond_data() -> None:
    """Variable-size sequence with offset beyond data raises DecodingError."""
    import struct
    # Two elements: first_offset = 8, so num_elements = 2
    # offsets: [8, 999]
    data = (
        struct.pack("<I", 8)
        + struct.pack("<I", 999)  # beyond data
        + b"\x01\x02"
    )
    with pytest.raises(DecodingError, match="beyond the data length"):
        ssz._decode_sequence(data, bytes)


def test_decode_sequence_offsets_not_monotonic() -> None:
    """Variable-size sequence with non-monotonic offsets raises DecodingError."""
    import struct
    # Two elements: first_offset = 8, so num_elements = 2
    # offsets: [8, 7] — 7 < 8, not monotonic
    data = (
        struct.pack("<I", 8)
        + struct.pack("<I", 7)  # less than previous
        + b"\x01\x02\x03\x04\x05\x06\x07\x08"
    )
    with pytest.raises(DecodingError, match="less than the previous offset"):
        ssz._decode_sequence(data, bytes)


def test_decode_sequence_variable_too_short() -> None:
    """Variable-size sequence with data shorter than offset size raises DecodingError."""
    with pytest.raises(DecodingError, match="data too short"):
        ssz._decode_sequence(b"\x01\x02", bytes)


def test_decode_sequence_invalid_first_offset() -> None:
    """First offset not a multiple of BYTES_PER_LENGTH_OFFSET raises DecodingError."""
    import struct
    data = struct.pack("<I", 5)  # 5 is not a multiple of 4
    with pytest.raises(DecodingError, match="not a multiple"):
        ssz._decode_sequence(data, bytes)


def test_decode_bytearray() -> None:
    """Decoding to bytearray type works."""
    result = ssz.decode_to(bytearray, b"\x01\x02")
    assert result == b"\x01\x02"


def test_encode_bytearray() -> None:
    """Encoding a bytearray works."""
    result = ssz.encode(bytearray(b"\x01\x02"))
    assert result == b"\x01\x02"


def test_annotated_with_multiple_metadata() -> None:
    """Annotated with With codec and MaxLength decodes correctly
    (Python flattens nested Annotated, so With + MaxLength appear in
    the same Annotated layer)."""
    @dataclass
    class AnnotatedMulti:
        value: Annotated[Annotated[U64, With(decode_custom)], MaxLength(10)]

    data = ssz.encode(AnnotatedMulti(U64(42)))
    result = ssz.decode_to(AnnotatedMulti, data)
    assert result == AnnotatedMulti(U64(42))


def test_annotated_flattened_two_with_raises() -> None:
    """When Python flattens nested Annotated, multiple With codecs are detected."""
    @dataclass
    class BadAnnotated:
        value: Annotated[Annotated[U64, With(decode_custom), With(decode_custom)], MaxLength(10)]

    with pytest.raises(DecodingError, match="multiple ssz.With"):
        ssz.decode_to(BadAnnotated, b"\x00" * 8)


def test_annotated_flattened_no_with_unwraps() -> None:
    """When Python flattens nested Annotated without With, it unwraps correctly."""
    @dataclass
    class PlainAnnotated:
        value: Annotated[Annotated[U64, "some metadata"], MaxLength(10)]

    data = ssz.encode(PlainAnnotated(U64(7)))
    result = ssz.decode_to(PlainAnnotated, data)
    assert result == PlainAnnotated(U64(7))
