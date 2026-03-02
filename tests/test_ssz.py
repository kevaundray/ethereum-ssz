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
