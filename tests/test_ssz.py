import pytest
from dataclasses import dataclass
from typing import Annotated, List, Tuple, Union

from ethereum_types.bytes import Bytes, Bytes4, Bytes32, Bytes48, Bytes96
from ethereum_types.numeric import U8, U16, U32, U64, U256, Uint

from ethereum_ssz import ssz
from ethereum_ssz.ssz import _is_fixed_size, _fixed_size_of
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
