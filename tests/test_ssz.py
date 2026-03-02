import pytest
from dataclasses import dataclass
from typing import Annotated, List, Tuple, Union

from ethereum_types.bytes import Bytes, Bytes4, Bytes32, Bytes48, Bytes96
from ethereum_types.numeric import U8, U16, U32, U64, U256, Uint

from ethereum_ssz import ssz
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
