"""
Defines the SSZ serialization and deserialization format.
"""

import sys
from dataclasses import is_dataclass
from typing import (
    Annotated,
    Any,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

from ethereum_types.bytes import FixedBytes
from ethereum_types.numeric import FixedUnsigned

from ethereum_ssz.exceptions import DecodingError, EncodingError

BYTES_PER_LENGTH_OFFSET = 4

# Python 3.10+ has types.UnionType for X | Y syntax
_UNION_TYPES: tuple = (Union,)
if sys.version_info >= (3, 10):
    from types import UnionType

    _UNION_TYPES = (Union, UnionType)


@runtime_checkable
class SSZ(Protocol):
    """Protocol for SSZ-serializable dataclass types."""

    __dataclass_fields__: dict


T = TypeVar("T")

# Extended covers all types that encode/decode can handle
Extended = Union[
    bool,
    FixedUnsigned,
    FixedBytes,
    bytes,
    bytearray,
    SSZ,
]


def encode(value: Extended) -> bytes:
    """
    Encode a value to SSZ bytes.
    """
    if isinstance(value, bool):
        if value:
            return b"\x01"
        else:
            return b"\x00"
    elif isinstance(value, FixedUnsigned):
        return _encode_uint(value)
    elif isinstance(value, FixedBytes):
        return bytes(value)
    elif isinstance(value, (bytes, bytearray)):
        return bytes(value)
    elif is_dataclass(value) and not isinstance(value, type):
        return _encode_container(value)
    else:
        raise EncodingError(
            f"unsupported type for SSZ encoding: {type(value)}"
        )


def decode_to(cls: Type[T], data: bytes) -> T:
    """
    Decode SSZ bytes to a value of the given type.
    """
    return _decode_value(cls, data)  # type: ignore[return-value]


def _decode_value(class_: Any, data: bytes) -> Any:
    """
    Internal dispatch for decoding SSZ data to the appropriate type.
    """
    # Unwrap Annotated types
    while get_origin(class_) is Annotated:
        class_ = get_args(class_)[0]

    origin = get_origin(class_)

    # Check for non-type annotations (Union, Tuple, List)
    if origin is not None and any(
        origin is ut for ut in _UNION_TYPES
    ):
        return _decode_annotation(class_, data)
    if origin is tuple:
        return _decode_annotation(class_, data)
    if origin is list:
        return _decode_annotation(class_, data)

    # Direct type dispatch
    if class_ is bool:
        return _decode_bool(data)
    elif isinstance(class_, type) and issubclass(class_, FixedUnsigned):
        return _decode_uint(class_, data)
    elif isinstance(class_, type) and issubclass(class_, FixedBytes):
        return _decode_fixed_bytes(class_, data)
    elif isinstance(class_, type) and issubclass(class_, (bytes, bytearray)):
        return _decode_bytes(data)
    elif is_dataclass(class_):
        return _decode_container(class_, data)
    else:
        raise DecodingError(
            f"unsupported type for SSZ decoding: {class_}"
        )


# --- Bool ---


def _decode_bool(data: bytes) -> bool:
    """Decode a boolean from SSZ bytes."""
    if len(data) != 1:
        raise DecodingError(
            f"invalid boolean: expected 1 byte, got {len(data)}"
        )
    if data == b"\x01":
        return True
    elif data == b"\x00":
        return False
    else:
        raise DecodingError(
            f"invalid boolean value: 0x{data[0]:02x}"
        )


# --- Uint ---


def _uint_byte_width(cls: Type[FixedUnsigned]) -> int:
    """Return the byte width of a FixedUnsigned type."""
    return (int(cls.MAX_VALUE.bit_length()) + 7) // 8


def _encode_uint(value: FixedUnsigned) -> bytes:
    """Encode a FixedUnsigned value to SSZ bytes."""
    byte_width = _uint_byte_width(type(value))
    return value.to_bytes(byte_width, "little")


def _decode_uint(cls: Type[FixedUnsigned], data: bytes) -> FixedUnsigned:
    """Decode SSZ bytes to a FixedUnsigned value."""
    expected_width = _uint_byte_width(cls)
    if len(data) != expected_width:
        raise DecodingError(
            f"invalid uint: expected {expected_width} bytes, "
            f"got {len(data)}"
        )
    return cls.from_le_bytes(data)


# --- Bytes ---


def _decode_fixed_bytes(cls: Type[FixedBytes], data: bytes) -> FixedBytes:
    """Decode SSZ bytes to a FixedBytes value."""
    try:
        return cls(data)
    except ValueError as e:
        raise DecodingError(
            f"invalid fixed bytes for {cls.__name__}: {e}"
        ) from e


def _decode_bytes(data: bytes) -> bytes:
    """Decode SSZ bytes to variable-length bytes."""
    return data


# --- Stubs for future implementation ---


def _encode_container(value: Any) -> bytes:
    """Encode a dataclass container to SSZ bytes."""
    raise NotImplementedError("container encoding not yet implemented")


def _decode_container(cls: Any, data: bytes) -> Any:
    """Decode SSZ bytes to a dataclass container."""
    raise NotImplementedError("container decoding not yet implemented")


def _decode_annotation(class_: Any, data: bytes) -> Any:
    """Decode SSZ bytes using a generic annotation (Union, Tuple, List)."""
    raise NotImplementedError("annotation decoding not yet implemented")
