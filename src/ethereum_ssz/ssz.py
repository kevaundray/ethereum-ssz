"""
Defines the SSZ serialization and deserialization format.
"""

import re
import struct
import sys
from dataclasses import fields, is_dataclass
from typing import (
    Annotated,
    Any,
    Dict,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
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


# --- Type Utilities ---

_fixed_bytes_length_cache: Dict[type, int] = {}


def _fixed_bytes_length(cls: type) -> int:
    """Return the fixed byte length of a FixedBytes subclass."""
    if cls in _fixed_bytes_length_cache:
        return _fixed_bytes_length_cache[cls]
    match = re.match(r"Bytes(\d+)", cls.__name__)
    if match:
        length = int(match.group(1))
        _fixed_bytes_length_cache[cls] = length
        return length
    # Fallback: trial construction
    for length in range(257):
        try:
            cls(b"\x00" * length)
            _fixed_bytes_length_cache[cls] = length
            return length
        except ValueError:
            continue
    raise EncodingError(f"cannot determine length of {cls.__name__}")


def _is_fixed_size(type_hint: object) -> bool:
    """Determine if an SSZ type is fixed-size."""
    # Unwrap Annotated
    while get_origin(type_hint) is Annotated:
        type_hint = get_args(type_hint)[0]

    origin = get_origin(type_hint)

    # Union → False
    if origin is not None and any(origin is ut for ut in _UNION_TYPES):
        return False

    # list[T] / List[T] → False
    if origin is list:
        return False

    # tuple handling
    if origin is tuple:
        args = get_args(type_hint)
        if not args:
            return False
        # tuple[T, ...] (variable length) → False
        if len(args) == 2 and args[1] is Ellipsis:
            return False
        # tuple[T1, T2, ...] (explicit types) → True if all fixed
        return all(_is_fixed_size(a) for a in args)

    # Direct types
    if type_hint is bool:
        return True
    if isinstance(type_hint, type) and issubclass(type_hint, FixedUnsigned):
        return True
    if isinstance(type_hint, type) and issubclass(type_hint, FixedBytes):
        return True
    if isinstance(type_hint, type) and issubclass(
        type_hint, (bytes, bytearray)
    ):
        return False

    # Dataclass → True if all fields are fixed-size
    if is_dataclass(type_hint):
        hints = get_type_hints(type_hint, include_extras=True)
        return all(_is_fixed_size(hints[f.name]) for f in fields(type_hint))

    return False


def _fixed_size_of(type_hint: object) -> int:
    """Return the byte size of a fixed-size SSZ type."""
    # Unwrap Annotated
    while get_origin(type_hint) is Annotated:
        type_hint = get_args(type_hint)[0]

    origin = get_origin(type_hint)

    # tuple[T1, T2, ...]
    if origin is tuple:
        args = get_args(type_hint)
        return sum(_fixed_size_of(a) for a in args)

    # Direct types
    if type_hint is bool:
        return 1
    if isinstance(type_hint, type) and issubclass(type_hint, FixedUnsigned):
        return _uint_byte_width(type_hint)
    if isinstance(type_hint, type) and issubclass(type_hint, FixedBytes):
        return _fixed_bytes_length(type_hint)

    # Dataclass
    if is_dataclass(type_hint):
        hints = get_type_hints(type_hint, include_extras=True)
        return sum(
            _fixed_size_of(hints[f.name]) for f in fields(type_hint)
        )

    raise EncodingError(
        f"cannot determine fixed size of {type_hint}"
    )


# --- Container Encoding ---


def _encode_value(value: Any, type_hint: object) -> bytes:
    """Encode a value given its type hint (used by container encoding)."""
    # Unwrap Annotated
    while get_origin(type_hint) is Annotated:
        type_hint = get_args(type_hint)[0]

    # Dispatch by type hint
    if type_hint is bool:
        return b"\x01" if value else b"\x00"
    if isinstance(type_hint, type) and issubclass(type_hint, FixedUnsigned):
        return _encode_uint(value)
    if isinstance(type_hint, type) and issubclass(type_hint, FixedBytes):
        return bytes(value)
    if isinstance(type_hint, type) and issubclass(
        type_hint, (bytes, bytearray)
    ):
        return bytes(value)
    if is_dataclass(type_hint):
        return _encode_container(value)

    raise EncodingError(
        f"unsupported type hint for encoding: {type_hint}"
    )


def _encode_container(value: Any) -> bytes:
    """Encode a dataclass container to SSZ bytes."""
    cls = type(value)
    hints = get_type_hints(cls, include_extras=True)
    field_list = fields(cls)

    # Handle empty container
    if not field_list:
        return b""

    fixed_parts: list = []
    variable_parts: list = []
    variable_indices: list = []

    for i, f in enumerate(field_list):
        field_type = hints[f.name]
        field_value = getattr(value, f.name)
        if _is_fixed_size(field_type):
            fixed_parts.append(_encode_value(field_value, field_type))
        else:
            fixed_parts.append(None)  # placeholder for offset
            variable_parts.append(_encode_value(field_value, field_type))
            variable_indices.append(i)

    # Calculate fixed part size
    fixed_part_size = 0
    for part in fixed_parts:
        if part is None:
            fixed_part_size += BYTES_PER_LENGTH_OFFSET
        else:
            fixed_part_size += len(part)

    # Build offsets and output
    result = bytearray()
    var_idx = 0
    # Calculate starting offset for variable data
    offset = fixed_part_size
    for part in fixed_parts:
        if part is None:
            result.extend(struct.pack("<I", offset))
            offset += len(variable_parts[var_idx])
            var_idx += 1
        else:
            result.extend(part)

    # Append variable parts
    for vp in variable_parts:
        result.extend(vp)

    return bytes(result)


# --- Container Decoding ---


def _decode_container(cls: Any, data: bytes) -> Any:
    """Decode SSZ bytes to a dataclass container."""
    hints = get_type_hints(cls, include_extras=True)
    field_list = fields(cls)

    # Handle empty container
    if not field_list:
        if len(data) != 0:
            raise DecodingError(
                f"expected 0 bytes for empty container "
                f"{cls.__name__}, got {len(data)}"
            )
        return cls()

    # Build field layout
    fixed_part_size = 0
    field_layout = []  # (field, type_hint, is_variable)
    for f in field_list:
        field_type = hints[f.name]
        is_var = not _is_fixed_size(field_type)
        field_layout.append((f, field_type, is_var))
        if is_var:
            fixed_part_size += BYTES_PER_LENGTH_OFFSET
        else:
            fixed_part_size += _fixed_size_of(field_type)

    if len(data) < fixed_part_size:
        raise DecodingError(
            f"data too short for {cls.__name__}: "
            f"expected at least {fixed_part_size} bytes, "
            f"got {len(data)}"
        )

    # Decode fixed fields and read offsets for variable fields
    pos = 0
    decoded_values = {}
    variable_fields = []  # (field_name, field_type, offset_value)

    for f, field_type, is_var in field_layout:
        if is_var:
            offset_val = struct.unpack_from("<I", data, pos)[0]
            variable_fields.append((f.name, field_type, offset_val))
            pos += BYTES_PER_LENGTH_OFFSET
        else:
            size = _fixed_size_of(field_type)
            field_data = data[pos : pos + size]
            decoded_values[f.name] = _decode_value(field_type, field_data)
            pos += size

    # Decode variable fields
    for i, (name, field_type, offset_val) in enumerate(variable_fields):
        if i + 1 < len(variable_fields):
            end = variable_fields[i + 1][2]
        else:
            end = len(data)
        field_data = data[offset_val:end]
        decoded_values[name] = _decode_value(field_type, field_data)

    # Construct the dataclass
    kwargs = {f.name: decoded_values[f.name] for f in field_list}
    return cls(**kwargs)


def _decode_annotation(class_: Any, data: bytes) -> Any:
    """Decode SSZ bytes using a generic annotation (Union, Tuple, List)."""
    raise NotImplementedError("annotation decoding not yet implemented")
