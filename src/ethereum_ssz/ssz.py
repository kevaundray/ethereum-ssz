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
    Callable,
    Dict,
    Protocol,
    Sequence,
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

Decoder = Callable[[bytes], Any]


class MaxLength:
    """
    When used with Annotated, specifies the maximum length
    of a List or variable-length bytes for Merkleization.
    Ignored during serialization/deserialization.
    """

    def __init__(self, length: int) -> None:
        self._length = length


class With:
    """
    When used with Annotated, indicates that a value needs to be
    decoded using a custom function.
    """

    def __init__(self, decoder: "Decoder") -> None:
        self._decoder = decoder


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
    # Handle Annotated types via _decode_annotated
    if get_origin(class_) is Annotated:
        result, unwrapped = _decode_annotated(class_, data)
        if result is not None:
            return result
        # unwrapped is the inner type, continue dispatch
        class_ = unwrapped

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

    origin = get_origin(type_hint)

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

    # Generic type dispatch (list, tuple, Union, etc.)
    if origin is list:
        return _encode_list(value, type_hint)
    if origin is tuple:
        return _encode_tuple(value, type_hint)
    if origin is not None and any(origin is ut for ut in _UNION_TYPES):
        # For Union types, encode the value based on its runtime type
        return encode(value)

    raise EncodingError(
        f"unsupported type hint for encoding: {type_hint}"
    )


def _encode_list(value: Any, type_hint: object) -> bytes:
    """Encode a list to SSZ bytes."""
    args = get_args(type_hint)
    element_type = args[0]
    return _encode_sequence(value, element_type)


def _encode_tuple(value: Any, type_hint: object) -> bytes:
    """Encode a tuple to SSZ bytes."""
    args = get_args(type_hint)
    if len(args) == 2 and args[1] is Ellipsis:
        # Homogeneous tuple: tuple[T, ...]
        return _encode_sequence(value, args[0])
    else:
        # Heterogeneous fixed tuple: tuple[T1, T2, ...]
        result = bytearray()
        for v, t in zip(value, args):
            result.extend(_encode_value(v, t))
        return bytes(result)


def _encode_sequence(values: Any, element_type: object) -> bytes:
    """Encode a sequence of elements to SSZ bytes."""
    if _is_fixed_size(element_type):
        # Fixed-size elements: just concatenate
        result = bytearray()
        for v in values:
            result.extend(_encode_value(v, element_type))
        return bytes(result)
    else:
        # Variable-size elements: offset-based encoding
        encoded_elements = [_encode_value(v, element_type) for v in values]
        offsets_size = len(values) * BYTES_PER_LENGTH_OFFSET
        result = bytearray()
        offset = offsets_size
        for encoded in encoded_elements:
            result.extend(struct.pack("<I", offset))
            offset += len(encoded)
        for encoded in encoded_elements:
            result.extend(encoded)
        return bytes(result)


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
            try:
                decoded_values[f.name] = _decode_value(
                    field_type, field_data
                )
            except DecodingError as e:
                raise DecodingError(
                    f"cannot decode field `{f.name}` of "
                    f"{cls.__name__}"
                ) from e
            pos += size

    # Validate variable field offsets
    for i, (name, field_type, offset_val) in enumerate(variable_fields):
        if i == 0 and offset_val != fixed_part_size:
            raise DecodingError(
                f"first variable offset {offset_val} for field "
                f"`{name}` of {cls.__name__} does not match "
                f"fixed part size ({fixed_part_size})"
            )
        if offset_val > len(data):
            raise DecodingError(
                f"offset {offset_val} for field `{name}` of "
                f"{cls.__name__} is beyond the data length "
                f"({len(data)})"
            )
        if i > 0:
            prev_offset = variable_fields[i - 1][2]
            if offset_val < prev_offset:
                raise DecodingError(
                    f"offset {offset_val} for field `{name}` of "
                    f"{cls.__name__} is less than the previous "
                    f"offset ({prev_offset})"
                )

    # Decode variable fields
    for i, (name, field_type, offset_val) in enumerate(variable_fields):
        if i + 1 < len(variable_fields):
            end = variable_fields[i + 1][2]
        else:
            end = len(data)
        field_data = data[offset_val:end]
        try:
            decoded_values[name] = _decode_value(field_type, field_data)
        except DecodingError as e:
            raise DecodingError(
                f"cannot decode field `{name}` of {cls.__name__}"
            ) from e

    # Construct the dataclass
    kwargs = {f.name: decoded_values[f.name] for f in field_list}
    return cls(**kwargs)


def _decode_annotation(class_: Any, data: bytes) -> Any:
    """Decode SSZ bytes using a generic annotation (Union, Tuple, List)."""
    origin = get_origin(class_)

    if origin is not None and any(origin is ut for ut in _UNION_TYPES):
        return _decode_union(class_, data)
    if origin is tuple:
        return _decode_tuple(class_, data)
    if origin is list:
        return _decode_list(class_, data)

    raise DecodingError(
        f"unsupported annotation for SSZ decoding: {class_}"
    )


def _decode_annotated(annotation: Any, data: bytes) -> tuple:
    """
    Decode an Annotated type.

    Returns (result, None) if a With codec handled decoding,
    or (None, unwrapped_type) if Annotated should be unwrapped
    and dispatch should continue.
    """
    args = get_args(annotation)
    metadata = args[1:]  # annotation.__metadata__ equivalent

    with_codecs = [m for m in metadata if isinstance(m, With)]
    if len(with_codecs) > 1:
        raise DecodingError("multiple ssz.With annotations")
    if len(with_codecs) == 1:
        codec = with_codecs[0]
        result = codec._decoder(data)
        return (result, None)

    # No With found: unwrap and continue dispatch
    inner = args[0]
    # Continue unwrapping if still Annotated
    while get_origin(inner) is Annotated:
        inner_args = get_args(inner)
        inner_metadata = inner_args[1:]
        inner_with = [m for m in inner_metadata if isinstance(m, With)]
        if len(inner_with) > 1:
            raise DecodingError("multiple ssz.With annotations")
        if len(inner_with) == 1:
            result = inner_with[0]._decoder(data)
            return (result, None)
        inner = inner_args[0]

    return (None, inner)


def _decode_list(annotation: Any, data: bytes) -> list:
    """Decode SSZ bytes to a list."""
    args = get_args(annotation)
    element_type = args[0]
    return _decode_sequence(data, element_type)


def _decode_tuple(annotation: Any, data: bytes) -> tuple:
    """Decode SSZ bytes to a tuple."""
    args = get_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        # Homogeneous tuple: tuple[T, ...]
        return tuple(_decode_sequence(data, args[0]))
    else:
        # Heterogeneous fixed tuple
        pos = 0
        values = []
        for t in args:
            size = _fixed_size_of(t)
            element_data = data[pos : pos + size]
            values.append(_decode_value(t, element_data))
            pos += size
        return tuple(values)


def _decode_sequence(data: bytes, element_type: object) -> list:
    """Decode SSZ bytes to a list of elements."""
    if len(data) == 0:
        return []

    if _is_fixed_size(element_type):
        element_size = _fixed_size_of(element_type)
        if len(data) % element_size != 0:
            raise DecodingError(
                f"data length {len(data)} is not a multiple of "
                f"element size {element_size}"
            )
        result = []
        for i in range(0, len(data), element_size):
            element_data = data[i : i + element_size]
            result.append(_decode_value(element_type, element_data))
        return result
    else:
        # Variable-size elements: offset-based decoding
        if len(data) < BYTES_PER_LENGTH_OFFSET:
            raise DecodingError(
                "data too short for variable-size sequence"
            )
        first_offset = struct.unpack_from("<I", data, 0)[0]
        if first_offset % BYTES_PER_LENGTH_OFFSET != 0:
            raise DecodingError(
                f"invalid first offset {first_offset}: "
                f"not a multiple of {BYTES_PER_LENGTH_OFFSET}"
            )
        if first_offset > len(data):
            raise DecodingError(
                f"first offset {first_offset} is beyond "
                f"the data length ({len(data)})"
            )
        num_elements = first_offset // BYTES_PER_LENGTH_OFFSET

        offsets = []
        for i in range(num_elements):
            offset = struct.unpack_from(
                "<I", data, i * BYTES_PER_LENGTH_OFFSET
            )[0]
            offsets.append(offset)

        # Validate offsets
        for i, offset in enumerate(offsets):
            if offset < first_offset:
                raise DecodingError(
                    f"offset {offset} at index {i} points into "
                    f"the offsets region (first_offset="
                    f"{first_offset})"
                )
            if offset > len(data):
                raise DecodingError(
                    f"offset {offset} at index {i} is beyond "
                    f"the data length ({len(data)})"
                )
            if i > 0 and offset < offsets[i - 1]:
                raise DecodingError(
                    f"offset {offset} at index {i} is less than "
                    f"the previous offset ({offsets[i - 1]})"
                )

        result = []
        for i in range(num_elements):
            start = offsets[i]
            if i + 1 < num_elements:
                end = offsets[i + 1]
            else:
                end = len(data)
            element_data = data[start:end]
            result.append(_decode_value(element_type, element_data))
        return result


def _decode_union(annotation: Any, data: bytes) -> Any:
    """Decode SSZ bytes by trying each variant of a Union type."""
    args = get_args(annotation)
    successes = []
    failures = []

    for variant in args:
        try:
            result = _decode_value(variant, data)
            successes.append(result)
        except (DecodingError, ValueError):
            failures.append(variant)

    if len(successes) == 1:
        return successes[0]
    elif len(successes) == 0:
        raise DecodingError("no matching union variant")
    else:
        raise DecodingError("multiple matching union variants")
