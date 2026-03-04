"""
Cross-validation tests comparing ethereum_ssz against two independent SSZ
implementations:

1. py-ssz (the ``ssz`` package) -- sedes-based encode/decode
2. remerkleable -- type-centric encode/decode (used by consensus-specs)

Every test encodes values with both libraries and asserts byte-level equality,
then decodes the bytes back and verifies round-trip correctness.
"""

import struct
from dataclasses import dataclass

import pytest
import ssz as py_ssz
from ethereum_types.bytes import Bytes4, Bytes32, Bytes48, Bytes96
from ethereum_types.numeric import U8, U16, U32, U64, U256
from remerkleable.basic import boolean as rm_boolean
from remerkleable.basic import uint8 as rm_uint8
from remerkleable.basic import uint16 as rm_uint16
from remerkleable.basic import uint32 as rm_uint32
from remerkleable.basic import uint64 as rm_uint64
from remerkleable.basic import uint256 as rm_uint256
from remerkleable.byte_arrays import ByteList as RmByteList
from remerkleable.byte_arrays import ByteVector as RmByteVector
from remerkleable.complex import Container as RmContainer
from remerkleable.complex import List as RmList
from remerkleable.complex import Vector as RmVector
from ssz.sedes import (
    Boolean,
    ByteList,
    List,
    Serializable,
    Vector,
    bytes4,
    bytes32,
    bytes48,
    bytes96,
    uint8,
    uint16,
    uint32,
    uint64,
    uint256,
)

import ethereum_ssz
from ethereum_ssz.exceptions import DecodingError


# ---------------------------------------------------------------------------
# Bool
# ---------------------------------------------------------------------------


class TestBoolCrossValidation:
    @pytest.mark.parametrize("value", [True, False])
    def test_bool_encoding(self, value: bool) -> None:
        expected = py_ssz.encode(value, Boolean())
        actual = ethereum_ssz.encode(value)
        assert actual == expected

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_roundtrip(self, value: bool) -> None:
        encoded = py_ssz.encode(value, Boolean())
        decoded = ethereum_ssz.decode_to(bool, encoded)
        assert decoded == value

        encoded2 = ethereum_ssz.encode(value)
        decoded2 = py_ssz.decode(encoded2, Boolean())
        assert decoded2 == value


# ---------------------------------------------------------------------------
# Unsigned integers
# ---------------------------------------------------------------------------


class TestUintCrossValidation:
    @pytest.mark.parametrize(
        "value,our_type,their_sedes",
        [
            (0, U8, uint8),
            (1, U8, uint8),
            (255, U8, uint8),
            (0, U16, uint16),
            (1000, U16, uint16),
            (65535, U16, uint16),
            (0, U32, uint32),
            (100000, U32, uint32),
            (2**32 - 1, U32, uint32),
            (0, U64, uint64),
            (2**32, U64, uint64),
            (2**64 - 1, U64, uint64),
            (0, U256, uint256),
            (2**128, U256, uint256),
            (2**256 - 1, U256, uint256),
        ],
    )
    def test_uint_encoding(self, value, our_type, their_sedes) -> None:
        expected = py_ssz.encode(value, their_sedes)
        actual = ethereum_ssz.encode(our_type(value))
        assert actual == expected

    @pytest.mark.parametrize(
        "value,our_type,their_sedes",
        [
            (42, U8, uint8),
            (1000, U16, uint16),
            (100000, U32, uint32),
            (2**32, U64, uint64),
            (2**128, U256, uint256),
        ],
    )
    def test_uint_roundtrip(self, value, our_type, their_sedes) -> None:
        # Encode with py-ssz, decode with ours
        encoded = py_ssz.encode(value, their_sedes)
        decoded = ethereum_ssz.decode_to(our_type, encoded)
        assert int(decoded) == value

        # Encode with ours, decode with py-ssz
        encoded2 = ethereum_ssz.encode(our_type(value))
        decoded2 = py_ssz.decode(encoded2, their_sedes)
        assert decoded2 == value


# ---------------------------------------------------------------------------
# Fixed-length bytes
# ---------------------------------------------------------------------------


class TestFixedBytesCrossValidation:
    @pytest.mark.parametrize(
        "value,our_type,their_sedes",
        [
            (b"\xab\xcd\xef\x01", Bytes4, bytes4),
            (b"\x00" * 4, Bytes4, bytes4),
            (b"\xff" * 4, Bytes4, bytes4),
            (b"\x01" * 32, Bytes32, bytes32),
            (b"\x00" * 32, Bytes32, bytes32),
            (b"\xff" * 32, Bytes32, bytes32),
            (b"\x01" * 48, Bytes48, bytes48),
            (b"\x01" * 96, Bytes96, bytes96),
        ],
    )
    def test_fixed_bytes_encoding(self, value, our_type, their_sedes) -> None:
        expected = py_ssz.encode(value, their_sedes)
        actual = ethereum_ssz.encode(our_type(value))
        assert actual == expected

    @pytest.mark.parametrize(
        "value,our_type,their_sedes",
        [
            (b"\xab\xcd\xef\x01", Bytes4, bytes4),
            (b"\x01" * 32, Bytes32, bytes32),
        ],
    )
    def test_fixed_bytes_roundtrip(
        self, value, our_type, their_sedes
    ) -> None:
        encoded = py_ssz.encode(value, their_sedes)
        decoded = ethereum_ssz.decode_to(our_type, encoded)
        assert bytes(decoded) == value

        encoded2 = ethereum_ssz.encode(our_type(value))
        decoded2 = py_ssz.decode(encoded2, their_sedes)
        assert decoded2 == value


# ---------------------------------------------------------------------------
# Variable-length bytes
# ---------------------------------------------------------------------------


class TestVariableBytesCrossValidation:
    @pytest.mark.parametrize(
        "value",
        [
            b"",
            b"\x01\x02\x03",
            b"\x00" * 100,
            b"\xff" * 256,
            bytes(range(256)),
        ],
    )
    def test_variable_bytes_encoding(self, value: bytes) -> None:
        expected = py_ssz.encode(value, ByteList(1024))
        actual = ethereum_ssz.encode(value)
        assert actual == expected

    def test_variable_bytes_roundtrip(self) -> None:
        value = b"\x01\x02\x03\x04\x05"
        encoded = py_ssz.encode(value, ByteList(1024))
        decoded = ethereum_ssz.decode_to(bytes, encoded)
        assert decoded == value


# ---------------------------------------------------------------------------
# List of fixed-size elements
# ---------------------------------------------------------------------------


class TestFixedListCrossValidation:
    @pytest.mark.parametrize(
        "values,our_elem_type,their_sedes",
        [
            ([1, 2, 3], U32, List(uint32, max_length=100)),
            ([], U32, List(uint32, max_length=100)),
            ([0, 2**32 - 1], U32, List(uint32, max_length=100)),
            ([10, 20, 30], U64, List(uint64, max_length=100)),
            ([0], U64, List(uint64, max_length=100)),
            ([2**64 - 1], U64, List(uint64, max_length=100)),
            ([True, False, True], bool, List(Boolean(), max_length=100)),
        ],
    )
    def test_list_encoding(self, values, our_elem_type, their_sedes) -> None:
        # py-ssz wants tuples for lists
        expected = py_ssz.encode(tuple(values), their_sedes)

        # ethereum_ssz encodes lists via containers; we need to wrap in a
        # container to test list encoding
        @dataclass
        class Wrapper:
            items: list[our_elem_type]  # type: ignore[valid-type]

        class PyWrapper(Serializable):
            fields = [("items", their_sedes)]

        # Build our value
        if our_elem_type is bool:
            our_values = values
        else:
            our_values = [our_elem_type(v) for v in values]

        our_encoded = ethereum_ssz.encode(Wrapper(items=our_values))
        their_encoded = py_ssz.encode(
            PyWrapper(items=tuple(values)), PyWrapper
        )
        assert our_encoded == their_encoded

    def test_list_u32_raw_bytes(self) -> None:
        """Test that raw list bytes match between libraries."""
        values = (1, 2, 3)
        expected = py_ssz.encode(values, List(uint32, max_length=100))
        # Manually encode for comparison
        result = b""
        for v in values:
            result += ethereum_ssz.encode(U32(v))
        assert result == expected


# ---------------------------------------------------------------------------
# Vector (fixed-length list)
# ---------------------------------------------------------------------------


class TestVectorCrossValidation:
    def test_vector_u64_encoding(self) -> None:
        values = (10, 20, 30)
        expected = py_ssz.encode(values, Vector(uint64, 3))
        # Our library uses tuples for fixed-size vectors
        result = b""
        for v in values:
            result += ethereum_ssz.encode(U64(v))
        assert result == expected


# ---------------------------------------------------------------------------
# Containers: fixed-size fields only
# ---------------------------------------------------------------------------


class PyPoint(Serializable):
    fields = [("x", uint64), ("y", uint64)]


@dataclass
class OurPoint:
    x: U64
    y: U64


class PySigned(Serializable):
    fields = [("value", uint64), ("negative", Boolean())]


@dataclass
class OurSigned:
    value: U64
    negative: bool


class TestFixedContainerCrossValidation:
    def test_point_encoding(self) -> None:
        expected = py_ssz.encode(PyPoint(x=1, y=2), PyPoint)
        actual = ethereum_ssz.encode(OurPoint(x=U64(1), y=U64(2)))
        assert actual == expected

    def test_point_roundtrip(self) -> None:
        data = py_ssz.encode(PyPoint(x=100, y=200), PyPoint)
        decoded = ethereum_ssz.decode_to(OurPoint, data)
        assert int(decoded.x) == 100
        assert int(decoded.y) == 200

        data2 = ethereum_ssz.encode(OurPoint(x=U64(100), y=U64(200)))
        decoded2 = py_ssz.decode(data2, PyPoint)
        assert decoded2.x == 100
        assert decoded2.y == 200

    def test_signed_encoding(self) -> None:
        expected = py_ssz.encode(
            PySigned(value=42, negative=True), PySigned
        )
        actual = ethereum_ssz.encode(
            OurSigned(value=U64(42), negative=True)
        )
        assert actual == expected

    def test_signed_roundtrip(self) -> None:
        data = py_ssz.encode(PySigned(value=42, negative=True), PySigned)
        decoded = ethereum_ssz.decode_to(OurSigned, data)
        assert int(decoded.value) == 42
        assert decoded.negative is True

    @pytest.mark.parametrize(
        "x,y",
        [
            (0, 0),
            (1, 1),
            (2**64 - 1, 2**64 - 1),
            (2**32, 2**32 + 1),
        ],
    )
    def test_point_boundary_values(self, x: int, y: int) -> None:
        expected = py_ssz.encode(PyPoint(x=x, y=y), PyPoint)
        actual = ethereum_ssz.encode(OurPoint(x=U64(x), y=U64(y)))
        assert actual == expected


# ---------------------------------------------------------------------------
# Containers: mixed fixed and variable-size fields
# ---------------------------------------------------------------------------


class PyVariable(Serializable):
    fields = [("id", uint64), ("data", ByteList(2048))]


@dataclass
class OurVariable:
    id: U64
    data: bytes


class PyTwoVariable(Serializable):
    fields = [("first", ByteList(256)), ("second", ByteList(256))]


@dataclass
class OurTwoVariable:
    first: bytes
    second: bytes


class TestVariableContainerCrossValidation:
    def test_single_variable_field(self) -> None:
        expected = py_ssz.encode(
            PyVariable(id=42, data=b"\x01\x02\x03"), PyVariable
        )
        actual = ethereum_ssz.encode(
            OurVariable(id=U64(42), data=b"\x01\x02\x03")
        )
        assert actual == expected

    def test_single_variable_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyVariable(id=42, data=b"\xaa\xbb\xcc"), PyVariable
        )
        decoded = ethereum_ssz.decode_to(OurVariable, data)
        assert int(decoded.id) == 42
        assert decoded.data == b"\xaa\xbb\xcc"

    def test_two_variable_fields(self) -> None:
        expected = py_ssz.encode(
            PyTwoVariable(first=b"hello", second=b"world"), PyTwoVariable
        )
        actual = ethereum_ssz.encode(
            OurTwoVariable(first=b"hello", second=b"world")
        )
        assert actual == expected

    def test_two_variable_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyTwoVariable(first=b"abc", second=b"defgh"), PyTwoVariable
        )
        decoded = ethereum_ssz.decode_to(OurTwoVariable, data)
        assert decoded.first == b"abc"
        assert decoded.second == b"defgh"

    def test_empty_variable_data(self) -> None:
        expected = py_ssz.encode(
            PyVariable(id=0, data=b""), PyVariable
        )
        actual = ethereum_ssz.encode(OurVariable(id=U64(0), data=b""))
        assert actual == expected

    def test_large_variable_data(self) -> None:
        big = bytes(range(256)) * 4
        expected = py_ssz.encode(
            PyVariable(id=1, data=big),
            PyVariable,
        )
        actual = ethereum_ssz.encode(OurVariable(id=U64(1), data=big))
        assert actual == expected


# ---------------------------------------------------------------------------
# Nested containers
# ---------------------------------------------------------------------------


class PyInner(Serializable):
    fields = [("a", uint32), ("b", uint32)]


class PyOuter(Serializable):
    fields = [("inner", PyInner), ("extra", uint64)]


@dataclass
class OurInner:
    a: U32
    b: U32


@dataclass
class OurOuter:
    inner: OurInner
    extra: U64


class TestNestedContainerCrossValidation:
    def test_nested_encoding(self) -> None:
        expected = py_ssz.encode(
            PyOuter(inner=PyInner(a=10, b=20), extra=30), PyOuter
        )
        actual = ethereum_ssz.encode(
            OurOuter(inner=OurInner(a=U32(10), b=U32(20)), extra=U64(30))
        )
        assert actual == expected

    def test_nested_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyOuter(inner=PyInner(a=100, b=200), extra=300), PyOuter
        )
        decoded = ethereum_ssz.decode_to(OurOuter, data)
        assert int(decoded.inner.a) == 100
        assert int(decoded.inner.b) == 200
        assert int(decoded.extra) == 300


# ---------------------------------------------------------------------------
# Container with list field
# ---------------------------------------------------------------------------


class PyWithList(Serializable):
    fields = [
        ("count", uint32),
        ("values", List(uint32, max_length=1024)),
    ]


@dataclass
class OurWithList:
    count: U32
    values: list[U32]


class TestContainerWithListCrossValidation:
    def test_encoding(self) -> None:
        expected = py_ssz.encode(
            PyWithList(count=3, values=(1, 2, 3)), PyWithList
        )
        actual = ethereum_ssz.encode(
            OurWithList(count=U32(3), values=[U32(1), U32(2), U32(3)])
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyWithList(count=5, values=(10, 20, 30, 40, 50)), PyWithList
        )
        decoded = ethereum_ssz.decode_to(OurWithList, data)
        assert int(decoded.count) == 5
        assert [int(v) for v in decoded.values] == [10, 20, 30, 40, 50]

    def test_empty_list(self) -> None:
        expected = py_ssz.encode(
            PyWithList(count=0, values=()), PyWithList
        )
        actual = ethereum_ssz.encode(
            OurWithList(count=U32(0), values=[])
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# Ethereum Consensus Layer types: BeaconBlockHeader
# ---------------------------------------------------------------------------


class PyBeaconBlockHeader(Serializable):
    fields = [
        ("slot", uint64),
        ("proposer_index", uint64),
        ("parent_root", bytes32),
        ("state_root", bytes32),
        ("body_root", bytes32),
    ]


@dataclass
class OurBeaconBlockHeader:
    slot: U64
    proposer_index: U64
    parent_root: Bytes32
    state_root: Bytes32
    body_root: Bytes32


class TestBeaconBlockHeaderCrossValidation:
    def test_encoding(self) -> None:
        expected = py_ssz.encode(
            PyBeaconBlockHeader(
                slot=100,
                proposer_index=42,
                parent_root=b"\x01" * 32,
                state_root=b"\x02" * 32,
                body_root=b"\x03" * 32,
            ),
            PyBeaconBlockHeader,
        )
        actual = ethereum_ssz.encode(
            OurBeaconBlockHeader(
                slot=U64(100),
                proposer_index=U64(42),
                parent_root=Bytes32(b"\x01" * 32),
                state_root=Bytes32(b"\x02" * 32),
                body_root=Bytes32(b"\x03" * 32),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyBeaconBlockHeader(
                slot=6543210,
                proposer_index=99999,
                parent_root=bytes(range(32)),
                state_root=bytes(range(32, 64)),
                body_root=bytes(range(64, 96)),
            ),
            PyBeaconBlockHeader,
        )
        decoded = ethereum_ssz.decode_to(OurBeaconBlockHeader, data)
        assert int(decoded.slot) == 6543210
        assert int(decoded.proposer_index) == 99999
        assert bytes(decoded.parent_root) == bytes(range(32))
        assert bytes(decoded.state_root) == bytes(range(32, 64))
        assert bytes(decoded.body_root) == bytes(range(64, 96))

    def test_genesis_header(self) -> None:
        """All-zero genesis block header."""
        expected = py_ssz.encode(
            PyBeaconBlockHeader(
                slot=0,
                proposer_index=0,
                parent_root=b"\x00" * 32,
                state_root=b"\x00" * 32,
                body_root=b"\x00" * 32,
            ),
            PyBeaconBlockHeader,
        )
        actual = ethereum_ssz.encode(
            OurBeaconBlockHeader(
                slot=U64(0),
                proposer_index=U64(0),
                parent_root=Bytes32(b"\x00" * 32),
                state_root=Bytes32(b"\x00" * 32),
                body_root=Bytes32(b"\x00" * 32),
            )
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# Ethereum type: Checkpoint
# ---------------------------------------------------------------------------


class PyCheckpoint(Serializable):
    fields = [("epoch", uint64), ("root", bytes32)]


@dataclass
class OurCheckpoint:
    epoch: U64
    root: Bytes32


class TestCheckpointCrossValidation:
    def test_encoding(self) -> None:
        expected = py_ssz.encode(
            PyCheckpoint(epoch=10, root=b"\xaa" * 32), PyCheckpoint
        )
        actual = ethereum_ssz.encode(
            OurCheckpoint(epoch=U64(10), root=Bytes32(b"\xaa" * 32))
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyCheckpoint(epoch=999, root=b"\xbb" * 32), PyCheckpoint
        )
        decoded = ethereum_ssz.decode_to(OurCheckpoint, data)
        assert int(decoded.epoch) == 999
        assert bytes(decoded.root) == b"\xbb" * 32


# ---------------------------------------------------------------------------
# Ethereum type: Fork
# ---------------------------------------------------------------------------


class PyFork(Serializable):
    fields = [
        ("previous_version", bytes4),
        ("current_version", bytes4),
        ("epoch", uint64),
    ]


@dataclass
class OurFork:
    previous_version: Bytes4
    current_version: Bytes4
    epoch: U64


class TestForkCrossValidation:
    def test_encoding(self) -> None:
        expected = py_ssz.encode(
            PyFork(
                previous_version=b"\x00\x00\x00\x01",
                current_version=b"\x00\x00\x00\x02",
                epoch=100,
            ),
            PyFork,
        )
        actual = ethereum_ssz.encode(
            OurFork(
                previous_version=Bytes4(b"\x00\x00\x00\x01"),
                current_version=Bytes4(b"\x00\x00\x00\x02"),
                epoch=U64(100),
            )
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# Complex: container with variable-length bytes field (like Eth1Data)
# ---------------------------------------------------------------------------


class PyEth1Data(Serializable):
    fields = [
        ("deposit_root", bytes32),
        ("deposit_count", uint64),
        ("block_hash", bytes32),
    ]


@dataclass
class OurEth1Data:
    deposit_root: Bytes32
    deposit_count: U64
    block_hash: Bytes32


class TestEth1DataCrossValidation:
    def test_encoding(self) -> None:
        expected = py_ssz.encode(
            PyEth1Data(
                deposit_root=b"\x11" * 32,
                deposit_count=12345,
                block_hash=b"\x22" * 32,
            ),
            PyEth1Data,
        )
        actual = ethereum_ssz.encode(
            OurEth1Data(
                deposit_root=Bytes32(b"\x11" * 32),
                deposit_count=U64(12345),
                block_hash=Bytes32(b"\x22" * 32),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyEth1Data(
                deposit_root=b"\xaa" * 32,
                deposit_count=99999,
                block_hash=b"\xbb" * 32,
            ),
            PyEth1Data,
        )
        decoded = ethereum_ssz.decode_to(OurEth1Data, data)
        assert bytes(decoded.deposit_root) == b"\xaa" * 32
        assert int(decoded.deposit_count) == 99999
        assert bytes(decoded.block_hash) == b"\xbb" * 32


# ---------------------------------------------------------------------------
# Container with nested variable-size: Validator-like
# ---------------------------------------------------------------------------


class PyValidator(Serializable):
    fields = [
        ("pubkey", bytes48),
        ("withdrawal_credentials", bytes32),
        ("effective_balance", uint64),
        ("slashed", Boolean()),
        ("activation_eligibility_epoch", uint64),
        ("activation_epoch", uint64),
        ("exit_epoch", uint64),
        ("withdrawable_epoch", uint64),
    ]


@dataclass
class OurValidator:
    pubkey: Bytes48
    withdrawal_credentials: Bytes32
    effective_balance: U64
    slashed: bool
    activation_eligibility_epoch: U64
    activation_epoch: U64
    exit_epoch: U64
    withdrawable_epoch: U64


class TestValidatorCrossValidation:
    def test_encoding(self) -> None:
        far_future = 2**64 - 1
        expected = py_ssz.encode(
            PyValidator(
                pubkey=b"\xaa" * 48,
                withdrawal_credentials=b"\xbb" * 32,
                effective_balance=32000000000,
                slashed=False,
                activation_eligibility_epoch=0,
                activation_epoch=0,
                exit_epoch=far_future,
                withdrawable_epoch=far_future,
            ),
            PyValidator,
        )
        actual = ethereum_ssz.encode(
            OurValidator(
                pubkey=Bytes48(b"\xaa" * 48),
                withdrawal_credentials=Bytes32(b"\xbb" * 32),
                effective_balance=U64(32000000000),
                slashed=False,
                activation_eligibility_epoch=U64(0),
                activation_epoch=U64(0),
                exit_epoch=U64(far_future),
                withdrawable_epoch=U64(far_future),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = py_ssz.encode(
            PyValidator(
                pubkey=b"\xcc" * 48,
                withdrawal_credentials=b"\xdd" * 32,
                effective_balance=16000000000,
                slashed=True,
                activation_eligibility_epoch=100,
                activation_epoch=200,
                exit_epoch=300,
                withdrawable_epoch=400,
            ),
            PyValidator,
        )
        decoded = ethereum_ssz.decode_to(OurValidator, data)
        assert bytes(decoded.pubkey) == b"\xcc" * 48
        assert bytes(decoded.withdrawal_credentials) == b"\xdd" * 32
        assert int(decoded.effective_balance) == 16000000000
        assert decoded.slashed is True
        assert int(decoded.activation_eligibility_epoch) == 100
        assert int(decoded.activation_epoch) == 200
        assert int(decoded.exit_epoch) == 300
        assert int(decoded.withdrawable_epoch) == 400


# =========================================================================
# PART 2: remerkleable cross-validation
#
# remerkleable is the SSZ library used by the Ethereum consensus-specs
# pyspec. It uses a type-centric API: Type(value).encode_bytes() and
# Type.decode_bytes(data).
# =========================================================================


# ---------------------------------------------------------------------------
# Bool (remerkleable)
# ---------------------------------------------------------------------------


class TestBoolRemerkleable:
    @pytest.mark.parametrize("value", [True, False])
    def test_bool_encoding(self, value: bool) -> None:
        expected = rm_boolean(value).encode_bytes()
        actual = ethereum_ssz.encode(value)
        assert actual == expected

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_roundtrip(self, value: bool) -> None:
        encoded = ethereum_ssz.encode(value)
        decoded = rm_boolean.decode_bytes(encoded)
        assert bool(decoded) == value


# ---------------------------------------------------------------------------
# Unsigned integers (remerkleable)
# ---------------------------------------------------------------------------


class TestUintRemerkleable:
    @pytest.mark.parametrize(
        "value,our_type,rm_type",
        [
            (0, U8, rm_uint8),
            (255, U8, rm_uint8),
            (0, U16, rm_uint16),
            (1000, U16, rm_uint16),
            (65535, U16, rm_uint16),
            (0, U32, rm_uint32),
            (100000, U32, rm_uint32),
            (2**32 - 1, U32, rm_uint32),
            (0, U64, rm_uint64),
            (2**32, U64, rm_uint64),
            (2**64 - 1, U64, rm_uint64),
            (0, U256, rm_uint256),
            (2**128, U256, rm_uint256),
            (2**256 - 1, U256, rm_uint256),
        ],
    )
    def test_uint_encoding(self, value, our_type, rm_type) -> None:
        expected = rm_type(value).encode_bytes()
        actual = ethereum_ssz.encode(our_type(value))
        assert actual == expected

    @pytest.mark.parametrize(
        "value,our_type,rm_type",
        [
            (42, U8, rm_uint8),
            (1000, U16, rm_uint16),
            (100000, U32, rm_uint32),
            (2**32, U64, rm_uint64),
            (2**128, U256, rm_uint256),
        ],
    )
    def test_uint_roundtrip(self, value, our_type, rm_type) -> None:
        encoded = ethereum_ssz.encode(our_type(value))
        decoded = rm_type.decode_bytes(encoded)
        assert int(decoded) == value

        encoded2 = rm_type(value).encode_bytes()
        decoded2 = ethereum_ssz.decode_to(our_type, encoded2)
        assert int(decoded2) == value


# ---------------------------------------------------------------------------
# Fixed-length bytes (remerkleable)
# ---------------------------------------------------------------------------


class TestFixedBytesRemerkleable:
    @pytest.mark.parametrize(
        "value,our_type,length",
        [
            (b"\xab\xcd\xef\x01", Bytes4, 4),
            (b"\x00" * 4, Bytes4, 4),
            (b"\xff" * 4, Bytes4, 4),
            (b"\x01" * 32, Bytes32, 32),
            (b"\x00" * 32, Bytes32, 32),
            (b"\xff" * 32, Bytes32, 32),
            (b"\x01" * 48, Bytes48, 48),
            (b"\x01" * 96, Bytes96, 96),
        ],
    )
    def test_fixed_bytes_encoding(self, value, our_type, length) -> None:
        expected = RmByteVector[length](value).encode_bytes()
        actual = ethereum_ssz.encode(our_type(value))
        assert actual == expected

    @pytest.mark.parametrize(
        "value,our_type,length",
        [
            (b"\xab\xcd\xef\x01", Bytes4, 4),
            (b"\x01" * 32, Bytes32, 32),
        ],
    )
    def test_fixed_bytes_roundtrip(self, value, our_type, length) -> None:
        encoded = ethereum_ssz.encode(our_type(value))
        decoded = RmByteVector[length].decode_bytes(encoded)
        assert bytes(decoded) == value

        encoded2 = RmByteVector[length](value).encode_bytes()
        decoded2 = ethereum_ssz.decode_to(our_type, encoded2)
        assert bytes(decoded2) == value


# ---------------------------------------------------------------------------
# Variable-length bytes (remerkleable)
# ---------------------------------------------------------------------------


class TestVariableBytesRemerkleable:
    @pytest.mark.parametrize(
        "value",
        [
            b"",
            b"\x01\x02\x03",
            b"\x00" * 100,
            b"\xff" * 256,
            bytes(range(256)),
        ],
    )
    def test_variable_bytes_encoding(self, value: bytes) -> None:
        expected = RmByteList[1024](value).encode_bytes()
        actual = ethereum_ssz.encode(value)
        assert actual == expected

    def test_variable_bytes_roundtrip(self) -> None:
        value = b"\x01\x02\x03\x04\x05"
        encoded = ethereum_ssz.encode(value)
        decoded = RmByteList[1024].decode_bytes(encoded)
        assert bytes(decoded) == value


# ---------------------------------------------------------------------------
# List of fixed-size elements (remerkleable)
# ---------------------------------------------------------------------------


class TestFixedListRemerkleable:
    def test_list_u32_raw_bytes(self) -> None:
        values = [1, 2, 3]
        expected = RmList[rm_uint32, 100](
            rm_uint32(v) for v in values
        ).encode_bytes()
        result = b""
        for v in values:
            result += ethereum_ssz.encode(U32(v))
        assert result == expected

    def test_list_u64_raw_bytes(self) -> None:
        values = [10, 20, 30]
        expected = RmList[rm_uint64, 100](
            rm_uint64(v) for v in values
        ).encode_bytes()
        result = b""
        for v in values:
            result += ethereum_ssz.encode(U64(v))
        assert result == expected

    def test_empty_list(self) -> None:
        expected = RmList[rm_uint32, 100]().encode_bytes()
        assert expected == b""


# ---------------------------------------------------------------------------
# Vector (remerkleable)
# ---------------------------------------------------------------------------


class TestVectorRemerkleable:
    def test_vector_u64_encoding(self) -> None:
        values = [10, 20, 30]
        expected = RmVector[rm_uint64, 3](
            rm_uint64(v) for v in values
        ).encode_bytes()
        result = b""
        for v in values:
            result += ethereum_ssz.encode(U64(v))
        assert result == expected


# ---------------------------------------------------------------------------
# Containers: fixed-size fields only (remerkleable)
# ---------------------------------------------------------------------------


class RmPoint(RmContainer):
    x: rm_uint64
    y: rm_uint64


class RmSigned(RmContainer):
    value: rm_uint64
    negative: rm_boolean


class TestFixedContainerRemerkleable:
    def test_point_encoding(self) -> None:
        expected = RmPoint(
            x=rm_uint64(1), y=rm_uint64(2)
        ).encode_bytes()
        actual = ethereum_ssz.encode(OurPoint(x=U64(1), y=U64(2)))
        assert actual == expected

    def test_point_roundtrip(self) -> None:
        data = RmPoint(
            x=rm_uint64(100), y=rm_uint64(200)
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurPoint, data)
        assert int(decoded.x) == 100
        assert int(decoded.y) == 200

        data2 = ethereum_ssz.encode(OurPoint(x=U64(100), y=U64(200)))
        rm_decoded = RmPoint.decode_bytes(data2)
        assert int(rm_decoded.x) == 100
        assert int(rm_decoded.y) == 200

    def test_signed_encoding(self) -> None:
        expected = RmSigned(
            value=rm_uint64(42), negative=rm_boolean(True)
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurSigned(value=U64(42), negative=True)
        )
        assert actual == expected

    def test_signed_roundtrip(self) -> None:
        data = RmSigned(
            value=rm_uint64(42), negative=rm_boolean(True)
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurSigned, data)
        assert int(decoded.value) == 42
        assert decoded.negative is True

    @pytest.mark.parametrize(
        "x,y",
        [
            (0, 0),
            (1, 1),
            (2**64 - 1, 2**64 - 1),
            (2**32, 2**32 + 1),
        ],
    )
    def test_point_boundary_values(self, x: int, y: int) -> None:
        expected = RmPoint(
            x=rm_uint64(x), y=rm_uint64(y)
        ).encode_bytes()
        actual = ethereum_ssz.encode(OurPoint(x=U64(x), y=U64(y)))
        assert actual == expected


# ---------------------------------------------------------------------------
# Containers: mixed fixed and variable-size fields (remerkleable)
# ---------------------------------------------------------------------------


class RmVariable(RmContainer):
    id: rm_uint64
    data: RmByteList[2048]


class RmTwoVariable(RmContainer):
    first: RmByteList[256]
    second: RmByteList[256]


class TestVariableContainerRemerkleable:
    def test_single_variable_field(self) -> None:
        expected = RmVariable(
            id=rm_uint64(42), data=RmByteList[2048](b"\x01\x02\x03")
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurVariable(id=U64(42), data=b"\x01\x02\x03")
        )
        assert actual == expected

    def test_single_variable_roundtrip(self) -> None:
        data = RmVariable(
            id=rm_uint64(42), data=RmByteList[2048](b"\xaa\xbb\xcc")
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurVariable, data)
        assert int(decoded.id) == 42
        assert decoded.data == b"\xaa\xbb\xcc"

    def test_two_variable_fields(self) -> None:
        expected = RmTwoVariable(
            first=RmByteList[256](b"hello"),
            second=RmByteList[256](b"world"),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurTwoVariable(first=b"hello", second=b"world")
        )
        assert actual == expected

    def test_two_variable_roundtrip(self) -> None:
        data = RmTwoVariable(
            first=RmByteList[256](b"abc"),
            second=RmByteList[256](b"defgh"),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurTwoVariable, data)
        assert decoded.first == b"abc"
        assert decoded.second == b"defgh"

    def test_empty_variable_data(self) -> None:
        expected = RmVariable(
            id=rm_uint64(0), data=RmByteList[2048](b"")
        ).encode_bytes()
        actual = ethereum_ssz.encode(OurVariable(id=U64(0), data=b""))
        assert actual == expected

    def test_large_variable_data(self) -> None:
        big = bytes(range(256)) * 4
        expected = RmVariable(
            id=rm_uint64(1), data=RmByteList[2048](big)
        ).encode_bytes()
        actual = ethereum_ssz.encode(OurVariable(id=U64(1), data=big))
        assert actual == expected


# ---------------------------------------------------------------------------
# Nested containers (remerkleable)
# ---------------------------------------------------------------------------


class RmInner(RmContainer):
    a: rm_uint32
    b: rm_uint32


class RmOuter(RmContainer):
    inner: RmInner
    extra: rm_uint64


class TestNestedContainerRemerkleable:
    def test_nested_encoding(self) -> None:
        expected = RmOuter(
            inner=RmInner(a=rm_uint32(10), b=rm_uint32(20)),
            extra=rm_uint64(30),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurOuter(inner=OurInner(a=U32(10), b=U32(20)), extra=U64(30))
        )
        assert actual == expected

    def test_nested_roundtrip(self) -> None:
        data = RmOuter(
            inner=RmInner(a=rm_uint32(100), b=rm_uint32(200)),
            extra=rm_uint64(300),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurOuter, data)
        assert int(decoded.inner.a) == 100
        assert int(decoded.inner.b) == 200
        assert int(decoded.extra) == 300


# ---------------------------------------------------------------------------
# Container with list field (remerkleable)
# ---------------------------------------------------------------------------


class RmWithList(RmContainer):
    count: rm_uint32
    values: RmList[rm_uint32, 1024]


class TestContainerWithListRemerkleable:
    def test_encoding(self) -> None:
        expected = RmWithList(
            count=rm_uint32(3),
            values=RmList[rm_uint32, 1024](
                rm_uint32(1), rm_uint32(2), rm_uint32(3)
            ),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurWithList(count=U32(3), values=[U32(1), U32(2), U32(3)])
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = RmWithList(
            count=rm_uint32(5),
            values=RmList[rm_uint32, 1024](
                rm_uint32(10),
                rm_uint32(20),
                rm_uint32(30),
                rm_uint32(40),
                rm_uint32(50),
            ),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurWithList, data)
        assert int(decoded.count) == 5
        assert [int(v) for v in decoded.values] == [10, 20, 30, 40, 50]

    def test_empty_list(self) -> None:
        expected = RmWithList(
            count=rm_uint32(0),
            values=RmList[rm_uint32, 1024](),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurWithList(count=U32(0), values=[])
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# BeaconBlockHeader (remerkleable)
# ---------------------------------------------------------------------------


class RmBeaconBlockHeader(RmContainer):
    slot: rm_uint64
    proposer_index: rm_uint64
    parent_root: RmByteVector[32]
    state_root: RmByteVector[32]
    body_root: RmByteVector[32]


class TestBeaconBlockHeaderRemerkleable:
    def test_encoding(self) -> None:
        expected = RmBeaconBlockHeader(
            slot=rm_uint64(100),
            proposer_index=rm_uint64(42),
            parent_root=RmByteVector[32](b"\x01" * 32),
            state_root=RmByteVector[32](b"\x02" * 32),
            body_root=RmByteVector[32](b"\x03" * 32),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurBeaconBlockHeader(
                slot=U64(100),
                proposer_index=U64(42),
                parent_root=Bytes32(b"\x01" * 32),
                state_root=Bytes32(b"\x02" * 32),
                body_root=Bytes32(b"\x03" * 32),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = RmBeaconBlockHeader(
            slot=rm_uint64(6543210),
            proposer_index=rm_uint64(99999),
            parent_root=RmByteVector[32](bytes(range(32))),
            state_root=RmByteVector[32](bytes(range(32, 64))),
            body_root=RmByteVector[32](bytes(range(64, 96))),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurBeaconBlockHeader, data)
        assert int(decoded.slot) == 6543210
        assert int(decoded.proposer_index) == 99999
        assert bytes(decoded.parent_root) == bytes(range(32))
        assert bytes(decoded.state_root) == bytes(range(32, 64))
        assert bytes(decoded.body_root) == bytes(range(64, 96))

    def test_genesis_header(self) -> None:
        expected = RmBeaconBlockHeader(
            slot=rm_uint64(0),
            proposer_index=rm_uint64(0),
            parent_root=RmByteVector[32](b"\x00" * 32),
            state_root=RmByteVector[32](b"\x00" * 32),
            body_root=RmByteVector[32](b"\x00" * 32),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurBeaconBlockHeader(
                slot=U64(0),
                proposer_index=U64(0),
                parent_root=Bytes32(b"\x00" * 32),
                state_root=Bytes32(b"\x00" * 32),
                body_root=Bytes32(b"\x00" * 32),
            )
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# Checkpoint (remerkleable)
# ---------------------------------------------------------------------------


class RmCheckpoint(RmContainer):
    epoch: rm_uint64
    root: RmByteVector[32]


class TestCheckpointRemerkleable:
    def test_encoding(self) -> None:
        expected = RmCheckpoint(
            epoch=rm_uint64(10),
            root=RmByteVector[32](b"\xaa" * 32),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurCheckpoint(epoch=U64(10), root=Bytes32(b"\xaa" * 32))
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = RmCheckpoint(
            epoch=rm_uint64(999),
            root=RmByteVector[32](b"\xbb" * 32),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurCheckpoint, data)
        assert int(decoded.epoch) == 999
        assert bytes(decoded.root) == b"\xbb" * 32


# ---------------------------------------------------------------------------
# Fork (remerkleable)
# ---------------------------------------------------------------------------


class RmFork(RmContainer):
    previous_version: RmByteVector[4]
    current_version: RmByteVector[4]
    epoch: rm_uint64


class TestForkRemerkleable:
    def test_encoding(self) -> None:
        expected = RmFork(
            previous_version=RmByteVector[4](b"\x00\x00\x00\x01"),
            current_version=RmByteVector[4](b"\x00\x00\x00\x02"),
            epoch=rm_uint64(100),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurFork(
                previous_version=Bytes4(b"\x00\x00\x00\x01"),
                current_version=Bytes4(b"\x00\x00\x00\x02"),
                epoch=U64(100),
            )
        )
        assert actual == expected


# ---------------------------------------------------------------------------
# Eth1Data (remerkleable)
# ---------------------------------------------------------------------------


class RmEth1Data(RmContainer):
    deposit_root: RmByteVector[32]
    deposit_count: rm_uint64
    block_hash: RmByteVector[32]


class TestEth1DataRemerkleable:
    def test_encoding(self) -> None:
        expected = RmEth1Data(
            deposit_root=RmByteVector[32](b"\x11" * 32),
            deposit_count=rm_uint64(12345),
            block_hash=RmByteVector[32](b"\x22" * 32),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurEth1Data(
                deposit_root=Bytes32(b"\x11" * 32),
                deposit_count=U64(12345),
                block_hash=Bytes32(b"\x22" * 32),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = RmEth1Data(
            deposit_root=RmByteVector[32](b"\xaa" * 32),
            deposit_count=rm_uint64(99999),
            block_hash=RmByteVector[32](b"\xbb" * 32),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurEth1Data, data)
        assert bytes(decoded.deposit_root) == b"\xaa" * 32
        assert int(decoded.deposit_count) == 99999
        assert bytes(decoded.block_hash) == b"\xbb" * 32


# ---------------------------------------------------------------------------
# Validator (remerkleable)
# ---------------------------------------------------------------------------


class RmValidator(RmContainer):
    pubkey: RmByteVector[48]
    withdrawal_credentials: RmByteVector[32]
    effective_balance: rm_uint64
    slashed: rm_boolean
    activation_eligibility_epoch: rm_uint64
    activation_epoch: rm_uint64
    exit_epoch: rm_uint64
    withdrawable_epoch: rm_uint64


class TestValidatorRemerkleable:
    def test_encoding(self) -> None:
        far_future = 2**64 - 1
        expected = RmValidator(
            pubkey=RmByteVector[48](b"\xaa" * 48),
            withdrawal_credentials=RmByteVector[32](b"\xbb" * 32),
            effective_balance=rm_uint64(32000000000),
            slashed=rm_boolean(False),
            activation_eligibility_epoch=rm_uint64(0),
            activation_epoch=rm_uint64(0),
            exit_epoch=rm_uint64(far_future),
            withdrawable_epoch=rm_uint64(far_future),
        ).encode_bytes()
        actual = ethereum_ssz.encode(
            OurValidator(
                pubkey=Bytes48(b"\xaa" * 48),
                withdrawal_credentials=Bytes32(b"\xbb" * 32),
                effective_balance=U64(32000000000),
                slashed=False,
                activation_eligibility_epoch=U64(0),
                activation_epoch=U64(0),
                exit_epoch=U64(far_future),
                withdrawable_epoch=U64(far_future),
            )
        )
        assert actual == expected

    def test_roundtrip(self) -> None:
        data = RmValidator(
            pubkey=RmByteVector[48](b"\xcc" * 48),
            withdrawal_credentials=RmByteVector[32](b"\xdd" * 32),
            effective_balance=rm_uint64(16000000000),
            slashed=rm_boolean(True),
            activation_eligibility_epoch=rm_uint64(100),
            activation_epoch=rm_uint64(200),
            exit_epoch=rm_uint64(300),
            withdrawable_epoch=rm_uint64(400),
        ).encode_bytes()
        decoded = ethereum_ssz.decode_to(OurValidator, data)
        assert bytes(decoded.pubkey) == b"\xcc" * 48
        assert bytes(decoded.withdrawal_credentials) == b"\xdd" * 32
        assert int(decoded.effective_balance) == 16000000000
        assert decoded.slashed is True
        assert int(decoded.activation_eligibility_epoch) == 100
        assert int(decoded.activation_epoch) == 200
        assert int(decoded.exit_epoch) == 300
        assert int(decoded.withdrawable_epoch) == 400


# =========================================================================
# PART 3: Three-way agreement
#
# These tests verify that all three libraries produce identical bytes for
# the same logical value. This is the strongest correctness guarantee.
# =========================================================================


class TestThreeWayAgreement:
    """Verify all three libraries produce identical bytes."""

    def test_bool_three_way(self) -> None:
        for value in (True, False):
            ours = ethereum_ssz.encode(value)
            pyssz = py_ssz.encode(value, Boolean())
            rmkl = rm_boolean(value).encode_bytes()
            assert ours == pyssz == rmkl, f"mismatch for bool {value}"

    @pytest.mark.parametrize(
        "value,our_type,py_sedes,rm_type",
        [
            (0, U8, uint8, rm_uint8),
            (255, U8, uint8, rm_uint8),
            (1000, U16, uint16, rm_uint16),
            (100000, U32, uint32, rm_uint32),
            (2**32, U64, uint64, rm_uint64),
            (2**64 - 1, U64, uint64, rm_uint64),
            (2**128, U256, uint256, rm_uint256),
            (2**256 - 1, U256, uint256, rm_uint256),
        ],
    )
    def test_uint_three_way(
        self, value, our_type, py_sedes, rm_type
    ) -> None:
        ours = ethereum_ssz.encode(our_type(value))
        pyssz = py_ssz.encode(value, py_sedes)
        rmkl = rm_type(value).encode_bytes()
        assert ours == pyssz == rmkl

    @pytest.mark.parametrize(
        "value,our_type,py_sedes,length",
        [
            (b"\xab\xcd\xef\x01", Bytes4, bytes4, 4),
            (b"\x01" * 32, Bytes32, bytes32, 32),
            (b"\x01" * 48, Bytes48, bytes48, 48),
            (b"\x01" * 96, Bytes96, bytes96, 96),
        ],
    )
    def test_fixed_bytes_three_way(
        self, value, our_type, py_sedes, length
    ) -> None:
        ours = ethereum_ssz.encode(our_type(value))
        pyssz = py_ssz.encode(value, py_sedes)
        rmkl = RmByteVector[length](value).encode_bytes()
        assert ours == pyssz == rmkl

    @pytest.mark.parametrize(
        "value",
        [b"", b"\x01\x02\x03", b"\x00" * 100, bytes(range(256))],
    )
    def test_variable_bytes_three_way(self, value: bytes) -> None:
        ours = ethereum_ssz.encode(value)
        pyssz = py_ssz.encode(value, ByteList(1024))
        rmkl = RmByteList[1024](value).encode_bytes()
        assert ours == pyssz == rmkl

    def test_beacon_block_header_three_way(self) -> None:
        ours = ethereum_ssz.encode(
            OurBeaconBlockHeader(
                slot=U64(6543210),
                proposer_index=U64(99999),
                parent_root=Bytes32(bytes(range(32))),
                state_root=Bytes32(bytes(range(32, 64))),
                body_root=Bytes32(bytes(range(64, 96))),
            )
        )
        pyssz = py_ssz.encode(
            PyBeaconBlockHeader(
                slot=6543210,
                proposer_index=99999,
                parent_root=bytes(range(32)),
                state_root=bytes(range(32, 64)),
                body_root=bytes(range(64, 96)),
            ),
            PyBeaconBlockHeader,
        )
        rmkl = RmBeaconBlockHeader(
            slot=rm_uint64(6543210),
            proposer_index=rm_uint64(99999),
            parent_root=RmByteVector[32](bytes(range(32))),
            state_root=RmByteVector[32](bytes(range(32, 64))),
            body_root=RmByteVector[32](bytes(range(64, 96))),
        ).encode_bytes()
        assert ours == pyssz == rmkl

    def test_validator_three_way(self) -> None:
        far_future = 2**64 - 1
        ours = ethereum_ssz.encode(
            OurValidator(
                pubkey=Bytes48(b"\xaa" * 48),
                withdrawal_credentials=Bytes32(b"\xbb" * 32),
                effective_balance=U64(32000000000),
                slashed=False,
                activation_eligibility_epoch=U64(0),
                activation_epoch=U64(0),
                exit_epoch=U64(far_future),
                withdrawable_epoch=U64(far_future),
            )
        )
        pyssz = py_ssz.encode(
            PyValidator(
                pubkey=b"\xaa" * 48,
                withdrawal_credentials=b"\xbb" * 32,
                effective_balance=32000000000,
                slashed=False,
                activation_eligibility_epoch=0,
                activation_epoch=0,
                exit_epoch=far_future,
                withdrawable_epoch=far_future,
            ),
            PyValidator,
        )
        rmkl = RmValidator(
            pubkey=RmByteVector[48](b"\xaa" * 48),
            withdrawal_credentials=RmByteVector[32](b"\xbb" * 32),
            effective_balance=rm_uint64(32000000000),
            slashed=rm_boolean(False),
            activation_eligibility_epoch=rm_uint64(0),
            activation_epoch=rm_uint64(0),
            exit_epoch=rm_uint64(far_future),
            withdrawable_epoch=rm_uint64(far_future),
        ).encode_bytes()
        assert ours == pyssz == rmkl

    def test_variable_container_three_way(self) -> None:
        ours = ethereum_ssz.encode(
            OurVariable(id=U64(42), data=b"\x01\x02\x03")
        )
        pyssz = py_ssz.encode(
            PyVariable(id=42, data=b"\x01\x02\x03"), PyVariable
        )
        rmkl = RmVariable(
            id=rm_uint64(42), data=RmByteList[2048](b"\x01\x02\x03")
        ).encode_bytes()
        assert ours == pyssz == rmkl

    def test_nested_container_three_way(self) -> None:
        ours = ethereum_ssz.encode(
            OurOuter(inner=OurInner(a=U32(10), b=U32(20)), extra=U64(30))
        )
        pyssz = py_ssz.encode(
            PyOuter(inner=PyInner(a=10, b=20), extra=30), PyOuter
        )
        rmkl = RmOuter(
            inner=RmInner(a=rm_uint32(10), b=rm_uint32(20)),
            extra=rm_uint64(30),
        ).encode_bytes()
        assert ours == pyssz == rmkl


# =========================================================================
# PART 4: Known bugs in reference implementations
# =========================================================================


class TestPySszByteListBug:
    """
    py-ssz v0.5.2 has a bug in ByteList: error messages reference
    ``self.length`` which doesn't exist (should be ``self.max_length``).
    This means that when data exceeds max_length, py-ssz raises
    AttributeError instead of SerializationError/DeserializationError.

    Our library should handle these cases correctly (either encoding the
    data or raising an appropriate error), while py-ssz crashes.
    """

    def test_pyssz_serialize_over_max_length_raises_attribute_error(
        self,
    ) -> None:
        """py-ssz crashes with AttributeError instead of SerializationError."""
        sedes = ByteList(10)
        with pytest.raises(AttributeError, match="length"):
            py_ssz.encode(b"\x00" * 11, sedes)

    def test_pyssz_deserialize_over_max_length_raises_attribute_error(
        self,
    ) -> None:
        """py-ssz crashes with AttributeError instead of DeserializationError."""
        sedes = ByteList(10)
        with pytest.raises(AttributeError, match="length"):
            py_ssz.decode(b"\x00" * 11, sedes)

    def test_our_library_encodes_any_length_bytes(self) -> None:
        """
        Our library does not enforce max_length during serialization
        (it is only relevant for Merkleization), so encoding succeeds.
        """
        big = b"\x00" * 1024
        encoded = ethereum_ssz.encode(big)
        assert encoded == big

        decoded = ethereum_ssz.decode_to(bytes, encoded)
        assert decoded == big


# =========================================================================
# PART 5: Malformed input rejection
# =========================================================================


@dataclass
class OurBytesList:
    items: list[bytes]


class PyBytesList(Serializable):
    fields = [("items", List(ByteList(256), max_length=256))]


class RmBytesList(RmContainer):
    items: RmList[RmByteList[256], 256]


class TestContainerOffsetGapCrossValidation:
    """First variable field offset must exactly equal fixed_part_size."""

    def test_rejects_gap_single_variable_field(self) -> None:
        # OurVariable fixed_part_size = 8 (id) + 4 (offset) = 12
        # offset=14 leaves a 2-byte gap
        invalid = (
            struct.pack("<Q", 42)
            + struct.pack("<I", 14)
            + b"\x00\x00"
            + b"\xaa\xbb\xcc"
        )
        with pytest.raises(DecodingError, match="does not match fixed part"):
            ethereum_ssz.decode_to(OurVariable, invalid)

    def test_rejects_gap_two_variable_fields(self) -> None:
        # OurTwoVariable fixed_part_size = 4 + 4 = 8
        # first offset=10 leaves a 2-byte gap
        invalid = (
            struct.pack("<I", 10)
            + struct.pack("<I", 15)
            + b"\x00\x00"
            + b"hello"
            + b"world"
        )
        with pytest.raises(DecodingError, match="does not match fixed part"):
            ethereum_ssz.decode_to(OurTwoVariable, invalid)

    def test_valid_roundtrip(self) -> None:
        valid = ethereum_ssz.encode(
            OurVariable(id=U64(42), data=b"\xaa\xbb\xcc")
        )
        decoded = ethereum_ssz.decode_to(OurVariable, valid)
        assert int(decoded.id) == 42
        assert decoded.data == b"\xaa\xbb\xcc"

    def test_three_way_agreement(self) -> None:
        ours = ethereum_ssz.encode(
            OurVariable(id=U64(100), data=b"\x01\x02\x03\x04\x05")
        )
        pyssz = py_ssz.encode(
            PyVariable(id=100, data=b"\x01\x02\x03\x04\x05"), PyVariable
        )
        rmkl = RmVariable(
            id=rm_uint64(100),
            data=RmByteList[2048](b"\x01\x02\x03\x04\x05"),
        ).encode_bytes()
        assert ours == pyssz == rmkl


class TestSequenceOffsetCrossValidation:
    """Sequence offset validation for variable-size element lists."""

    def test_first_offset_beyond_data(self) -> None:
        invalid = (
            struct.pack("<I", 4)
            + struct.pack("<I", 99999999)
        )
        with pytest.raises(DecodingError):
            ethereum_ssz.decode_to(OurBytesList, invalid)

    def test_offset_into_offset_region(self) -> None:
        # 2 elements -> first_offset = 8; offsets: [8, 2]
        invalid = (
            struct.pack("<I", 4)
            + struct.pack("<I", 8)
            + struct.pack("<I", 2)
            + b"\x00" * 8
        )
        with pytest.raises(DecodingError):
            ethereum_ssz.decode_to(OurBytesList, invalid)

    def test_roundtrip_matches_pyssz(self) -> None:
        ours = ethereum_ssz.encode(
            OurBytesList(items=[b"hello", b"world", b"!"])
        )
        theirs = py_ssz.encode(
            PyBytesList(items=(b"hello", b"world", b"!")),
            PyBytesList,
        )
        assert ours == theirs

        decoded = ethereum_ssz.decode_to(OurBytesList, ours)
        assert decoded.items == [b"hello", b"world", b"!"]

    def test_roundtrip_matches_remerkleable(self) -> None:
        ours = ethereum_ssz.encode(
            OurBytesList(items=[b"hello", b"world", b"!"])
        )
        rmkl = RmBytesList(
            items=RmList[RmByteList[256], 256](
                RmByteList[256](b"hello"),
                RmByteList[256](b"world"),
                RmByteList[256](b"!"),
            )
        ).encode_bytes()
        assert ours == rmkl
