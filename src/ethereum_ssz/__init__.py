"""
Simple Serialize (SSZ) as used by the Ethereum Consensus Layer.
"""

from .ssz import (  # noqa: F401
    SSZ,
    Extended,
    MaxLength,
    With,
    decode_to,
    encode,
)

__version__ = "0.1.0"
