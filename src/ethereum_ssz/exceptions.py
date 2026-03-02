"""
Exceptions that can be thrown while serializing/deserializing SSZ.
"""

from typing_extensions import override


class SSZException(Exception):
    """
    Common base class for all SSZ exceptions.
    """


class DecodingError(SSZException):
    """
    Indicates that SSZ decoding failed.
    """

    @override
    def __str__(self) -> str:
        message = [super().__str__()]
        current: BaseException = self
        while isinstance(current, DecodingError) and current.__cause__:
            current = current.__cause__
            if isinstance(current, DecodingError):
                as_str = super(DecodingError, current).__str__()
            else:
                as_str = str(current)
            message.append(f"\tbecause {as_str}")
        return "\n".join(message)


class EncodingError(SSZException):
    """
    Indicates that SSZ encoding failed.
    """
