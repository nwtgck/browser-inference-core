"""Producer-only binding to the public libzstd C API.

The library is not copied into browser assets; ruzstd is the independent browser
implementation. These named parameter IDs are from zstd.h's stable API. The
installed library version is recorded in the packaging report, not inferred from
an operating-system package name.
"""
import ctypes as C

# Stable ZSTD_cParameter values from zstd.h.
COMPRESSION_LEVEL = 100
WINDOW_LOG = 101
CONTENT_SIZE_FLAG = 200
CHECKSUM_FLAG = 201
MAX_BYTES = 64 * 1024 * 1024

_library = C.CDLL("libzstd.so.1")


def _bind(name, arguments, result):
    function = getattr(_library, name)
    function.argtypes = arguments
    function.restype = result
    return function


_size = C.c_size_t
_pointer = C.c_void_p
_create_encoder = _bind("ZSTD_createCCtx", [], _pointer)
_free_encoder = _bind("ZSTD_freeCCtx", [_pointer], _size)
_create_decoder = _bind("ZSTD_createDCtx", [], _pointer)
_free_decoder = _bind("ZSTD_freeDCtx", [_pointer], _size)
_parameter = _bind("ZSTD_CCtx_setParameter", [_pointer, C.c_int, C.c_int], _size)
_encoder_prefix = _bind("ZSTD_CCtx_refPrefix", [_pointer, _pointer, _size], _size)
_decoder_prefix = _bind("ZSTD_DCtx_refPrefix", [_pointer, _pointer, _size], _size)
_compress = _bind("ZSTD_compress2", [_pointer, _pointer, _size, _pointer, _size], _size)
_decompress = _bind("ZSTD_decompressDCtx", [_pointer, _pointer, _size, _pointer, _size], _size)
_bound = _bind("ZSTD_compressBound", [_size], _size)
_is_error = _bind("ZSTD_isError", [_size], C.c_uint)
_error_name = _bind("ZSTD_getErrorName", [_size], C.c_char_p)
version = _bind("ZSTD_versionString", [], C.c_char_p)().decode()


def _check(result):
    if _is_error(result):
        raise ValueError(_error_name(result).decode())
    return result


def encode(data: bytes, base: bytes = b"", level: int = 19, window: int = 20) -> bytes:
    if len(data) > MAX_BYTES or len(base) > MAX_BYTES:
        raise ValueError("Zstandard input exceeds limit")
    context = _create_encoder()
    if not context:
        raise MemoryError("Cannot allocate Zstandard encoder")
    try:
        for parameter, value in ((COMPRESSION_LEVEL, level), (WINDOW_LOG, window),
                                 (CONTENT_SIZE_FLAG, 1), (CHECKSUM_FLAG, 1)):
            _check(_parameter(context, parameter, value))
        if base:
            _check(_encoder_prefix(context, base, len(base)))
        output = C.create_string_buffer(_bound(len(data)))
        written = _check(_compress(context, output, len(output), data, len(data)))
        result = output.raw[:written]
    finally:
        _free_encoder(context)
    if decode(result, base, len(data)) != data:
        raise ValueError("Zstandard roundtrip mismatch")
    return result


def decode(data: bytes, base: bytes, expected: int) -> bytes:
    if type(expected) is not int or not 0 <= expected <= MAX_BYTES or len(base) > MAX_BYTES:
        raise ValueError("Invalid Zstandard output bound")
    context = _create_decoder()
    if not context:
        raise MemoryError("Cannot allocate Zstandard decoder")
    try:
        if base:
            _check(_decoder_prefix(context, base, len(base)))
        output = C.create_string_buffer(expected)
        written = _check(_decompress(context, output, expected, data, len(data)))
        if written != expected:
            raise ValueError("Zstandard length mismatch")
        return output.raw[:written]
    finally:
        _free_decoder(context)
