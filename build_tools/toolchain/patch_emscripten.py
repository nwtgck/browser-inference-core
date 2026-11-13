"""Enable upstream Asyncify argument preservation for Wasm32 bigint exports."""
import hashlib
from pathlib import Path

ASYNCIFY_SOURCE = Path('src/lib/libasync.js')


def patched_asyncify_source(source: bytes, patch: dict) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == patch['patchedSha256']:
        return source
    if digest != patch['sourceSha256']:
        raise RuntimeError('Unexpected Emscripten Asyncify source; refusing to patch it')
    # Emscripten 6.0.9 already restores entry arguments for Memory64. Wasm32
    # exports with i64 parameters need the same logic: undefined cannot become bigint.
    replacements = (
        (b'#if ASYNCIFY == 1 && MEMORY64\n', b'#if ASYNCIFY == 1 && (MEMORY64 || WASM_BIGINT)\n'),
        (b'#if MEMORY64\n', b'#if MEMORY64 || WASM_BIGINT\n'),
    )
    result = source
    for before, after in replacements:
        if result.count(before) != 2:
            raise RuntimeError('Unexpected Asyncify argument-preservation guards')
        result = result.replace(before, after)
    if hashlib.sha256(result).hexdigest() != patch['patchedSha256']:
        raise RuntimeError('Patched Emscripten Asyncify source does not match its pin')
    return result


def apply_asyncify_bigint_patch(emscripten: Path, patch: dict):
    file = emscripten / ASYNCIFY_SOURCE
    source = file.read_bytes()
    result = patched_asyncify_source(source, patch)
    if result != source:
        file.write_bytes(result)


def verify_asyncify_bigint_patch(emscripten: Path, patch: dict):
    file = emscripten / ASYNCIFY_SOURCE
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != patch['patchedSha256']:
        raise RuntimeError('Asyncify requires the pinned bigint rewind patch; prepare the pinned local toolchain with python3 -m build_tools.toolchain.prepare')
