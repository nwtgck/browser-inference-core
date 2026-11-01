"""Keep extracted factories on the prepared composition's canonical symbols."""
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_webgpu_aligned_word_load as word


def final_canonical(prepared: Path) -> bool:
    library = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    old = '#include "ggml-wgsl-shaders.hpp"'
    new = '#include "' + word.HEADER_NAME + '"'
    canonical = prepared / word.HEADER_NAME
    if library.count(new) == 1 and old not in library:
        if not canonical.is_file():
            raise ValueError('Final canonical word-loader embedding is missing')
        if hashlib.sha256(canonical.read_bytes()).hexdigest() != word.HEADER_SHA256:
            raise ValueError('Unexpected final canonical word-loader embedding')
        return True
    if library.count(old) == 1 and new not in library and not canonical.exists():
        return False  # Explicit historical include ownership, without a stale final header.
    raise ValueError('Prepared canonical embedding include ownership changed')


def canonical_prefix(prefix: str, prepared: Path) -> str:
    if not final_canonical(prepared):
        return prefix
    old = '#include "ggml-wgsl-shaders.hpp"'
    if prefix.count(old) != 1:
        raise ValueError('Recording prefix canonical include changed')
    return prefix.replace(old, '#include "' + word.HEADER_NAME + '"')


def embedding_sha256(prepared: Path, name: str, historical: str) -> str:
    return word.EMBEDDING_OUTPUTS[name] if final_canonical(prepared) else historical
