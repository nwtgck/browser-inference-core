"""Keep extracted factories on the prepared composition's canonical symbols."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_webgpu_aligned_word_load as word
import prepare_webgpu_subgroup_size_control as optional
import prepare_webgpu_source as source


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
    if not final_canonical(prepared):
        return historical
    helper = prepared / optional.HELPER_NAME
    if helper.exists():
        if hashlib.sha256(helper.read_bytes()).hexdigest() != optional.GENERATED_OUTPUTS[optional.HELPER_NAME]:
            raise ValueError('Unexpected optional-pipeline helper identity')
        return optional.GENERATED_OUTPUTS[name] if name in optional.GENERATED_OUTPUTS else word.EMBEDDING_OUTPUTS[name]
    return word.EMBEDDING_OUTPUTS[name]


def legacy_fixed_sources(prepared: Path) -> dict[str, str]:
    """Reverse only the checked cold stage for SDK-free legacy recording stubs.

    Complete current factories are exercised by webgpu-subgroup-size-control.
    These older stubs record the unchanged fixed/forced paths, not SDK lifetime.
    """
    names = ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp']
    original = {name: (prepared / name).read_text() for name in names}
    if not (prepared / optional.HELPER_NAME).exists():
        return original
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    modes = [profile for profile in (False, True)
             if all(digest(original[name]) == optional.OUTPUTS[profile][name] for name in names)]
    if len(modes) != 1:
        raise ValueError('Legacy fixed fixture needs an exact optional-stage composition')
    profile = modes[0]
    root = Path(__file__).resolve().parents[1]
    patch_name, patch_hash = optional.PATCHES[profile]
    patch_path = root / source.PATCH_DIRECTORY / patch_name
    if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch_hash:
        raise ValueError('Legacy fixed fixture patch identity changed')
    with tempfile.TemporaryDirectory(prefix='legacy-fixed-factory-') as temporary:
        work = Path(temporary)
        for name in names:
            target = work / optional.BASE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original[name])
        subprocess.run(['git', 'apply', '--no-index', '-R', str(patch_path)], cwd=work,
                       env=source.patch_environment(work), check=True, capture_output=True)
        result = {name: (work / optional.BASE / name).read_text() for name in names}
    if any(digest(result[name]) != optional.INPUTS[profile][name] for name in names):
        raise ValueError('Legacy fixed fixture inverse changed unrelated source')
    return result
