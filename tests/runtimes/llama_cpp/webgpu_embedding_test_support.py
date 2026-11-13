"""Keep extracted factories on the prepared composition's canonical symbols."""
import hashlib
import json
import re
from pathlib import Path
from build_tools.project import runtime_root, runtime_test_root
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.support.shader_embedding import compiled_shaders
from tests.support.llama_cpp.webgpu_stages import aligned_word_load as word
from tests.support.llama_cpp.webgpu_stages import subgroup_size_control as optional
from tests.support.llama_cpp.webgpu_stages import q6_subgroup as q6
from tests.support.llama_cpp.webgpu_stages import buffer_limit, batched_write_fallback
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import legacy_webgpu_source as reference_source
def final_canonical(prepared: Path) -> bool:
    library = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    old = '#include "ggml-wgsl-shaders.hpp"'
    new = '#include "' + word.HEADER_NAME + '"'
    canonical = prepared / word.HEADER_NAME
    if library.count(new) == 1 and old not in library:
        if canonical.is_symlink() or not canonical.is_file():
            raise ValueError('Final canonical word-loader embedding is missing')
        # The actual C++ consumer checks symbols/types; numerical tests check WGSL.
        # A historical C++ byte hash would reject harmless generator chunk changes.
        return True
    if library.count(old) == 1 and new not in library and not canonical.exists():
        return False  # Explicit historical include ownership, without a stale final header.
    raise ValueError('Prepared canonical embedding include ownership changed')


def canonical_prefix(prefix: str, prepared: Path) -> str:
    library = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    prefix = matvec_constants(library) + '\n' + prefix
    if not final_canonical(prepared):
        return prefix
    old = '#include "ggml-wgsl-shaders.hpp"'
    if prefix.count(old) != 1:
        raise ValueError('Recording prefix canonical include changed')
    return prefix.replace(old, '#include "' + word.HEADER_NAME + '"')


def matvec_constants(library: str) -> str:
    constants = re.findall(r'^#define GGML_WEBGPU_MUL_MAT_VEC_MAX_ROWS\s+\d+\s*$', library, re.MULTILINE)
    if len(constants) != 1:
        raise ValueError('Matvec row-limit definition changed')
    return constants[0]


def embedding_sha256(prepared: Path, name: str, historical: str) -> str:
    if not final_canonical(prepared):
        return historical
    helper = prepared / optional.HELPER_NAME
    if helper.exists() and hashlib.sha256(helper.read_bytes()).hexdigest() != optional.GENERATED_OUTPUTS[optional.HELPER_NAME]:
        raise ValueError('Unexpected optional-pipeline helper identity')
    stage = 'optional' if helper.exists() else 'aligned'
    root = runtime_root('llama-cpp')
    contract = json.loads((runtime_test_root('llama-cpp') / 'webgpu-embeddings/inputs.json').read_text())['stages'][stage][name]
    with tempfile.TemporaryDirectory(prefix='shader-contract-') as temporary:
        value = compiled_shaders({'actual': prepared / name}, [contract['symbol']], Path(temporary))['actual'][contract['symbol']].encode()
    if len(value) != contract['bytes'] or hashlib.sha256(value).hexdigest() != contract['sha256']:
        raise ValueError('Compiled shader body differs from the numerical fixture: ' + name)
    return hashlib.sha256((prepared / name).read_bytes()).hexdigest()


def legacy_fixed_sources(prepared: Path) -> dict[str, str]:
    """Reverse only the checked cold stage for SDK-free legacy recording stubs.

    Complete current factories are exercised by webgpu-subgroup-size-control.
    These older stubs record the unchanged fixed/forced paths, not SDK lifetime.
    """
    names = ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp']
    original = {name: (prepared / name).read_text() for name in names}
    if not (prepared / optional.HELPER_NAME).exists():
        return original
    # Reverse only the exact final browser write-hook stage before older prefixes.
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    write_modes = [profile for profile in (False, True)
                   if digest(original['ggml-webgpu.cpp']) == batched_write_fallback.OUTPUTS[profile]['ggml-webgpu.cpp']
                   and digest(original['ggml-webgpu-shader-lib.hpp']) == q6.OUTPUTS[profile]['ggml-webgpu-shader-lib.hpp']]
    if write_modes:
        profile, = write_modes
        root = runtime_root('llama-cpp')
        patch_path = root / source.PATCH_DIRECTORY / batched_write_fallback.PATCH_NAME
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != batched_write_fallback.PATCH_SHA256:
            raise ValueError('Legacy batched-write prefix patch identity changed')
        with tempfile.TemporaryDirectory(prefix='legacy-batched-write-prefix-') as temporary:
            work = Path(temporary)
            target = work / batched_write_fallback.BASE / 'ggml-webgpu.cpp'
            target.parent.mkdir(parents=True)
            target.write_text(original[target.name])
            for flags in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '-R', *flags, str(patch_path)], cwd=work,
                               env=source.patch_environment(work), check=True, capture_output=True)
            original[target.name] = target.read_text()
        if digest(original['ggml-webgpu.cpp']) != batched_write_fallback.INPUTS[profile]['ggml-webgpu.cpp']:
            raise ValueError('Legacy batched-write inverse changed unrelated source')
    # Restore the independently checked allocation-limit getter before the
    # historical Q6/optional inverses; reject every unrecognized composition.
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    limit_modes = [profile for profile in (False, True)
                   if digest(original['ggml-webgpu.cpp']) == buffer_limit.OUTPUTS[profile]['ggml-webgpu.cpp']
                   and digest(original['ggml-webgpu-shader-lib.hpp']) == q6.OUTPUTS[profile]['ggml-webgpu-shader-lib.hpp']]
    if limit_modes:
        profile, = limit_modes
        root = runtime_root('llama-cpp')
        patch_path = root / source.PATCH_DIRECTORY / buffer_limit.PATCH_NAME
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != buffer_limit.PATCH_SHA256:
            raise ValueError('Legacy allocation-limit prefix patch identity changed')
        with tempfile.TemporaryDirectory(prefix='legacy-buffer-limit-prefix-') as temporary:
            work = Path(temporary)
            target = work / buffer_limit.BASE / 'ggml-webgpu.cpp'
            target.parent.mkdir(parents=True)
            target.write_text(original[target.name])
            for flags in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '-R', *flags, str(patch_path)], cwd=work,
                               env=source.patch_environment(work), check=True, capture_output=True)
            original[target.name] = target.read_text()
        if digest(original['ggml-webgpu.cpp']) != buffer_limit.INPUTS[profile]['ggml-webgpu.cpp']:
            raise ValueError('Legacy allocation-limit inverse changed unrelated source')
    # The appended Q6 selector has no shader/SDK changes. Restore its exact
    # prefix before reversing the independently checked optional stage.
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    q6_modes = [profile for profile in (False, True)
                if all(digest(original[name]) == q6.OUTPUTS[profile][name] for name in names)]
    if q6_modes:
        profile, = q6_modes
        root = runtime_root('llama-cpp')
        patch_path = root / reference_source.PATCH_DIRECTORY / q6.PATCH_NAME
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != q6.PATCH_SHA256:
            raise ValueError('Legacy Q6 prefix patch identity changed')
        with tempfile.TemporaryDirectory(prefix='legacy-q6-prefix-') as temporary:
            work = Path(temporary)
            for name in names:
                target = work / q6.BASE / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(original[name])
            subprocess.run(['git', 'apply', '--no-index', '-R', str(patch_path)], cwd=work,
                           env=reference_source.patch_environment(work), check=True, capture_output=True)
            original = {name: (work / q6.BASE / name).read_text() for name in names}
        if any(digest(original[name]) != q6.INPUTS[profile][name] for name in names):
            raise ValueError('Legacy Q6 inverse changed unrelated source')
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    modes = [profile for profile in (False, True)
             if all(digest(original[name]) == optional.OUTPUTS[profile][name] for name in names)]
    if len(modes) != 1:
        raise ValueError('Legacy fixed fixture needs an exact optional-stage composition')
    profile = modes[0]
    root = runtime_root('llama-cpp')
    patch_name, patch_hash = optional.PATCHES[profile]
    patch_path = root / reference_source.PATCH_DIRECTORY / patch_name
    if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch_hash:
        raise ValueError('Legacy fixed fixture patch identity changed')
    with tempfile.TemporaryDirectory(prefix='legacy-fixed-factory-') as temporary:
        work = Path(temporary)
        for name in names:
            target = work / optional.BASE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original[name])
        subprocess.run(['git', 'apply', '--no-index', '-R', str(patch_path)], cwd=work,
                       env=reference_source.patch_environment(work), check=True, capture_output=True)
        result = {name: (work / optional.BASE / name).read_text() for name in names}
    if any(digest(result[name]) != optional.INPUTS[profile][name] for name in names):
        raise ValueError('Legacy fixed fixture inverse changed unrelated source')
    return result
