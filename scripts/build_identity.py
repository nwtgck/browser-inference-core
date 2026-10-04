"""Build and validation inputs for the image runtime, independent of llama pins.

This is not an artifact trust mechanism. The caller must first bind and validate
an immutable package. Package/report-only changes are intentionally not compiler
inputs; common compiler/toolchain and runtime test changes propagate.
"""
from pathlib import Path
import hashlib
import json
from package_inputs import read_regular, file_identity, digest

SHARED_INPUTS = ('scripts/browser_toolchain.py', 'scripts/patch_emscripten.py',
                 'scripts/setup_toolchain.py', 'scripts/validate_runtime_package.py')

def image_fingerprint(repo: Path, profile: str, variant: str) -> str:
    root = repo / 'stable-diffusion-cpp'
    profiles = json.loads(read_regular(root / 'config/profiles.json'))
    variants = json.loads(read_regular(root / 'config/variants.json'))
    if profile not in profiles or variant not in variants:
        raise ValueError('Unknown image profile/variant')
    inputs = {}
    for folder in ('bridge', 'cmake', 'scripts', 'tests', 'upstream-patches'):
        for path in sorted((root / folder).rglob('*')):
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_file() or path.is_symlink():
                inputs[path.relative_to(repo).as_posix()] = file_identity(path)['sha256']
    for path in [root / 'CMakeLists.txt', root / 'config/upstreams.json',
                 repo / 'scripts/build_identity.py',
                 *(repo / name for name in SHARED_INPUTS),
                 *sorted((repo / 'toolchain').glob('*'))]:
        if path.is_file() or path.is_symlink():
            inputs[path.relative_to(repo).as_posix()] = file_identity(path)['sha256']
    for path in [repo/'.github/workflows/build.yml', *sorted((repo/'.github/actions').rglob('*.yml'))]:
        if path.is_file() or path.is_symlink():
            inputs[path.relative_to(repo).as_posix()] = file_identity(path)['sha256']
    return digest({'formatVersion': 2, 'inputs': inputs, 'profile': profile,
                   'configuration': profiles[profile], 'variant': variant,
                   'variantConfiguration': variants[variant]})
