#!/usr/bin/env python3
"""Prepare an experimental shader-header overlay without modifying vendor/."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'ggml-webgpu-moe-direct-slot.patch'
SHADER_DIRECTORY = 'ggml/src/ggml-webgpu/wgsl-shaders'
SHADER_PATH = SHADER_DIRECTORY + '/mul_mat_id_vec.wgsl'
# Dispatch semantics and generated-header lookup are private upstream contracts.
REVIEWED_REVISIONS = {
    '7fe450e19305b828c199d602c23a8337aaa1f03b': {
        'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': '9264fed8de5cf97248ec423d0887ef0722cf738bf70649a5d5e884d8ab4a6383',
        'ggml/src/ggml-webgpu/CMakeLists.txt': 'aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a',
        'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
        'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '73bd0c2a6b4dc7154826fe659385e9eae777b446623e613f53146656aba73faf',
        'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '27a937b61cb4ae1c1ade89472b1ef35dbb03b9e35780f058638e017b67de75f1',
    },
    'd81235049384534c167caea52b85a694f6103d14': {
        'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': '9264fed8de5cf97248ec423d0887ef0722cf738bf70649a5d5e884d8ab4a6383',
        'ggml/src/ggml-webgpu/CMakeLists.txt': 'aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a',
        'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
        'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'a37cb03d2d85b3ecdcd203dd515f35f22b1ca895934b2aab17066d575299ba84',
        'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a664575b8bcd54e0f300e869b5abac86da6ee237dc3237c93863417b5e7bccb0',
    },
}

PATCH_SHA256 = '00015fcf095dc72747e8479475d037419b15d1b98f0c32723b02ab7e3e47266f'
PATCHED_SHADER_SHA256 = '17641387890d07f0a2e3d33f5b850dcfbf3ffc141107d6e0c9bb8464d0afd6a8'


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked direct-slot input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reviewed_source(source: Path) -> str:
    source = source.resolve()
    top_level = subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=source, text=True).strip()
    if Path(top_level).resolve() != source:
        raise ValueError('MoE direct-slot requires a real upstream Git checkout; '
                         'initialize the pinned llama.cpp submodule')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    inputs = REVIEWED_REVISIONS.get(commit)
    if inputs is None:
        raise ValueError('MoE direct-slot needs semantic review for upstream commit ' + commit)
    for relative, expected in inputs.items():
        if digest(source / relative) != expected:
            raise ValueError('MoE direct-slot needs semantic review: ' + relative + ' at ' + commit)
    return commit


def prepare(source: Path, output: Path, patch: Path) -> dict[str, Path]:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('The direct-slot overlay must be outside the upstream source tree')
    if digest(patch) != PATCH_SHA256:
        raise ValueError('MoE direct-slot patch identity changed; review before building')
    verify_reviewed_source(source)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='moe-direct-slot-', dir=output) as temporary:
        work = Path(temporary)
        shaders = work / SHADER_DIRECTORY
        shutil.copytree(source / SHADER_DIRECTORY, shaders)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags,
                            str(patch.resolve())], cwd=work, check=True, capture_output=True)
        if digest(work / SHADER_PATH) != PATCHED_SHADER_SHA256:
            raise ValueError('Unexpected patched MoE shader identity')
        header = work / 'ggml-wgsl-shaders.hpp'
        subprocess.run([sys.executable, str(source / SHADER_DIRECTORY / 'embed_wgsl.py'),
                        '--input_dir', str(shaders), '--output_file', str(header)], check=True)
        # Publish only complete checked outputs; no-op configure preserves mtimes.
        result = {}
        for key, path in [('shader', work / SHADER_PATH), ('header', header)]:
            target = output / path.name
            if not target.exists() or target.read_bytes() != path.read_bytes():
                path.replace(target)
            result[key] = target
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, ROOT / PATCH_DIRECTORY / PATCH_NAME)


if __name__ == '__main__':
    main()
