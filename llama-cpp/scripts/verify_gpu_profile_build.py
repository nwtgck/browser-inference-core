"""Check the actual backend compile definition, not only a CMake cache option."""
from __future__ import annotations
import json
from pathlib import Path
import shlex


def verify(database: Path, *, webgpu: bool, enabled: bool) -> None:
    if enabled and not webgpu:
        raise ValueError('GPU profiling requires a WebGPU profile')
    entries = json.loads(database.read_text())
    backend = [entry for entry in entries if Path(entry['file']).name == 'ggml-webgpu.cpp']
    if len(backend) != (1 if webgpu else 0):
        raise ValueError('Unexpected WebGPU backend translation units in compile_commands.json')
    for entry in backend:
        arguments = entry.get('arguments') or shlex.split(entry['command'])
        definitions = [arg for arg in arguments if arg.startswith('-DGGML_WEBGPU_GPU_PROFILE')]
        expected = ['-DGGML_WEBGPU_GPU_PROFILE=1'] if enabled else []
        if definitions != expected:
            # #ifdef treats even a definition with value 0 as enabled.
            raise ValueError(f'GPU profiling compile definition mismatch: expected {expected}, got {definitions}')
