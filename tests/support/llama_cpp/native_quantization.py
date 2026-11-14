"""Compile unmodified current quantization code for native numerical references.

The complete C translation unit is checked. A GNU-compatible native linker drops
unused sections, not source text. This does not build or execute WebAssembly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def git_environment() -> dict[str, str]:
    """Read only the named local worktree, without inherited Git overrides."""
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_LAZY_FETCH': '1',
                'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1'})
    return env


def source_commit(upstream: Path) -> str | None:
    """Do not attribute a parent repository's HEAD to a source-only input."""
    upstream = upstream.resolve()
    if not (upstream / '.git').exists():
        return None
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=upstream,
                                  env=git_environment(), text=True).strip()
    if Path(top).resolve() != upstream:
        raise ValueError('Quantization input is not the named Git worktree')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream,
                                   env=git_environment(), text=True).strip()


def build_reference(upstream: Path, output: Path, harness: Path, *, name: str) -> Path:
    upstream, output, harness = upstream.resolve(), output.resolve(), harness.resolve()
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Reference executable name must be a single file name')
    if output == upstream or upstream in output.parents or output in upstream.parents:
        raise ValueError('Native reference output must not overlap its source')
    if harness.suffix not in ('.c', '.cpp'):
        raise ValueError('Native reference harness must be C or C++')
    files = ['ggml/src/ggml-quants.c', 'ggml/src/ggml-quants.h',
             'ggml/src/ggml-common.h', 'ggml/src/ggml-impl.h', 'ggml/include/ggml.h']
    # Observed major inputs, not an exhaustive dependency closure or acceptance
    # whitelist. Compiler-emitted dependency text is retained separately below.
    observed = {path: hashlib.sha256((upstream / path).read_bytes()).hexdigest() for path in files}
    harness_hash = hashlib.sha256(harness.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    includes = ['-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src')]
    cc, cxx = os.environ.get('CC', 'cc'), os.environ.get('CXX', 'c++')
    with tempfile.TemporaryDirectory(prefix=name + '-', dir=output) as temporary:
        work = Path(temporary)
        obj, executable = work / 'quantization.o', work / name
        commands = [
            [cc, '-std=c11', '-O2', '-ffp-contract=off', '-ffunction-sections', '-fdata-sections',
             '-D_POSIX_C_SOURCE=200809L', *includes, '-MMD', '-MF', str(work / 'quantization.d'),
             '-c', str(upstream / files[0]), '-o', str(obj)],
            [cc if harness.suffix == '.c' else cxx,
             '-std=c11' if harness.suffix == '.c' else '-std=c++20', '-O2',
             '-Wall', '-Wextra', '-Werror', '-ffp-contract=off', *includes,
             '-MMD', '-MF', str(work / 'harness.d'), str(harness), str(obj),
             '-Wl,--gc-sections', '-lm', '-o', str(executable)],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            except subprocess.CalledProcessError as error:
                error.add_note(error.stderr)
                raise
        if (observed != {path: hashlib.sha256((upstream / path).read_bytes()).hexdigest() for path in files}
                or harness_hash != hashlib.sha256(harness.read_bytes()).hexdigest()):
            raise ValueError('Native reference inputs changed during compilation')
        receipt = {'formatVersion': 1, 'scope': 'Complete current C translation unit; native CPU only',
                   'sourceSha256': observed, 'harnessSha256': harness_hash,
                   'compileCommands': commands,
                   'dependencyFiles': {path.name: path.read_text() for path in work.glob('*.d')},
                   'executableSha256': hashlib.sha256(executable.read_bytes()).hexdigest()}
        receipt_path = work / 'receipt.json'
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        executable.replace(output / name)
        receipt_path.replace(output / (name + '.json'))
    return output / name
