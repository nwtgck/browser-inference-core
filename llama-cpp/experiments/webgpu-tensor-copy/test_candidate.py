#!/usr/bin/env python3
"""Apply the source-bound patch to a temporary copy and execute its actual C++ hook.
The API is a CPU mock; this is not a Dawn, Wasm, browser or GPU runtime test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent

def function(source, name):
    pos = source.index(name + '(')
    start = source.rfind('\n', 0, pos) + 1
    brace = source.index('{', pos)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    identity = json.loads((HERE / 'inputs.json').read_text())
    original = (args.upstream / identity['source']).read_bytes()
    assert hashlib.sha256(original).hexdigest() == identity['sha256'], 'upstream source identity mismatch'
    with tempfile.TemporaryDirectory(prefix='webgpu-copy-') as directory:
        root = Path(directory)
        target = root / identity['source']
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        patch = HERE / 'ggml-webgpu-same-device-tensor-copy.patch'
        subprocess.run(['git','apply','--check',str(patch)], cwd=root, check=True)
        subprocess.run(['git','apply',str(patch)], cwd=root, check=True)
        actual = target.read_text()
        assert '/* .cpy_tensor      = */ ggml_backend_webgpu_buffer_cpy_tensor,' in actual
        hook = '\n'.join(function(actual, name) for name in (
            'ggml_webgpu_tensor_offset', 'ggml_backend_webgpu_buffer_get_base',
            'ggml_backend_webgpu_buffer_cpy_tensor'))
        backend_bytes = (args.upstream / identity['test_backend_source']).read_bytes()
        assert hashlib.sha256(backend_bytes).hexdigest() == identity['test_backend_sha256'], 'generic backend source identity mismatch'
        backend = backend_bytes.decode()
        generic = '\n'.join(function(backend, name) for name in (
            'ggml_backend_buffer_copy_tensor', 'ggml_backend_tensor_copy'))
        code = (HERE / 'mock-harness.cpp').read_text().replace('// @EXTRACTED_HOOK@',hook).replace('// @EXTRACTED_GENERIC_COPY@',generic)
        harness = root / 'test.cpp'
        harness.write_text(code)
        if args.prepare_only:
            print('PASS: exact source identity, patch application and actual function extraction (no compilation)')
            return
        binary = root / 'test'
        subprocess.run([args.compiler,'-std=c++20','-Wall','-Wextra','-Werror','-O0',str(harness),'-o',str(binary)],check=True)
        subprocess.run([str(binary)], check=True)

if __name__ == '__main__':
    main()
