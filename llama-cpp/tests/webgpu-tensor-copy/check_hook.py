#!/usr/bin/env python3
"""Apply the source-bound patch to a temporary copy and execute its actual C++ hook.
The runtime test uses a CPU mock; optional pinned-header checks validate C++ API syntax only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import prepare_webgpu_tensor_copy as overlay

HERE = Path(__file__).resolve().parent

def verify_hash(data, expected, label):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(label + ' identity mismatch')


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
    parser.add_argument('--dawn-package', type=Path, help='Optional exact unpacked Dawn package for a real-header hook syntax check')
    args = parser.parse_args()
    identity = json.loads((HERE / 'inputs.json').read_text())
    original = (args.upstream / identity['source']).read_bytes()
    verify_hash(original, identity['sha256'], 'upstream source')
    with tempfile.TemporaryDirectory(prefix='webgpu-copy-') as directory:
        root = Path(directory)
        target = overlay.prepare(args.upstream, root / 'overlay',
                                 overlay.ROOT / overlay.PATCH_DIRECTORY / overlay.PATCH_NAME)
        actual = target.read_text()
        if '/* .cpy_tensor      = */ ggml_backend_webgpu_buffer_cpy_tensor,' not in actual:
            raise ValueError('copy hook is not registered')
        hook = '\n'.join(function(actual, name) for name in (
            'ggml_webgpu_tensor_offset', 'ggml_backend_webgpu_buffer_get_base',
            'ggml_backend_webgpu_buffer_cpy_tensor'))
        backend_bytes = (args.upstream / identity['test_backend_source']).read_bytes()
        verify_hash(backend_bytes, identity['test_backend_sha256'], 'generic backend source')
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
        if args.dawn_package:
            dawn = args.dawn_package.resolve()
            for name, digest in identity['dawn_header_sha256'].items():
                verify_hash((dawn / name).read_bytes(), digest, 'Dawn header ' + name)
            for name, digest in identity['ggml_header_sha256'].items():
                verify_hash((args.upstream / name).read_bytes(), digest, 'ggml header ' + name)
            # Real ggml and WebGPU types; only the owning context is reduced to the fields used by the hook.
            context_start = actual.index('struct ggml_backend_webgpu_buffer_context {')
            context_end = actual.index('\n};', context_start) + 3
            context = actual[context_start:context_end]
            real_header_code = """#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <webgpu/webgpu_cpp.h>
#include <memory>
#include <string>
#include <type_traits>
struct webgpu_global_context_struct { wgpu::Device device; wgpu::Queue queue; };
using webgpu_global_context = std::shared_ptr<webgpu_global_context_struct>;
static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;
""" + context + '\n' + hook + """
static_assert(std::is_same_v<decltype(&ggml_backend_webgpu_buffer_cpy_tensor), decltype(ggml_backend_buffer_i::cpy_tensor)>);
"""
            real_header_source = root / 'real-header-hook.cpp'
            real_header_source.write_text(real_header_code)
            subprocess.run([args.compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-fsyntax-only',
                            '-DNDEBUG', '-D__EMSCRIPTEN__', '-D__wasm64__', '-DGGML_SCHED_MAX_COPIES=4',
                            '-I' + str(args.upstream.resolve() / 'ggml/include'),
                            '-I' + str(args.upstream.resolve() / 'ggml/src'),
                            '-isystem', str(dawn / 'webgpu/include'),
                            '-isystem', str(dawn / 'webgpu_cpp/include'), str(real_header_source)], check=True)
            print('PASS: actual hook with pinned ggml and Dawn headers (host syntax only; no Wasm or GPU execution)')

if __name__ == '__main__':
    main()
