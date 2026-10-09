#!/usr/bin/env python3
"""Source-bound parameter upload checks. No model or GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_hash(path, expected):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f'Input identity mismatch: {path}')


def section(text, start, end):
    return text[text.index(start):text.index(end)]


def default_source(text):
    lines = []
    inside = False
    keep = True
    for line in text.splitlines():
        if line == '#ifdef GGML_WEBGPU_BATCH_PARAM_UPLOADS':
            require(not inside, 'Unexpected nested upload guard')
            inside, keep = True, False
        elif inside and line == '#else':
            keep = True
        elif inside and line == '#endif':
            inside, keep = False, True
        elif keep and line.strip():
            lines.append(line)
    require(not inside, 'Unclosed upload guard')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--dawn-package', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    identity = json.loads((here / 'inputs.json').read_text())
    upstream = args.upstream.resolve()
    verify_hash(upstream / identity['source'], identity['sha256'])
    original = (upstream / identity['source']).read_text()
    with tempfile.TemporaryDirectory(prefix='webgpu-param-upload-') as work:
        root = Path(work)
        source = root / identity['source']
        source.parent.mkdir(parents=True)
        source.write_text(original)
        patch = here.parents[1] / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/ggml-webgpu-batch-param-uploads.patch'
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
        actual = source.read_text()
        require(default_source(actual) == [line for line in original.splitlines() if line.strip()],
                'Disabled source must preserve original behavior')
        graph = section(actual, 'static ggml_status ggml_backend_webgpu_graph_compute(',
                        '\nstruct ggml_backend_webgpu_event_context')
        original_graph = section(original, 'static ggml_status ggml_backend_webgpu_graph_compute(',
                                 '\nstruct ggml_backend_webgpu_event_context')
        require(graph.count('ctx->param_arena.flush(ctx->global_ctx->queue);') == 2,
                'Both graph submit paths must flush')
        require(default_source(graph) == [line for line in original_graph.splitlines() if line.strip()],
                'Graph cadence or synchronization changed')
        arena = section(actual, 'struct webgpu_param_arena {', '\nstruct webgpu_encoded_op')
        print('PASS: source identity, patch application, disabled equivalence, graph boundaries')
        if args.prepare_only:
            return
        prefix = (here / 'arena-prefix.cpp').read_text()
        graph_prefix = prefix.replace('struct Device {};',
                                     'struct CommandEncoder; struct Device { CommandEncoder CreateCommandEncoder(); };')
        suites = {
            'arena': prefix + arena + (here / 'arena-cases.cpp').read_text(),
            'graph': graph_prefix + arena + (here / 'graph-stubs.cpp').read_text() + graph
                     + (here / 'graph-cases.cpp').read_text(),
        }
        for suite, code in suites.items():
            path = root / f'{suite}.cpp'
            path.write_text(code)
            for enabled in (False, True):
                mode = 'batch' if enabled else 'default'
                binary = root / f'{suite}-{mode}'
                command = [args.compiler, '-std=c++17', '-g', '-O1', str(path), '-o', str(binary)]
                if enabled:
                    command.append('-DGGML_WEBGPU_BATCH_PARAM_UPLOADS')
                if args.sanitize:
                    command.append('-fsanitize=address,undefined')
                subprocess.run(command, check=True)
                subprocess.run([str(binary)], check=True)
                if suite == 'arena':
                    for case in range(1, 7 if enabled else 3):
                        result = subprocess.run([str(binary), str(case)], capture_output=True, text=True)
                        require(result.returncode == -6 and (case == 2 or 'Assertion' in result.stderr),
                                f'Expected bounds rejection for {mode}, case {case}: {result.stderr}')
                print(f'PASS: {suite} {mode}')
        if args.dawn_package:
            dawn = args.dawn_package.resolve()
            for name, digest in identity['dawn_header_sha256'].items():
                verify_hash(dawn / name, digest)
            for name, digest in identity['ggml_header_sha256'].items():
                verify_hash(upstream / name, digest)
            real = root / 'real-arena.cpp'
            real.write_text('''#include "ggml-impl.h"
#include <webgpu/webgpu_cpp.h>
#include <vector>
#include <cstring>
#define ROUNDUP_POW2(x,p) (((x)+((p)-1))&~((p)-1))
static void ggml_webgpu_create_buffer(wgpu::Device &, wgpu::Buffer &, size_t, wgpu::BufferUsage, const char *);
''' + arena)
            for enabled in (False, True):
                command = [args.compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                           '-Wno-unused-function', '-fsyntax-only', '-DGGML_SCHED_MAX_COPIES=4',
                           '-D__EMSCRIPTEN__', '-D__wasm64__',
                           '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src'),
                           '-isystem', str(dawn / 'webgpu/include'),
                           '-isystem', str(dawn / 'webgpu_cpp/include'), str(real)]
                if enabled:
                    command.append('-DGGML_WEBGPU_BATCH_PARAM_UPLOADS')
                subprocess.run(command, check=True)
            print('PASS: actual arena with pinned ggml/Dawn headers in both modes (host syntax only)')


if __name__ == '__main__':
    main()
