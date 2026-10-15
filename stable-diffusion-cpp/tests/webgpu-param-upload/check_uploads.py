#!/usr/bin/env python3
"""Compile the prepared arena, encoder and graph loop against an ordered queue double.

This exercises the live parameter consumers and both submit boundaries, not a
second implementation of batching. It does not execute GPU shaders or Wasm.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import tempfile


def section(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('wait_probe', here.parent / 'webgpu-wait-probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    source = args.source.read_text()
    arena = section(source, 'struct webgpu_param_arena {', '\nstruct webgpu_encoded_op')
    build = probe.function(source, 'static webgpu_encoded_op ggml_backend_webgpu_build_multi(')
    submit = probe.function(source, 'static void ggml_backend_webgpu_submit_commands(')
    graph = probe.function(source, 'static ggml_status ggml_backend_webgpu_graph_compute(')
    if build.count('ctx->param_arena.stage(') != 1 or graph.count('ctx->param_arena.flush(') != 2:
        raise ValueError('Review changed parameter producer or command submission boundaries')
    if source.count('param_arena.alloc_slot(') != 1:
        raise ValueError('Unreviewed parameter producer')
    code = (here / 'queue-prefix.cpp').read_text() + arena
    code += (here / 'queue-graph.cpp').read_text() + submit + build
    code += (here / 'queue-encoder.cpp').read_text() + graph
    code += (here / 'queue-cases.cpp').read_text()
    with tempfile.TemporaryDirectory(prefix='sdc-param-uploads-') as directory:
        root = Path(directory)
        path = root / 'queue.cpp'
        path.write_text(code)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(code)
        for batched in (False, True):
            name = 'batched' if batched else 'per-kernel'
            binary = root / name
            command = [args.cxx, '-std=c++17', '-O1', '-g', '-Wall', '-Wextra',
                       '-Werror', '-Wno-unused-parameter', str(path), '-o', str(binary)]
            if batched:
                command.append('-DGGML_WEBGPU_BATCH_PARAM_UPLOADS=1')
            if args.sanitize:
                command.append('-fsanitize=address,undefined')
            subprocess.run(command, check=True, timeout=90)
            subprocess.run([str(binary)], check=True, timeout=30)
            for case in range(1, 7 if batched else 3):
                result = subprocess.run([str(binary), str(case)], capture_output=True, text=True, timeout=10)
                if result.returncode != -6 or (case != 2 and 'Assertion' not in result.stderr):
                    raise ValueError(f'Bounds check did not reject {name} case {case}: {result.stderr}')
            print(f'PASS: {name} actual arena/build_multi/submit/graph loop and bounds failures', flush=True)


if __name__ == '__main__':
    main()
