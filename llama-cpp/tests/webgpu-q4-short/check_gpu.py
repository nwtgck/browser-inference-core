#!/usr/bin/env python3
"""Opt-in actual-factory Q4_K GPU check, with upstream CPU dequantization.

Uses an existing Node WebGPU package only; no browser, download or model run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_q4_short as q4
import prepare_webgpu_tensor_copy as copy
from test_webgpu_q4_short_overlay import build_host, function, prepare

REFERENCE_INPUTS = {
    'ggml/src/ggml-quants.c': '5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab',
    'ggml/src/ggml-quants.h': '28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da',
    'ggml/src/ggml-common.h': '0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c',
    'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
}


def build_reference(upstream, output):
    for path, expected in REFERENCE_INPUTS.items():
        if copy.digest(upstream / path) != expected:
            raise ValueError('Q4 CPU reference contract changed: ' + path)
    text = (upstream / 'ggml/src/ggml-quants.c').read_text()
    code = output / 'reference.cpp'
    code.write_text('#include "ggml-quants.h"\n#include "ggml-impl.h"\n' +
                    function(text, 'static inline void get_scale_min_k4(') + '\n' +
                    function(text, 'void dequantize_row_q4_K(') + '\n' +
                    Path(__file__).with_name('reference-main.cpp').read_text())
    exe = output / 'reference'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'), '-isystem', str(upstream / 'ggml/src'),
                    str(code), '-o', str(exe)], check=True)
    return exe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--wall-benchmarks', action='store_true')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        parser.error('--node-command-json must be a nonempty array of strings')
    if args.wall_benchmarks and not args.benchmarks:
        parser.error('--wall-benchmarks requires --benchmarks')
    upstream, work = args.llama_source.resolve(strict=True), args.work.resolve()
    for root in (upstream, ROOT):
        if work == root or root in work.parents or work in root.parents:
            parser.error('--work must not overlap source trees')
    module = args.webgpu_module.resolve(strict=True)
    work.mkdir(parents=True, exist_ok=True)
    prepare(upstream, work / 'prepared')
    host = build_host(upstream, work / 'prepared', work / 'host')
    shaders = work / 'shaders'
    subprocess.run([str(host), str(shaders)], check=True)
    reference = build_reference(upstream, work / 'host')
    identity = {
        'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': q4.PATCH_SHA256,
        'preparedHeaderSha256': copy.digest(work / 'prepared/ggml-webgpu-shader-lib.hpp'),
        'upstreamContracts': q4.CONTRACT_INPUTS,
        'referenceContracts': REFERENCE_INPUTS,
        'shaderSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(shaders.glob('*.wgsl'))},
        'harnessSha256': {p.name: copy.digest(p) for p in sorted(Path(__file__).parent.iterdir())
                          if p.suffix in ('.py', '.mjs', '.cpp')},
        'overlayTestSha256': copy.digest(ROOT / 'tests/test_webgpu_q4_short_overlay.py'),
        'scope': 'Actual prepared factories and upstream emitter; standalone numerical validation only',
    }
    (work / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = module.as_uri()
    env['LCB_Q4_BENCHMARKS'] = '1' if args.benchmarks else '0'
    env['LCB_Q4_WALL_BENCHMARKS'] = '1' if args.wall_benchmarks else '0'
    fixtures = work / 'fixtures'
    subprocess.run(command + [str(Path(__file__).with_name('fixtures.mjs')), str(shaders), str(fixtures)], check=True)
    cases = json.loads((shaders / 'cases.json').read_text())
    for index, _ in enumerate(cases):
        payload = fixtures / (str(index) + '.bin')
        with payload.open('rb') as fixture_input, payload.with_suffix('.f64').open('wb') as reference_output:
            subprocess.run([str(reference)], stdin=fixture_input, stdout=reference_output, check=True)
    env['LCB_Q4_CPU_REFERENCE_DIR'] = str(fixtures)
    return subprocess.run(command + [str(Path(__file__).with_name('check_gpu.mjs')), str(shaders), str(work / 'report.json')], env=env).returncode


if __name__ == '__main__': raise SystemExit(main())
