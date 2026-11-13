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

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path[:0] = [str(ROOT.parents[1]), str(runtime_test_root('llama-cpp'))]
from build_tools.runtimes.llama_cpp import source as source
from build_tools.runtimes.llama_cpp import webgpu_plan
from tests.support.llama_cpp.webgpu_stages import tensor_copy as copy
from tests.runtimes.llama_cpp.test_webgpu_q4_short_overlay import build_host, prepare

from tests.support.llama_cpp.native_quantization import build_reference as compile_reference, source_commit


def build_reference(upstream, output):
    return compile_reference(upstream, output, Path(__file__).with_name('reference-main.cpp'), name='reference')


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
        'upstreamCommit': source_commit(upstream),
        'sourcePreparation': json.loads((work / 'prepared' / webgpu_plan.RECEIPT).read_text()),
        'nativeRecorder': json.loads(host.with_suffix('.json').read_text()),
        'preparedHeaderSha256': copy.digest(work / 'prepared/ggml-webgpu-shader-lib.hpp'),
        'referenceCompilation': json.loads(reference.with_name(reference.name + '.json').read_text()),
        'shaderSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(shaders.glob('*.wgsl'))},
        'harnessSha256': {p.name: copy.digest(p) for p in sorted(Path(__file__).parent.iterdir())
                          if p.suffix in ('.py', '.mjs', '.cpp')},
        'overlayTestSha256': copy.digest(runtime_test_root('llama-cpp') / 'test_webgpu_q4_short_overlay.py'),
        'scope': 'Complete current factories and AST-selected dispatcher; standalone numerical validation only',
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
