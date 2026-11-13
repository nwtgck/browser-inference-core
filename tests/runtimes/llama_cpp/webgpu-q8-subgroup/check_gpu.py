#!/usr/bin/env python3
"""Opt-in actual-factory Q8 subgroup numerical check with full Float64 references.

Requires native C++20, Clang for AST ranges, pinned Dawn headers, Node >=22
and node-webgpu (LCB_TEST_DAWN_PACKAGE selects the headers). Nothing is downloaded. On Mesa
software Vulkan, LP_NATIVE_VECTOR_WIDTH=128/256/512 can expose fixed SG4/8/16;
the checker verifies actual reported limits before executing that specialization.
"""
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
sys.path[:0] = [str(ROOT.parents[1]), str(Path(__file__).parent)]
import emit_factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--subgroup-size', type=int, choices=[4, 8, 16, 32, 64], required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--force-gc', action='store_true')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if not isinstance(command, list) or not command or any(not isinstance(value, str) or not value for value in command):
        parser.error('--node-command-json must be a nonempty array of strings')
    upstream = args.llama_source.resolve(strict=True)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    emit_factory.prepare(upstream, work / 'prepared')
    executable = emit_factory.build_host(upstream, work / 'prepared', work / 'host')
    shaders = work / 'shaders'
    subprocess.run([str(executable), str(shaders)], check=True)
    preparation = json.loads((work / 'prepared/source-preparation.json').read_text())
    identity = {
        'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'preparation': preparation,
        'patchSha256': next(item['sha256'] for item in preparation['patches']
                            if item['file'] == 'ggml-webgpu-q8-subgroup-matvec.patch'),
        'preparedHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'embeddedShaderHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-q8-subgroup-shaders.hpp').read_bytes()).hexdigest(),
        'shaderSha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(shaders.glob('*.wgsl'))},
        'scope': 'Complete current factory, real Dawn bindings with a mocked C API, compiler-AST-selected dispatch; standalone numerical checks only',
        'fixtures': 'Deterministic synthetic Q8_0 and Float32 data, full Float64 reference for every output',
    }
    (work / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
    cases = [
        {'name': 'lower-k-bound', 'k': 2048, 'm': 1024, 'mode': 0},
        {'name': 'tail-offset-strides', 'k': 2560, 'm': 1025, 'mode': 1},
        {'name': 'mixed-k4096', 'k': 4096, 'm': 1024, 'mode': 2},
        {'name': 'mixed-k6144-tail', 'k': 6144, 'm': 1025, 'mode': 2},
        {'name': 'upper-k-bound', 'k': 8192, 'm': 1024, 'mode': 2},
        {'name': 'large-output', 'k': 2560, 'm': 10240, 'mode': 2},
        {'name': 'wide-projection', 'k': 2560, 'm': 6144, 'mode': 2},
        {'name': 'long-projection', 'k': 6144, 'm': 2560, 'mode': 2},
        {'name': 'batch-broadcast-tail', 'k': 2048, 'm': 1025, 'mode': 2, 'batched': True},
    ]
    manifest = {
        'baseline': 'upstream-reduction',
        'variants': [
            {'name': 'upstream-reduction', 'file': 'q8-baseline.wgsl', 'wgSize': 256, 'outputsPerWg': 4},
            {'name': 'independent-subgroups', 'file': f'q8-independent-sg{args.subgroup_size}.wgsl',
             'wgSize': 128, 'outputsPerWg': 128 // args.subgroup_size * 4,
             'experiment': {'fixedSubgroupSizeRequired': args.subgroup_size}},
        ],
        'cases': cases,
        'benchmarks': {'caseNames': ['large-output', 'wide-projection', 'long-projection'],
                       'warmupPairs': 4, 'measuredPairs': 12, 'dispatchesPerPass': 1} if args.benchmarks else False,
    }
    manifest_path = work / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = args.webgpu_module.resolve(strict=True).as_uri()
    env['LCB_Q8_FORCE_GC'] = '1' if args.force_gc else '0'
    if args.force_gc:
        command.append('--expose-gc')
    result = subprocess.run(command + [str(Path(__file__).with_name('check_gpu.mjs')), str(shaders),
                                      str(work / 'report.json'), str(manifest_path)], env=env)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
