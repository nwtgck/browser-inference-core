#!/usr/bin/env python3
"""Optional real Node WebGPU comparison against upstream RMS and SCALE shaders."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
spec = importlib.util.spec_from_file_location('rms_scale_tests', ROOT / 'tests/test_webgpu_rms_scale.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--reference-scale-wg-size', type=int, default=1024,
                        help='Must equal the adapter maxComputeInvocationsPerWorkgroup used by the production SCALE factory')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        parser.error('--node-command-json must be a nonempty string array')
    if not 1 <= args.reference_scale_wg_size <= 1024:
        parser.error('--reference-scale-wg-size must be between 1 and 1024')
    upstream, work = args.source.resolve(strict=True), args.work.resolve()
    for root in (ROOT, upstream):
        if work == root or root in work.parents or work in root.parents:
            parser.error('--work must not overlap source trees')
    module = args.webgpu_module.resolve(strict=True)
    work.mkdir(parents=True, exist_ok=True)
    prepared = tests.prepare(upstream, work / 'prepared')
    shaders = tests.emit_factory(upstream, prepared, work / 'factory', args.reference_scale_wg_size)
    source = {
        'scope': 'Actual prepared RMS/scale factory and checked upstream row_norm epilogue, standalone shader checks only',
        'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': tests.fusion.PATCH_SHA256,
        'preparedHeaderSha256': hashlib.sha256((prepared / 'ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'preparedSourceSha256': hashlib.sha256((prepared / 'ggml-webgpu.cpp').read_bytes()).hexdigest(),
        'upstreamContracts': tests.fusion.CONTRACT_INPUTS,
        'referenceScaleWorkgroupSize': args.reference_scale_wg_size,
        'shaderSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(shaders.glob('*.wgsl'))},
        'harnessSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py', '.mjs', '.cpp')},
    }
    (work / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = module.as_uri()
    return subprocess.run(command + [str(Path(__file__).with_name('check_gpu.mjs')), str(shaders), str(work / 'report.json'),
                                     'benchmark' if args.benchmarks else 'numerical', str(args.reference_scale_wg_size)], env=env).returncode


if __name__ == '__main__': raise SystemExit(main())
