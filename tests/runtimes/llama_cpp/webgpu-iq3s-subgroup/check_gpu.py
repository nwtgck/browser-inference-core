#!/usr/bin/env python3
"""Opt-in actual-factory IQ3_S numerical checks with current CPU references.

Only local compiler/Node/WebGPU installations are used. No model weights,
browser execution, download, timing, or target-device speed claim.
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
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT.parents[1]), str(HERE)]
from build_tools.runtimes.llama_cpp import webgpu_plan
from tests.support.llama_cpp.native_quantization import source_commit
import emit_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--subgroup-size', type=int, choices=[16, 32, 64], required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    args = parser.parse_args()
    try:
        node = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if not isinstance(node, list) or not node or any(not isinstance(value, str) or not value for value in node):
        parser.error('--node-command-json must be a nonempty array of strings')
    upstream = args.llama_source.resolve(strict=True)
    work = args.work.resolve()
    for boundary in (upstream, ROOT):
        if work == boundary or boundary in work.parents or work in boundary.parents:
            parser.error('--work must not overlap source trees')
    work.mkdir(parents=True, exist_ok=True)
    emit_factory.prepare(upstream, work / 'prepared')
    subprocess.run([sys.executable, str(HERE / 'make_fixtures.py'), '--llama-source', str(upstream),
                    '--work', str(work / 'cpu')], check=True)
    executable = emit_factory.build_host(upstream, work / 'prepared', work / 'host')
    shaders = work / 'shaders'
    subprocess.run([str(executable), str(shaders), str(work / 'cpu/fixtures/profiles.txt'), str(args.subgroup_size)], check=True)
    preparation = json.loads((work / 'prepared' / webgpu_plan.RECEIPT).read_text())
    identity = {
        'scope': 'Complete current selector/cache/factory and AST-selected dispatcher; synthetic CPU-reference numerical checks only',
        'upstreamCommit': source_commit(upstream),
        'sourcePreparation': preparation,
        'nativeRecorder': json.loads(executable.with_suffix('.json').read_text()),
        'preparedHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'preparedCppSha256': hashlib.sha256((work / 'prepared/ggml-webgpu.cpp').read_bytes()).hexdigest(),
        'embeddedShaderHeaderSha256': hashlib.sha256((work / 'prepared' / 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp').read_bytes()).hexdigest(),
        'shaderSha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(shaders.glob('*.wgsl'))},
        'fixtureManifestSha256': hashlib.sha256((work / 'cpu/fixtures/manifest.json').read_bytes()).hexdigest(),
        'cpuOracle': json.loads((work / 'cpu/fixture-source.json').read_text()),
        'caseDecisions': json.loads((shaders / 'case-decisions.json').read_text()),
        'subgroup': args.subgroup_size,
    }
    (work / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = args.webgpu_module.resolve(strict=True).as_uri()
    return subprocess.run(node + [str(HERE / 'check_gpu.mjs'), str(shaders), str(work / 'cpu/fixtures'),
                                  str(work / 'report.json'), str(args.subgroup_size)], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
