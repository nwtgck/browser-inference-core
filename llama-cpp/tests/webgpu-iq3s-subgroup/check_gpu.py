#!/usr/bin/env python3
"""Opt-in actual-factory IQ3_S numerical checks with pinned CPU references.

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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT / 'scripts'), str(HERE)]
import prepare_webgpu_iq3s_subgroup as iq3
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
    identity = {
        'scope': 'Actual prepared selector/cache/factory and full dispatcher, upstream emitter/preprocessor; synthetic CPU-reference numerical checks only',
        'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': iq3.PATCH_SHA256,
        'preparedHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'preparedCppSha256': hashlib.sha256((work / 'prepared/ggml-webgpu.cpp').read_bytes()).hexdigest(),
        'embeddedShaderHeaderSha256': iq3.HEADER_SHA256,
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
