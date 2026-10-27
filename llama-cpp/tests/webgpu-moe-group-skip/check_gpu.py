#!/usr/bin/env python3
"""Opt-in actual canonical/prepared MoE matrix shader check; no browser/model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts')]
import prepare_webgpu_source as source
import prepare_webgpu_moe_group_skip as group
from emit_factory import build
from make_fixtures import generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--backend', choices=['vulkan', 'metal', 'd3d12', 'opengl'])
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--filter', default='')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        parser.error('--node-command-json must be a nonempty array of strings')
    upstream, work = args.llama_source.resolve(strict=True), args.work.resolve()
    for boundary in (upstream, ROOT):
        if work == boundary or boundary in work.parents or work in boundary.parents:
            parser.error('--work must not overlap source trees')
    work.mkdir(parents=True, exist_ok=True)
    prepared = source.prepare(upstream, work / 'prepared').parent
    executable = build(upstream, prepared, work / 'host')
    manifest = generate(upstream, work, executable, large=args.large)
    identity = {
        'scope': 'Actual canonical/prepared matrix factories and upstream gather/emitter; self-authored inputs only',
        'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': group.PATCH_SHA256,
        'preparedHeaderSha256': hashlib.sha256((prepared / 'ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'matrixEmbeddingSha256': group.HEADER_SHA256,
        'fixtureManifestSha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'reviewedInputs': group.CONTRACT_INPUTS,
        'variantNames': ['baseline', 'candidate'],
        'gatherWorkgroupSize': 256,
        'gridPlannerTestLimit': 17,
    }
    (work / ('source-large.json' if args.large else 'source.json')).write_text(json.dumps(identity, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = args.webgpu_module.resolve(strict=True).as_uri()
    if args.backend:
        env['LCB_TEST_WEBGPU_BACKEND'] = args.backend
    arguments = [str(Path(__file__).with_name('check_gpu.mjs')), str(work), '--filter', args.filter]
    if args.large:
        arguments.append('--large')
    if args.benchmarks:
        arguments.append('--benchmark')
    return subprocess.run(command + arguments, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
