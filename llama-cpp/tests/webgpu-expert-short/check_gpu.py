#!/usr/bin/env python3
"""Opt-in numerical check of actual prepared expert matvec workgroups.

Requires a local pinned upstream, g++, Node >=22 and the Dawn `webgpu` package.
This is not a browser/model benchmark. Nothing is installed or downloaded.
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
import prepare_webgpu_expert_short as expert
from test_webgpu_expert_short_overlay import build_host


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--benchmarks', action='store_true')
    parser.add_argument('--force-gc', action='store_true')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must contain valid JSON')
    if (not isinstance(command, list) or not command or
            any(not isinstance(x, str) or not x for x in command)):
        parser.error('--node-command-json must be a nonempty array of strings')
    upstream = args.llama_source.resolve(strict=True)
    work = args.work.resolve()
    for root in (upstream, ROOT):
        if work == root or root in work.parents or work in root.parents:
            parser.error('--work must not overlap source trees')
    work.mkdir(parents=True, exist_ok=True)
    source.prepare(upstream, work / 'prepared')
    executable = build_host(upstream, work / 'prepared', work / 'host')
    shaders = work / 'shaders'
    subprocess.run([str(executable), str(shaders)], check=True)
    identity = {
        'upstreamCommit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': expert.PATCH_SHA256,
        'preparedHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'compositionInputs': expert.COMPOSITION_INPUTS,
        'upstreamContracts': expert.CONTRACT_INPUTS,
        'shaderSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(shaders.glob('*.wgsl'))},
        'scope': 'Actual prepared factory and upstream emitter; standalone numerical validation only',
    }
    (work / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = args.webgpu_module.resolve(strict=True).as_uri()
    env['LCB_EXPERT_BENCHMARKS'] = '1' if args.benchmarks else '0'
    env['LCB_EXPERT_FORCE_GC'] = '1' if args.force_gc else '0'
    if args.force_gc:
        command.append('--expose-gc')
    result = subprocess.run(command + [str(Path(__file__).with_name('check_gpu.mjs')),
                                      str(shaders), str(work / 'report.json')], env=env)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
