#!/usr/bin/env python3
"""Opt-in numerical check of actual prepared Q8 matvec specializations.

Requires local upstream, a C++20 compiler, pinned Dawn headers, Node >=22,
and the Dawn `webgpu` package. Set LCB_TEST_DAWN_PACKAGE to the header package.
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
    work.mkdir(parents=True, exist_ok=True)
    emit_factory.prepare(upstream, work / 'prepared')
    executable = emit_factory.build_host(upstream, work / 'prepared', work / 'host')
    shaders = work / 'shaders'
    subprocess.run([str(executable), str(shaders)], check=True)
    preparation = json.loads((work / 'prepared/source-preparation.json').read_text())
    identity = {
        'upstreamCommit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
        'patchSha256': next(item['sha256'] for item in preparation['patches']
                            if item['file'] == 'ggml-webgpu-q8-short-workgroup.patch'),
        'preparedHeaderSha256': hashlib.sha256((work / 'prepared/ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(),
        'preparation': preparation,
        'shaderSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(shaders.glob('*.wgsl'))},
        'scope': 'Complete current factory and real Dawn bindings with a mocked C API for emission; standalone numerical validation only',
    }
    (work / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
    env = dict(os.environ)
    env['LCB_TEST_WEBGPU_MODULE'] = args.webgpu_module.resolve(strict=True).as_uri()
    env['LCB_Q8_BENCHMARKS'] = '1' if args.benchmarks else '0'
    env['LCB_Q8_FORCE_GC'] = '1' if args.force_gc else '0'
    if args.force_gc:
        command.append('--expose-gc')
    result = subprocess.run(command + [str(Path(__file__).with_name('check_gpu.mjs')),
                                      str(shaders), str(work / 'report.json')], env=env)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
