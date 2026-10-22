#!/usr/bin/env python3
"""Run optional source-bound HC GPU checks using an externally supplied Node/Dawn.

This does not build or execute the bicore runtime and is not a speed benchmark.
The normal test suite has no Node/Dawn dependency; invoke this check explicitly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_hc as hc


def extract_shaders(header: str) -> dict[str, str]:
    result = {}
    for name in ('hc_pre', 'hc_post'):
        match = re.search(r'const char\* wgsl_' + name + r' = R"wgsl\((.*?)\)wgsl";', header, re.S)
        if match is None:
            raise ValueError('Reviewed HC embedding changed: ' + name)
        code = match.group(1)
        expected = hc.OUTPUTS['wgsl-shaders/' + name + '.wgsl']
        if hashlib.sha256(code.encode()).hexdigest() != expected:
            raise ValueError('HC shader identity changed: ' + name)
        result[name + '.wgsl'] = code
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--case-start', type=int, default=0)
    parser.add_argument('--case-count', type=int, default=288)
    args = parser.parse_args()
    if args.case_start < 0 or args.case_count <= 0 or args.case_start + args.case_count > 288:
        parser.error('case range must be inside [0, 288)')
    command = json.loads(args.node_command_json)
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        parser.error('--node-command-json must be a nonempty string array')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hc-gpu-check-') as tmp:
        assets = Path(tmp)
        source.prepare(args.source, assets)
        for name, code in extract_shaders((assets / hc.HEADER_NAME).read_text()).items():
            (assets / name).write_text(code)
        subprocess.run([*command, str(Path(__file__).with_name('check-gpu.mjs')), str(assets), str(output), str(args.case_start), str(args.case_count)],
                       check=True, timeout=180)
    result = json.loads(output.read_text())
    if result.get('ok') is not True or len(result.get('cases', [])) != args.case_count:
        raise RuntimeError('Incomplete or failed HC GPU regression')


if __name__ == '__main__':
    main()
