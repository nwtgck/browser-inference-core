#!/usr/bin/env python3
"""Opt-in synthetic router alias WGSL comparison; no browser or model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts')]
import prepare_webgpu_source as source
import prepare_webgpu_router_tail as router
from router_factory import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--webgpu-module', type=Path, required=True)
    parser.add_argument('--node-command-json', default='["node"]')
    parser.add_argument('--backend', choices=['vulkan', 'metal', 'd3d12', 'opengl'])
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        command = json.loads(args.node_command_json)
    except json.JSONDecodeError:
        parser.error('--node-command-json must be a nonempty JSON string array')
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        parser.error('--node-command-json must be a nonempty JSON string array')
    if not args.webgpu_module.is_file():
        parser.error('--webgpu-module must be an existing local WebGPU module')
    upstream, work = args.llama_source.resolve(), args.work.resolve()
    prepared = source.prepare(upstream, work / 'prepared', gpu_profile=args.profile).parent
    executable = build(upstream, prepared, work / 'factory')
    manifest = {'status': 'prepared', 'scope': 'Source-bound synthetic F32, actual baseline/fused factories; GPU execution is opt-in.',
                'source': {'upstreamCommit': '620d2225cc436ab0b001e3200e590e3b286ef9ab',
                           'reviewedInputs': router.CONTRACT_INPUTS, 'patchSha256': router.PATCH_SHA256,
                           'preparedFiles': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared.iterdir() if p.is_file()}},
                'shaderSha256': {}, 'baselineShaderSha256': {}}
    for wg in (256, 512, 1024):
        shaders = work / 'generated' / str(wg)
        shaders.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(executable), str(shaders), str(wg)], check=True, capture_output=True)
        identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shaders.glob('*.wgsl')}
        manifest['shaderSha256'][str(wg)] = identity
        manifest['baselineShaderSha256'][str(wg)] = {n: value for n, value in identity.items() if not n.startswith('router_tail')}
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        return 0
    arguments = [*command, str(Path(__file__).with_name('run_gpu.mjs')), '--work', str(work),
                 '--webgpu-module', str(args.webgpu_module.resolve())]
    if args.backend:
        arguments += ['--backend', args.backend]
    subprocess.run(arguments, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
