#!/usr/bin/env python3
"""Build two independent native implementations and compare synthetic traces.

This Linux-only regression downloads nothing and does not benchmark a real model
or GPU. Configure both builds even on a warm run; only compiler intermediates may
be reused. Each command is logged and a failed comparison fails the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def identity(path: Path) -> dict:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def run(source: Path, work: Path, jobs: int) -> None:
    source, work = source.resolve(), work.resolve()
    # A common parent containing separate source/build directories is fine;
    # none of the owned output trees may overlap the upstream checkout.
    for name in ('qsa-results', 'native-reference', 'native-fast', 'test-reference', 'test-fast'):
        destination = (work / name).resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError('Regression outputs must be outside the upstream source')
    if jobs < 1:
        raise ValueError('jobs must be positive')
    if not sys.platform.startswith('linux'):
        raise ValueError('The static-library test linker is currently Linux-only')
    results = work / 'qsa-results'
    logs = results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    status = {'scope': 'synthetic native parity; not WebGPU or real-model throughput',
              'commands': [], 'fixtures': {}, 'libraries': {}, 'maskParity': [], 'success': False}

    def save():
        (results / 'status.json').write_text(json.dumps(status, indent=2) + '\n')

    def command(name, argv):
        argv = [str(arg) for arg in argv]
        started = time.monotonic()
        with (logs / (name + '.txt')).open('w') as log:
            process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, OMP_NUM_THREADS='1'))
        status['commands'].append({'name': name, 'command': argv, 'exit': process.returncode,
                                   'elapsedSeconds': time.monotonic() - started})
        save()
        if process.returncode:
            raise RuntimeError(f'{name} failed; see {logs / (name + ".txt")}')

    save()
    try:
        for mode, flag in (('reference', 'OFF'), ('fast', 'ON')):
            build = work / ('native-' + mode)
            test = work / ('test-' + mode)
            command(mode + '-configure', ['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release', '-DLCB_NATIVE_CHECK=ON', '-DLCB_FILE_BACKED_PLE=ON',
                    '-DLCB_QSA_ALL_POOLS=' + flag, '-DLCB_LLAMA_SOURCE=' + str(source)])
            command(mode + '-compile', ['cmake', '--build', build, '--target', 'core', '--parallel', jobs])
            command(mode + '-test-configure', ['cmake', '-S', ROOT / 'tests/qsa-selection',
                    '-B', test, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                    '-DLCB_LLAMA_SOURCE=' + str(source), '-DLCB_NATIVE_BUILD=' + str(build)])
            command(mode + '-test-compile', ['cmake', '--build', test, '--parallel', jobs])
            status['libraries'][mode] = identity(build / 'llama/src/libllama.a')
        if status['libraries']['reference'] == status['libraries']['fast']:
            raise RuntimeError('Reference and optimized libraries unexpectedly have the same identity')
        for quant in ('F16', 'Q3_K', 'Q4_K'):
            fixture = results / (quant + '.gguf')
            command('generate-' + quant, [work / 'test-reference/qsa-model', 'generate', fixture, quant])
            status['fixtures'][quant] = identity(fixture)
            for mask_mode in ('fused', 'unfused'):
                # Keep the original paths for the default native mode.
                trace_root = results / quant if mask_mode == 'fused' else results / quant / 'unfused'
                trace_command = 'trace' if mask_mode == 'fused' else 'trace-unfused'
                for mode in ('reference', 'fast'):
                    command('trace-' + quant + '-' + mask_mode + '-' + mode,
                            [work / ('test-' + mode) / 'qsa-model', trace_command, fixture, trace_root / mode])
                command('compare-' + quant + '-' + mask_mode,
                        [sys.executable, ROOT / 'tests/qsa-selection/compare.py',
                         '--reference', trace_root / 'reference', '--optimized', trace_root / 'fast',
                         '--mask-mode', mask_mode])
            # The fallback is not allowed to silently change numerical outputs.
            for mode in ('reference', 'fast'):
                fused = identity(results / quant / mode / 'logits.f32')
                unfused = identity(results / quant / 'unfused' / mode / 'logits.f32')
                if fused != unfused:
                    raise RuntimeError('Fused/unfused outputs differ: ' + quant + '/' + mode)
                status['maskParity'].append({'quant': quant, 'mode': mode, **fused})
        status['success'] = True
    finally:
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'vendor/llama.cpp')
    parser.add_argument('--work', type=Path, default=ROOT / 'build/qsa-regression')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    run(args.source, args.work, args.jobs)


if __name__ == '__main__':
    main()
