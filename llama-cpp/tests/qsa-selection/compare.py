#!/usr/bin/env python3
"""Compare independent builds: finite, byte-identical outputs AND graph coverage.

The all-pools path changes only selection and masks, not attention arithmetic.
This oracle deliberately rejects two copies of the same baseline executable.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

FIELDS = ('phase', 'indexer', 'top_k', 'query', 'raw_key', 'selection_mask', 'pools64', 'pools128')
SHORT = ('short-prefill', 'short-reused', 'prefill', 'before-threshold', 'shared-prefix',
         'sequence-zero-divergence', 'sequence-one-divergence', 'after-sequence-removal',
         'mixed', 'embedding-chunks', 'unified-prefix', 'parallel-sequences')
CROSS = ('cross-threshold', 'restore-cross-threshold')
PHASES = set(SHORT + CROSS + ('reserve', 'parallel-reserve', 'ranked-prefill'))


def read_nodes(path: Path):
    totals = defaultdict(Counter)
    phases = []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise AssertionError('Missing or unknown graph-observation columns')
        for row in reader:
            phase = row.pop('phase')
            if phase not in PHASES or set(row) != set(FIELDS[1:]):
                raise AssertionError('Missing or unknown observation phase/columns')
            values = {key: int(value) for key, value in row.items()}
            if any(value < 0 for value in values.values()):
                raise AssertionError('Negative observation count')
            phases.append(phase)
            totals[phase].update(values)
    if set(phases) != PHASES:
        raise AssertionError('Incomplete phase coverage')
    return dict(totals), phases


def compare(reference: Path, optimized: Path):
    original = (reference / 'logits.f32').read_bytes()
    changed = (optimized / 'logits.f32').read_bytes()
    if len(original) == 0 or len(original) % 4 or len(original) != len(changed):
        raise AssertionError('Missing, malformed or unequal-length output trace')
    for data in (original, changed):
        if any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', data)):
            raise AssertionError('Nonfinite model output')
    # This fixed synthetic fixture has nonconstant output. A broken generator
    # or an empty-compute trace returning a constant tensor is not coverage.
    first = struct.unpack_from('<f', original)[0]
    if all(value == first for (value,) in struct.iter_unpack('<f', original)):
        raise AssertionError('Degenerate constant model output')
    if original != changed:
        maximum = max(abs(a[0] - b[0]) for a, b in zip(struct.iter_unpack('<f', original), struct.iter_unpack('<f', changed)))
        raise AssertionError('Full-model outputs differ; maximum absolute error: ' + str(maximum))
    before, before_phases = read_nodes(reference / 'nodes.csv')
    after, after_phases = read_nodes(optimized / 'nodes.csv')
    if before_phases != after_phases:
        raise AssertionError('Observation schedules differ')
    for phase in SHORT:
        for operation in ('indexer', 'top_k', 'query'):
            if before[phase][operation] <= 0 or after[phase][operation] != 0:
                raise AssertionError('All-pools bypass not exercised: ' + phase + '/' + operation)
    for phase in SHORT:
        if not (0 < after[phase]['selection_mask'] < before[phase]['selection_mask']):
            raise AssertionError('Graph-local mask sharing not exercised: ' + phase)
    for phase in PHASES - {'reserve', 'parallel-reserve'}:
        if before[phase]['raw_key'] <= 0 or after[phase]['raw_key'] != before[phase]['raw_key']:
            raise AssertionError('Per-layer key-cache write coverage differs: ' + phase)
    for phase in CROSS:
        for operation in ('indexer', 'top_k', 'query'):
            if not (0 < after[phase][operation] < before[phase][operation]):
                raise AssertionError('Both threshold branches were not exercised: ' + phase + '/' + operation)
    if after[CROSS[0]] != after[CROSS[1]]:
        raise AssertionError('Restoration did not reproduce graph branch observations')
    if after['ranked-prefill'] != before['ranked-prefill'] or after['ranked-prefill']['indexer'] <= 0:
        raise AssertionError('Ranked path was not preserved')
    return {'identical': True, 'bytes': len(original), 'floatValues': len(original) // 4,
            'sha256': hashlib.sha256(original).hexdigest(),
            'referenceNodes': before, 'optimizedNodes': after,
            'scope': 'synthetic native full-model parity, not a real-model or WebGPU throughput result'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--optimized', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.optimized), indent=2))


if __name__ == '__main__':
    main()
