"""Generate deterministic IQ3_S test fixtures with the pinned CPU dequantizer."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--llama-source', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
upstream = args.llama_source.resolve(strict=True)
lab = args.work.resolve()
for boundary in (upstream, here.parents[1]):
    if lab == boundary or boundary in lab.parents or lab in boundary.parents:
        parser.error('--work must not overlap source trees')
lab.mkdir(parents=True, exist_ok=True)
fixtures = lab / 'fixtures'
fixtures.mkdir(exist_ok=True)
quants_path = upstream / 'ggml/src/ggml-quants.c'
quants = quants_path.read_text()
start = quants.index('void dequantize_row_iq3_s(')
function = quants[start:quants.index('\n}', start) + 2]
oracle_source = lab / 'oracle-generated.c'
oracle_source.write_text('#define GGML_COMMON_DECL_C\n#define GGML_COMMON_IMPL_C\n#include "ggml-common.h"\n#include "ggml-impl.h"\n'
                        + function + '\n' + (here / 'oracle-main.c').read_text())
oracle = lab / 'oracle'
command = ['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
           '-isystem', str(upstream / 'ggml/include'), '-isystem', str(upstream / 'ggml/src'),
           str(oracle_source), '-lm', '-o', str(oracle)]
subprocess.run(command, check=True)
subprocess.run([str(oracle), '--self-test'], check=True)

base_cases = [
    ('tiny-k256-m7-shared-padded', 256, 7, True, True, True),
    ('tiny-k512-m63-separate-padded', 512, 63, False, True, True),
    ('medium-k2048-m256-shared-contiguous', 2048, 256, True, False, False),
    ('hot-k2560-m640-separate-contiguous', 2560, 640, False, False, False),
    ('hot-k2560-m641-shared-padded', 2560, 641, True, True, True),
    ('hot-k2560-m640-shared-contiguous', 2560, 640, True, False, False),
    ('wide-k4096-m63-separate-padded', 4096, 63, False, True, True),
    ('wide-k4096-m641-separate-padded', 4096, 641, False, True, True),
]
cases = [{'seed': seed, 'shape': shape, 'skippedSlots': [], 'baselineCase': None}
         for seed, shape in enumerate(base_cases)]
for base_index, suffix, skipped_slots in [
    (5, 'mixed-skipped', [1, 4, 9]),
    (3, 'mixed-skipped', [0, 3, 8]),
    (4, 'mixed-skipped', [2, 5, 7]),
    (4, 'all-skipped', list(range(10))),
]:
    shape = base_cases[base_index]
    cases.append({'seed': base_index, 'shape': (shape[0] + '-' + suffix, *shape[1:]),
                  'skippedSlots': skipped_slots, 'baselineCase': shape[0]})
scales = [0x0000, 0x13ad, 0xa65b, 0x2d71, 0xb5b3, 0x3835, 0xbc27, 0x4159, 0x4913]
high_bits = [0x00, 0xff, 0x55, 0xaa, 0x81, 0x18, 0x7e, 0xe7]
subscales = [0x00, 0xff, 0x10, 0xef, 0x53, 0xa7, 0x38, 0xc2]
manifest = []
details = []
for case in cases:
    seed = case['seed']
    name, k, m, shared, padded, duplicates = case['shape']
    directory = fixtures / name
    directory.mkdir(exist_ok=True)
    experts, slots, blocks = 11, 10, k // 256
    input_rows = 1 if shared else slots
    offset_a, offset_x, offset_ids, offset_d = (1, 3, 2, 7) if padded else (0, 0, 0, 0)
    stride01 = blocks + int(padded)
    stride02 = m * stride01 + (3 if padded else 0)
    stride11 = k + (5 if padded else 0)
    stride12 = input_rows * stride11 + (3 if padded else 0)
    byte_count = ((offset_a + experts * stride02) * 110 + 16 + 3) // 4 * 4
    weights = bytearray([0x42]) * byte_count
    for expert in range(experts):
        for row in range(m):
            for block in range(blocks):
                base = (offset_a + expert * stride02 + row * stride01 + block) * 110
                phase = expert * 5 + row * 3 + block * 7 + seed
                struct.pack_into('<H', weights, base, scales[phase % len(scales)])
                weights[base + 2:base + 66] = bytes((expert * 29 + row * 71 + block * 17 + j * 43 + seed * 31) % 256 for j in range(64))
                weights[base + 66:base + 74] = bytes(high_bits[(phase + j) % len(high_bits)] for j in range(8))
                weights[base + 74:base + 106] = bytes((0 if (phase + j) % 7 == 0 else 255 if (phase + j) % 7 == 1
                    else (expert * 67 + row * 41 + block * 19 + j * 37 + seed * 13) % 256) for j in range(32))
                weights[base + 106:base + 110] = bytes(subscales[(phase + j) % len(subscales)] for j in range(4))
    input_count = offset_x + input_rows * stride11 + 16
    input_data = bytearray(struct.pack('<f', 99.5) * input_count)
    for input_row in range(input_rows):
        for i in range(k):
            pair = i // 2
            value = ((pair * 53 + input_row * 17 + seed * 31) % 1009 - 504) / 997
            value *= (-1 if i % 2 else 1) * 2 ** (pair % 17 - 8)
            if (i + input_row) % 113 == 0:
                value = 0.0
            struct.pack_into('<f', input_data, (offset_x + input_row * stride11 + i) * 4, value)
    selected = [10, 3, 7, 0, 9, 1, 8, 2, 6, 4]
    if duplicates:
        selected[-1] = selected[0]
    for slot in case['skippedSlots']:
        selected[slot] = 0xffffffff
    ids = [0xdeadbeef] * (offset_ids + slots + 4)
    ids[offset_ids:offset_ids + slots] = selected
    ids_data = struct.pack('<' + 'I' * len(ids), *ids)
    params = [offset_a, offset_x, offset_ids, offset_d, k, m, experts, slots, input_rows,
              stride01, stride11, stride02, stride12]
    (directory / 'weights.bin').write_bytes(weights)
    (directory / 'input.bin').write_bytes(input_data)
    (directory / 'ids.bin').write_bytes(ids_data)
    (directory / 'params.bin').write_bytes(struct.pack('<13I', *params))
    subprocess.run([str(oracle), str(directory / 'weights.bin'), str(directory / 'input.bin'), str(directory / 'ids.bin'),
                    str(directory / 'params.bin'), str(directory / 'reference.bin')], check=True)
    reference = (directory / 'reference.bin').read_bytes()
    if len(reference) != m * slots * 16:
        raise ValueError('Unexpected reference length: ' + name)
    for dot, sum_abs in struct.iter_unpack('<dd', reference):
        if not math.isfinite(dot) or not math.isfinite(sum_abs) or sum_abs < abs(dot) or sum_abs < 0:
            raise ValueError('Invalid reference: ' + name)
    if case['baselineCase'] is not None:
        baseline = next(row for row in manifest if row['name'] == case['baselineCase'])
        baseline_directory = fixtures / case['baselineCase']
        if params != baseline['params'] or weights != (baseline_directory / 'weights.bin').read_bytes() \
                or input_data != (baseline_directory / 'input.bin').read_bytes():
            raise ValueError('Skipped-slot fixture changed its baseline inputs: ' + name)
        baseline_reference = (baseline_directory / 'reference.bin').read_bytes()
        for slot in range(slots):
            actual = reference[slot * m * 16:(slot + 1) * m * 16]
            expected = bytes(m * 16) if slot in case['skippedSlots'] \
                else baseline_reference[slot * m * 16:(slot + 1) * m * 16]
            if actual != expected:
                raise ValueError('Skipped-slot oracle changed a baseline row: ' + name)
    row = {'name': name, 'k': k, 'm': m, 'storedExperts': experts, 'slots': slots, 'bNe1': input_rows,
           'baselineCase': case['baselineCase'], 'skippedSlots': case['skippedSlots'],
           'params': params, 'weightsFile': f'{name}/weights.bin', 'inputFile': f'{name}/input.bin',
           'idsFile': f'{name}/ids.bin', 'referenceFile': f'{name}/reference.bin',
           'hashes': {key: hashlib.sha256((directory / file).read_bytes()).hexdigest()
                      for key, file in [('weights', 'weights.bin'), ('input', 'input.bin'), ('ids', 'ids.bin'), ('reference', 'reference.bin')]}}
    manifest.append(row)
    details.append({'name': name, 'padded': padded, 'sharedInput': shared, 'duplicateIds': duplicates,
                    'baselineCase': case['baselineCase'], 'skippedSlots': case['skippedSlots'],
                    'selectedIds': selected, 'weightsBytes': len(weights), 'referenceOutputs': m * slots,
                    'referenceSumAbsMax': max(pair[1] for pair in struct.iter_unpack('<dd', reference))})
(fixtures / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
profiles = [' '.join([row['name'], *map(str, row['params']),
    str((fixtures / row['weightsFile']).stat().st_size), str((fixtures / row['inputFile']).stat().st_size),
    str((fixtures / row['idsFile']).stat().st_size)]) for row in manifest]
(fixtures / 'profiles.txt').write_text('\n'.join(profiles) + '\n')
metadata = {
    'scope': 'Deterministic IQ3_S synthetic fixtures with upstream CPU reference; no GPU or model weights',
    'upstreamCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip(),
    'cpuOracle': 'Unmodified pinned dequantize_row_iq3_s function, upstream block/grid/FP16 conversion headers, per-block Float32 dequantization, Float64 products and sums with actual uploaded Float32 input',
    'dequantFunctionSha256': hashlib.sha256(function.encode()).hexdigest(),
    'sourceSha256': {name: hashlib.sha256((upstream / name).read_bytes()).hexdigest() for name in
        ['ggml/src/ggml-quants.c', 'ggml/src/ggml-common.h', 'ggml/src/ggml-impl.h', 'ggml/include/ggml.h']},
    'generatedOracleSha256': hashlib.sha256(oracle_source.read_bytes()).hexdigest(),
    'oracleExecutableSha256': hashlib.sha256(oracle.read_bytes()).hexdigest(),
    'compileCommand': command,
    'benchmarkCase': 'hot-k2560-m640-separate-contiguous',
    'suggestedNumericalDispatchSplit': 17,
    'uniformLayout': ['offset_src0', 'offset_src1', 'offset_ids', 'offset_dst', 'k', 'm', 'n_expert',
                      'n_expert_used', 'b_ne1', 'stride_01', 'stride_11', 'stride_02', 'stride_12'],
    'referenceLayout': 'Float64LE interleaved [dot,sumabs], index=slot*m+row; no output offset in reference',
    'ordinaryFixtureStatus': 'Not generated; indexed-first contract',
    'cases': details,
}
(lab / 'fixture-source.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps({'cases': len(manifest), 'referenceOutputs': sum(row['m'] * row['slots'] for row in manifest),
                  'manifest': str(fixtures / 'manifest.json'), 'gpuDispatches': 0}))
