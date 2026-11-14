import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { compareQsaTraces, readQsaNodes, runQsaCli, type MaskMode } from '#tests/runtimes/llama-cpp/tools/qsa.test-support';

const short = ['short-prefill', 'short-reused', 'prefill', 'before-threshold', 'shared-prefix', 'sequence-zero-divergence', 'sequence-one-divergence', 'after-sequence-removal', 'mixed', 'embedding-chunks', 'unified-prefix', 'parallel-sequences'];
const phases = [...short, 'cross-threshold', 'restore-cross-threshold', 'reserve', 'parallel-reserve', 'ranked-prefill'];
const header = 'phase,indexer,top_k,query,raw_key,selection_mask,fused_mask,pools64,pools128';

async function writeTraces({ root, maskMode }: { root: string; maskMode: MaskMode }): Promise<{ reference: string; optimized: string }> {
  const paths = { reference: join(root, 'reference'), optimized: join(root, 'optimized') };
  let fused: boolean;
  switch (maskMode) {
  case 'fused': fused = true;
    break;
  case 'unfused': fused = false;
    break;
  default: {
    const _ex: never = maskMode;
    throw new Error(`Unknown mask mode: ${String(_ex)}`);
  }
  }
  for (const [name, path] of Object.entries(paths)) {
    await mkdir(path, { recursive: true });
    const rows = [header];
    for (const phase of phases) {
      let ranked = 4;
      let selection = 4;
      if (name === 'optimized' && short.includes(phase)) {
        ranked = 0;
        selection = 1;
      } else if (name === 'optimized' && ['cross-threshold', 'restore-cross-threshold'].includes(phase)) {
        ranked = 2;
        selection = 2;
      }
      rows.push([phase, ranked, ranked, ranked, 4, selection, fused ? ranked : 0, 1, 1].join(','));
    }
    await writeFile(join(path, 'nodes.csv'), `${rows.join('\n')}\n`);
    const bytes = Buffer.alloc(12);
    [0.25, -0.5, 2].forEach((value, index) => bytes.writeFloatLE(value, index * 4));
    await writeFile(join(path, 'logits.f32'), bytes);
  }
  return paths;
}

test('QSA comparison requires real branch coverage and exact finite output parity', async context => {
  await using temporary = {
    path: await mkdtemp(join(tmpdir(), 'qsa-traces-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  for (const maskMode of ['fused', 'unfused'] as const) {
    await context.test(`accepts independent ${maskMode} traces`, async () => {
      const paths = await writeTraces({ root: temporary.path, maskMode });
      const result = await compareQsaTraces({ ...paths, maskMode });
      assert.equal(result.identical, true);
      assert.equal(result.floatValues, 3);
    });
  }
  await context.test('rejects two baseline graph traces despite equal outputs', async () => {
    const paths = await writeTraces({ root: temporary.path, maskMode: 'fused' });
    await copyFile(join(paths.reference, 'nodes.csv'), join(paths.optimized, 'nodes.csv'));
    await assert.rejects(compareQsaTraces({ ...paths, maskMode: 'fused' }), /All-pools/u);
  });
  for (const invalid of ['empty', 'malformed', 'nonfinite', 'constant', 'different'] as const) {
    await context.test(`rejects ${invalid} output`, async () => {
      const paths = await writeTraces({ root: temporary.path, maskMode: 'fused' });
      const original = await readFile(join(paths.reference, 'logits.f32'));
      switch (invalid) {
      case 'empty': await writeFile(join(paths.optimized, 'logits.f32'), Buffer.alloc(0));
        break;
      case 'malformed': await writeFile(join(paths.optimized, 'logits.f32'), Buffer.alloc(5));
        break;
      case 'nonfinite': original.writeFloatLE(Number.NaN, 0);
        await writeFile(join(paths.optimized, 'logits.f32'), original);
        break;
      case 'constant':
        original.fill(0);
        await writeFile(join(paths.reference, 'logits.f32'), original);
        await writeFile(join(paths.optimized, 'logits.f32'), original);
        break;
      case 'different': original.writeFloatLE(0.2501, 0);
        await writeFile(join(paths.optimized, 'logits.f32'), original);
        break;
      default: {
        const _ex: never = invalid;
        throw new Error(`Unknown corruption: ${String(_ex)}`);
      }
      }
      await assert.rejects(compareQsaTraces({ ...paths, maskMode: 'fused' }));
    });
  }
  await context.test('requires the selected mask mode and matching schedule', async () => {
    const paths = await writeTraces({ root: temporary.path, maskMode: 'unfused' });
    await assert.rejects(compareQsaTraces({ ...paths, maskMode: 'fused' }), /fused mask not exercised/u);
    const nodes = await readFile(join(paths.optimized, 'nodes.csv'), 'utf8');
    const [columns, ...rows] = nodes.trimEnd().split('\n');
    await writeFile(join(paths.optimized, 'nodes.csv'), `${[columns, ...rows.reverse()].join('\n')}\n`);
    await assert.rejects(compareQsaTraces({ ...paths, maskMode: 'unfused' }), /schedules differ/u);
  });
  await context.test('rejects missing phases, unknown columns and negative counts', async () => {
    const paths = await writeTraces({ root: temporary.path, maskMode: 'fused' });
    const nodes = await readFile(join(paths.reference, 'nodes.csv'), 'utf8');
    for (const changed of [nodes.replace(/ranked-prefill[^\n]+\n/u, ''), nodes.replace('pools128', 'unknown'), nodes.replace('short-prefill,4', 'short-prefill,-1')]) {
      await writeFile(join(paths.reference, 'nodes.csv'), changed);
      await assert.rejects(readQsaNodes({ path: join(paths.reference, 'nodes.csv') }));
    }
  });
});

function selectionMask({ scores, pools, tail, cells, budget }: { scores: readonly number[]; pools: readonly (readonly number[])[]; tail: readonly number[]; cells: number; budget: number }): number[] {
  const selected = scores.map((_score, index) => index).sort((a, b) => {
    const left = scores[a];
    const right = scores[b];
    assert.ok(left !== undefined && right !== undefined);
    return right - left || a - b;
  }).slice(0, budget);
  const visible = new Set(tail);
  for (const index of selected) {
    const score = scores[index];
    const pool = pools[index];
    assert.ok(score !== undefined && pool !== undefined);
    const live = Math.min(1, Math.max(0, score + 1));
    if (live === 1) {
      for (const cell of pool) {
        visible.add(cell);
      }
    } else if (live !== 0) {
      throw new Error('Qwen4Exp score invariant was violated');
    }
  }
  return Array.from({ length: cells }, (_value, cell) => visible.has(cell) ? 0 : -Infinity);
}

test('independent QSA mask oracle preserves visibility, tails and storage order', () => {
  let seed = 81371;
  const random = (): number => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed;
  };
  const scores = [0, 1e-8, 0.5, 5, 1e20];
  for (let count = 1; count <= 8; count++) {
    for (let pattern = 0; pattern < 2 ** count; pattern++) {
      for (const pooling of [1, 2, 4]) {
        const pools = Array.from({ length: count }, (_value, index) => Array.from({ length: pooling }, (_cell, offset) => index * pooling + offset));
        for (let index = pools.length - 1; index > 0; index--) {
          const other = random() % (index + 1);
          const a = pools[index];
          const b = pools[other];
          assert.ok(a !== undefined && b !== undefined);
          pools[index] = b;
          pools[other] = a;
        }
        const ranked = Array.from({ length: count }, (_value, index) => (pattern & (1 << index)) === 0 ? -Infinity : scores[random() % scores.length]!);
        const unranked = Array.from({ length: count }, (_value, index) => (pattern & (1 << index)) === 0 ? -Infinity : 0);
        const cells = count * pooling + pooling;
        for (const tail of [[], [cells], [cells - 1], Array.from({ length: pooling }, (_value, index) => cells - pooling + index)]) {
          assert.deepEqual(selectionMask({ scores: ranked, pools, tail, cells, budget: count }), selectionMask({ scores: unranked, pools, tail, cells, budget: count }));
        }
      }
    }
  }
  assert.deepEqual(selectionMask({ scores: [2, -Infinity], pools: [[1, 3], [4, 6]], tail: [9], cells: 10, budget: 2 }), [-Infinity, 0, -Infinity, 0, -Infinity, -Infinity, -Infinity, -Infinity, -Infinity, 0]);
  assert.notDeepEqual(selectionMask({ scores: [0, 1, 5], pools: [[0], [1], [2]], tail: [], cells: 3, budget: 2 }), selectionMask({ scores: [0, 0, 0], pools: [[0], [1], [2]], tail: [], cells: 3, budget: 2 }));
  assert.throws(() => selectionMask({ scores: [-0.5], pools: [[0]], tail: [], cells: 1, budget: 1 }), /score invariant/u);
  assert.notDeepEqual(selectionMask({ scores: [2, -Infinity], pools: [[0], [1]], tail: [], cells: 2, budget: 2 }), selectionMask({ scores: [-Infinity, 2], pools: [[0], [1]], tail: [], cells: 2, budget: 2 }));
  const mask = selectionMask({ scores: [2, -Infinity], pools: [[1, 3], [4, 6]], tail: [9, 10], cells: 10, budget: 2 });
  for (const value of mask) {
    assert.ok(value === 0 || value === -Infinity);
    assert.equal(Math.fround(value), value);
    // Both zero and negative infinity have exact IEEE binary16 encodings.
    const binary16 = value === 0 ? 0x0000 : 0xfc00;
    assert.equal(binary16 === 0x0000 ? 0 : -Infinity, value);
  }
});

test('QSA dump-row redirection requires the exact float32 range guard', () => {
  const redirect = ({ cell, cells, slot }: { cell: number; cells: number; slot: number }): number => {
    const dump = Math.fround(Math.fround(slot + 1) + Math.fround(cells - 1));
    return Math.trunc(Math.fround(Math.fround(Math.fround(cell) - dump) + dump));
  };
  for (const cells of [256, 2 ** 23, 2 ** 24 - 8]) {
    for (const cell of [0, 1, 3, cells / 2, cells - 1]) {
      for (let slot = 0; slot < 8; slot++) {
        assert.equal(redirect({ cell, cells, slot }), cell);
      }
    }
  }
  assert.notEqual(redirect({ cell: 1, cells: 2 ** 24, slot: 0 }), redirect({ cell: 1, cells: 2 ** 24, slot: 2 }));
});

test('QSA help exits without configuring a native build', async () => {
  await runQsaCli({ argv: ['--help'] });
  await runQsaCli({ argv: ['--local', '--help'] });
});
