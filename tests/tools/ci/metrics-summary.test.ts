import assert from 'node:assert/strict';
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { MetricContext, SpanRecord } from '#tools/ci/metrics';
import { ninjaSummary, readRecords, summarize, unionLength } from '#tools/ci/summary';

const context: MetricContext = { runId: undefined, runAttempt: undefined, job: undefined, sourceCommit: undefined, runtime: undefined, profile: undefined, variant: undefined };

function pair({ id, parent, start, length, phase }: { id: string; parent: string | undefined; start: number; length: number; phase: string }): SpanRecord[] {
  const common = { ...context, schemaVersion: 1 as const, mode: 'basic' as const, clock: 'monotonic_ns' as const, spanId: id.repeat(32), parentSpanId: parent?.repeat(32), startedNs: start, phase, packageKind: undefined };
  return [{ ...common, event: 'span_start' }, { ...common, event: 'span_end', elapsedNs: length, result: 'success', exitCode: 0 }];
}

test('parallel child coverage is unioned, incomplete spans are unknown', () => {
  assert.equal(unionLength({ intervals: [{ start: 2, end: 7 }, { start: 4, end: 9 }] }), 7);
  const records = [
    ...pair({ id: 'a', parent: undefined, start: 0, length: 10, phase: 'parent' }),
    ...pair({ id: 'b', parent: 'a', start: 2, length: 5, phase: 'child' }),
    ...pair({ id: 'c', parent: 'a', start: 4, length: 5, phase: 'child' }),
  ];
  const report = summarize({ records, errors: 0 });
  assert.equal(report.phases['parent']?.exclusiveNs, 3);
  assert.equal(report.phases['child']?.inclusiveNs, 10);
  const incomplete = summarize({ records: [records[0]!], errors: 0 });
  assert.equal(incomplete.phases['parent']?.incomplete, 1);
  assert.equal(incomplete.phases['parent']?.inclusiveNs, null);
  assert.equal(incomplete.phases['parent']?.exclusiveNs, null);
  const bad = summarize({ records: [...records, records[0]!], errors: 0 });
  assert.ok(bad.invalidOrMissingRecords > 0);
  assert.equal(bad.phases['parent']?.exclusiveNs, null);
});

test('Ninja outputs use covered intervals rather than duplicated command time', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-ninja-test-'));
  try {
    const file = join(root, '.ninja_log');
    await writeFile(file, '# ninja log v5\n0\t8\t0\ta.o\taa\n0\t8\t0\tb.o\tbb\n8\t18\t0\truntime/core.mjs\tcc\n8\t18\t0\truntime/core.wasm\tcc\n');
    const result = await ninjaSummary({ path: file });
    assert.equal(result.status, 'observed');
    if (result.status !== 'observed') {
      throw new Error('Expected a Ninja observation');
    }
    assert.equal(result.outputRows, 4);
    assert.equal(result.coveredMs, 18);
    assert.equal(result.categories['runtime-outputs']?.coveredMs, 10);
    await writeFile(file, '# ninja log v99\n');
    assert.equal((await ninjaSummary({ path: file })).status, 'unknown-format');
    await writeFile(file, '# ninja log v5\n100\t1\t0\tx\tcc\n');
    assert.equal((await ninjaSummary({ path: file })).status, 'malformed');
    assert.equal((await ninjaSummary({ path: join(root, 'missing') })).status, 'unavailable');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('invalid and linked diagnostic records are counted, never treated as successful spans', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-bad-metrics-'));
  try {
    await writeFile(join(root, 'events-one.jsonl'), '{broken\n{"schemaVersion":999}\n');
    await symlink('events-one.jsonl', join(root, 'events-linked.jsonl'));
    const result = await readRecords({ directory: root });
    assert.deepEqual(result.records, []);
    assert.equal(result.errors, 3);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
