import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { cacheObservation, createMetrics, environmentObservation, metricContext, Metrics } from '#tools/ci/metrics';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';
import { readRecords, summarize } from '#tools/ci/summary';

test('nested typed work retains parent relationships and the original failure', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-metrics-test-'));
  try {
    const directory = join(root, 'metrics');
    const metrics = new Metrics({
      checkoutRoot: process.cwd(),
      enabled: true,
      directory,
      context: metricContext({ context: { job: 'build' } }),
    });
    const original = new Error('operation error with private details');
    await assert.rejects(metrics.measure({
      phase: 'outer',
      packageKind: undefined,
      operation: () =>
        metrics.measure({
          phase: 'inner',
          packageKind: 'llama',
          operation: async () => {
            throw original;
          },
        }),
    }), error => error === original);
    const start = metrics.records.find(record => record.event === 'span_start' && record.phase === 'outer');
    const end = metrics.records.find(record => record.event === 'span_end' && record.phase === 'inner');
    assert.ok(start && end && end.event === 'span_end');
    assert.equal(end.parentSpanId, start.spanId);
    assert.equal(end.result, 'failure');
    const { records, errors } = await readRecords({ directory });
    assert.equal(errors, 0);
    const report = summarize({ records, errors });
    assert.equal(report.phases['inner:llama']?.failed, 1);
    const content = (await Promise.all((await readdir(directory)).map(file => readFile(join(directory, file), 'utf8')))).join('');
    assert.doesNotMatch(content, /DO-NOT-RECORD|private details/u);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('disabled or broken metrics do not prevent work or replace its result', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-metrics-off-'));
  try {
    const missing = join(root, 'off');
    const off = new Metrics({ checkoutRoot: process.cwd(), enabled: false, directory: missing, context: metricContext({ context: {} }) });
    assert.equal(await off.measure({ phase: 'work', packageKind: undefined, operation: async () => 4 }), 4);
    await assert.rejects(readdir(missing));
    const broken = join(root, 'file');
    await writeFile(broken, 'not a directory');
    const metrics = new Metrics({ checkoutRoot: process.cwd(), enabled: true, directory: broken, context: metricContext({ context: {} }) });
    const original = new Error('original failure');
    await assert.rejects(metrics.measure({
      phase: 'work',
      packageKind: undefined,
      operation: async () => {
        throw original;
      },
    }), error => error === original);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('cache observations distinguish exact, fallback, missing and unsuccessful actions', () => {
  const prefix = 'bic-cc-partition-';
  const observe = ({ kind, requested, matched, hit, outcome }: { kind: 'em' | 'cc'; requested: string; matched: string; hit: string; outcome: string }) =>
    cacheObservation({ kind, requested, matched, hit, outcome, saveEligible: false }).state;
  assert.equal(observe({ kind: 'cc', requested: prefix + 'a'.repeat(40), matched: prefix + 'b'.repeat(40), hit: 'false', outcome: 'success' }), 'prefix');
  assert.equal(observe({ kind: 'em', requested: 'bic-em-partition-' + 'a'.repeat(40), matched: 'bic-em-partition-' + 'b'.repeat(40), hit: 'false', outcome: 'success' }), 'prefix');
  assert.equal(observe({ kind: 'em', requested: 'bic-em-partition-' + 'a'.repeat(40), matched: 'bic-em-other-' + 'b'.repeat(40), hit: 'false', outcome: 'success' }), 'unexpected-key-or-status');
  assert.equal(observe({ kind: 'cc', requested: 'a'.repeat(40), matched: 'b'.repeat(40), hit: 'false', outcome: 'success' }), 'unexpected-key-or-status');
  assert.equal(observe({ kind: 'em', requested: 'expected', matched: 'other', hit: 'false', outcome: 'success' }), 'unexpected-key-or-status');
  assert.equal(observe({ kind: 'em', requested: 'expected', matched: '', hit: '', outcome: 'success' }), 'miss');
  assert.equal(observe({ kind: 'em', requested: 'expected', matched: 'expected', hit: 'true', outcome: 'success' }), 'exact');
  assert.equal(observe({ kind: 'em', requested: 'expected', matched: '', hit: '', outcome: 'failure' }), 'not-completed');
});

test('cache wiring observation exposes state without disclosing paths', () => {
  const root = '/tmp/example-private-workspace';
  const value = environmentObservation({
    workRoot: root,
    settings: {
      emCache: join(root, '.tools/emsdk/upstream/emscripten/cache'),
      compilerWrapper: 'ccache',
      ccacheDirectory: join(root, 'objects'),
      ccacheConfiguration: join(root, '.cache/ccache.conf'),
    },
  });
  assert.equal(value.emCache, 'expected');
  assert.equal(value.compilerWrapper, 'ccache');
  assert.equal(value.ccacheConfiguration, 'expected');
  assert.equal(value.ccacheDirectory, 'set');
  assert.doesNotMatch(JSON.stringify(value), /example-private-workspace/u);
});

test('metrics factory reads current values and retains directory defaults', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-metrics-factory-'));
  try {
    using environment = stubEnvironment({ values: { BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: '', RUNNER_TEMP: root, GITHUB_JOB: 'fixture', BIC_CACHE_ID: 'current-compiler' } });
    const metrics = createMetrics({ checkoutRoot: process.cwd() });
    await metrics.measure({ phase: 'factory', packageKind: undefined, operation: async () => undefined });
    await metrics.observeEnvironment();
    const { records, errors } = await readRecords({ directory: join(root, 'bic-pipeline-metrics') });
    assert.equal(errors, 0);
    assert.ok(records.some(record => record.event === 'span_start' && record.job === 'fixture'));
    using disabled = stubEnvironment({ values: { BIC_METRICS_MODE: 'off' } });
    const off = createMetrics({ checkoutRoot: process.cwd() });
    await off.measure({ phase: 'off', packageKind: undefined, operation: async () => undefined });
    assert.equal(off.records.length, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
