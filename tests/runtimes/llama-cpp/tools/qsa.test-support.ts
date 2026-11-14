import assert from 'node:assert/strict';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import yargs from 'yargs';
import type { FileIdentity } from '#dto/shared';
import { createMetrics, environmentObservation } from '#tools/ci/metrics';
import { fileIdentity, optionalStat } from '#tools/shared/files';
import { ProcessError, runProcess } from '#tools/shared/process';
import { observeSource, type SourceMode } from '#tools/source/capture';
import { buildPrepared, prepareBuildInputs, type BuildContext } from '#runtimes/llama-cpp/tools/build';
import { loadConfiguration } from '#runtimes/llama-cpp/tools/config';
import { prepareSources } from '#runtimes/llama-cpp/tools/prepare';

const countFields = ['indexer', 'top_k', 'query', 'raw_key', 'selection_mask', 'fused_mask', 'pools64', 'pools128'] as const;
type Counts = Record<typeof countFields[number], number>;
const zeroCounts: Counts = { indexer: 0, top_k: 0, query: 0, raw_key: 0, selection_mask: 0, fused_mask: 0, pools64: 0, pools128: 0 };
const shortPhases = ['short-prefill', 'short-reused', 'prefill', 'before-threshold', 'shared-prefix', 'sequence-zero-divergence', 'sequence-one-divergence', 'after-sequence-removal', 'mixed', 'embedding-chunks', 'unified-prefix', 'parallel-sequences'];
const crossPhases = ['cross-threshold', 'restore-cross-threshold'];
const phases = [...shortPhases, ...crossPhases, 'reserve', 'parallel-reserve', 'ranked-prefill'];
const phaseSet = new Set(phases);
export type MaskMode = 'fused' | 'unfused';

function csvRows({ text }: { text: string }): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index++;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      assert.equal(field, '', 'Malformed quoted CSV field');
      quoted = true;
    } else if (character === ',') {
      row.push(field);
      field = '';
    } else if (character === '\r' || character === '\n') {
      if (character === '\r' && text[index + 1] === '\n') {
        index++;
      }
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += character;
    }
  }
  assert.equal(quoted, false, 'Unterminated quoted CSV field');
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

export async function readQsaNodes({ path }: { path: string }): Promise<{ totals: Record<string, Counts>; schedule: string[] }> {
  const [header, ...rows] = csvRows({ text: await readFile(path, 'utf8') });
  assert.deepEqual(header, ['phase', ...countFields], 'Missing or unknown graph-observation columns');
  const totals: Record<string, Counts> = {};
  const schedule: string[] = [];
  for (const row of rows) {
    assert.equal(row.length, countFields.length + 1, 'Missing or unknown observation columns');
    const phase = row[0];
    assert.ok(phase !== undefined && phaseSet.has(phase), 'Unknown observation phase');
    const counts = totals[phase] ?? { ...zeroCounts };
    countFields.forEach((field, index) => {
      const value = row[index + 1];
      assert.ok(value !== undefined && /^[+-]?\d+$/u.test(value.trim()), 'Malformed observation count');
      const number = Number(value);
      assert.ok(Number.isSafeInteger(number) && number >= 0, 'Negative or unsupported observation count');
      counts[field] += number;
    });
    totals[phase] = counts;
    schedule.push(phase);
  }
  assert.deepEqual(new Set(schedule), phaseSet, 'Incomplete phase coverage');
  return { totals, schedule };
}

function countsFor({ nodes, phase }: { nodes: Record<string, Counts>; phase: string }): Counts {
  const counts = nodes[phase];
  assert.ok(counts, `Missing graph phase: ${phase}`);
  return counts;
}

export async function compareQsaTraces({ reference, optimized, maskMode }: { reference: string; optimized: string; maskMode: MaskMode }) {
  const original = await readFile(join(reference, 'logits.f32'));
  const changed = await readFile(join(optimized, 'logits.f32'));
  assert.ok(original.length > 0 && original.length % 4 === 0 && original.length === changed.length, 'Missing, malformed or unequal-length output trace');
  for (const bytes of [original, changed]) {
    for (let offset = 0; offset < bytes.length; offset += 4) {
      assert.ok(Number.isFinite(bytes.readFloatLE(offset)), 'Nonfinite model output');
    }
  }
  const first = original.readFloatLE(0);
  let nonconstant = false;
  let maximumError = 0;
  for (let offset = 0; offset < original.length; offset += 4) {
    if (original.readFloatLE(offset) !== first) {
      nonconstant = true;
    }
    maximumError = Math.max(maximumError, Math.abs(original.readFloatLE(offset) - changed.readFloatLE(offset)));
  }
  assert.ok(nonconstant, 'Degenerate constant model output');
  assert.ok(original.equals(changed), `Full-model outputs differ; maximum absolute error: ${maximumError}`);
  const before = await readQsaNodes({ path: join(reference, 'nodes.csv') });
  const after = await readQsaNodes({ path: join(optimized, 'nodes.csv') });
  assert.deepEqual(before.schedule, after.schedule, 'Observation schedules differ');
  for (const phase of phases.filter(value => value !== 'reserve' && value !== 'parallel-reserve')) {
    const a = countsFor({ nodes: before.totals, phase });
    const b = countsFor({ nodes: after.totals, phase });
    switch (maskMode) {
    case 'fused':
      assert.ok(a.fused_mask > 0 && a.fused_mask === a.selection_mask, `Reference fused mask not exercised: ${phase}`);
      assert.equal(b.fused_mask, b.top_k, `Ranked fused mask coverage differs: ${phase}`);
      if (shortPhases.includes(phase)) {
        assert.equal(b.fused_mask, 0, `All-pools path unexpectedly uses a fused ranked mask: ${phase}`);
      }
      break;
    case 'unfused':
      assert.ok(a.fused_mask === 0 && b.fused_mask === 0, `Unfused run unexpectedly executed a fused mask: ${phase}`);
      break;
    default: {
      const _ex: never = maskMode;
      throw new Error(`Unknown mask mode: ${String(_ex)}`);
    }
    }
    assert.ok(a.selection_mask > 0 && a.selection_mask === a.top_k, `Reference selection mask coverage differs: ${phase}`);
    assert.ok(a.raw_key > 0 && b.raw_key === a.raw_key, `Per-layer key-cache write coverage differs: ${phase}`);
  }
  for (const phase of shortPhases) {
    const a = countsFor({ nodes: before.totals, phase });
    const b = countsFor({ nodes: after.totals, phase });
    for (const operation of ['indexer', 'top_k', 'query'] as const) {
      assert.ok(a[operation] > 0 && b[operation] === 0, `All-pools bypass not exercised: ${phase}/${operation}`);
    }
    assert.ok(b.selection_mask > 0 && b.selection_mask < a.selection_mask, `Graph-local mask sharing not exercised: ${phase}`);
  }
  for (const phase of crossPhases) {
    const a = countsFor({ nodes: before.totals, phase });
    const b = countsFor({ nodes: after.totals, phase });
    for (const operation of ['indexer', 'top_k', 'query'] as const) {
      assert.ok(b[operation] > 0 && b[operation] < a[operation], `Both threshold branches were not exercised: ${phase}/${operation}`);
    }
  }
  assert.deepEqual(after.totals['cross-threshold'], after.totals['restore-cross-threshold'], 'Restoration did not reproduce graph branch observations');
  assert.deepEqual(after.totals['ranked-prefill'], before.totals['ranked-prefill'], 'Ranked path was not preserved');
  assert.ok(countsFor({ nodes: after.totals, phase: 'ranked-prefill' }).indexer > 0, 'Ranked path was not exercised');
  return { identical: true, maskMode, ...await fileIdentity({ path: join(reference, 'logits.f32') }), floatValues: original.length / 4, referenceNodes: before.totals, optimizedNodes: after.totals, scope: 'synthetic native full-model parity, not a real-model or WebGPU throughput result' };
}

export async function runQsaRegression({ context, mode, jobs }: { context: BuildContext; mode: SourceMode; jobs: number }): Promise<void> {
  assert.ok(process.platform === 'linux', 'The static-library fixture linker currently requires Linux');
  assert.ok(Number.isSafeInteger(jobs) && jobs > 0, 'jobs must be positive');
  const configuration = await loadConfiguration({ checkoutRoot: context.checkoutRoot });
  const profile = configuration.profiles['cpu-wasm32'];
  const variant = configuration.variants['browser'];
  assert.ok(profile !== undefined && variant !== undefined, 'Missing native fixture base configuration');
  const source = await observeSource({ checkoutRoot: context.checkoutRoot, mode });
  const results = join(context.outputRoot, 'qsa-results');
  const logs = join(results, 'logs');
  await mkdir(logs, { recursive: true });
  const status: { scope: string; commands: { name: string; command: readonly string[]; exit: number | undefined; elapsedSeconds: number }[]; fixtures: Record<string, FileIdentity>; libraries: Record<string, FileIdentity>; maskParity: ({ quant: string; mode: string } & FileIdentity)[]; flashMaskParity: ({ maskType: string; mode: string } & FileIdentity)[]; success: boolean } = { scope: 'synthetic native parity; not WebGPU or real-model throughput', commands: [], fixtures: {}, libraries: {}, maskParity: [], flashMaskParity: [], success: false };
  const save = async () => writeFile(join(results, 'status.json'), `${JSON.stringify(status, undefined, 2)}\n`);
  const command = async ({ name, args }: { name: string; args: readonly string[] }): Promise<void> => {
    const [executable, ...arguments_] = args;
    assert.ok(executable);
    const started = performance.now();
    let exit: number | undefined;
    try {
      const result = await runProcess({ command: executable, args: arguments_, cwd: context.checkoutRoot, env: { ...process.env, OMP_NUM_THREADS: '1' }, capture: 'file' });
      exit = result.exitCode;
      await writeFile(join(logs, `${name}.txt`), Buffer.concat([result.stdout, result.stderr]));
    } catch (error) {
      if (error instanceof ProcessError) {
        exit = error.exitCode ?? undefined;
        await writeFile(join(logs, `${name}.txt`), Buffer.concat([error.stdout, error.stderr]));
      }
      throw error;
    } finally {
      status.commands.push({ name, command: args, exit, elapsedSeconds: (performance.now() - started) / 1000 });
      await save();
    }
  };
  const executables: Record<string, string> = {};
  try {
    for (const [name, qsa] of [['reference', false], ['optimized', true]] as const) {
      const selection = { profile: name, variant: 'browser', kind: 'native' as const, jobs, fresh: false, includeToolchainNotices: false, configureOnly: false, python: 'python3' };
      const captured = await prepareBuildInputs({ context, source, selection, additionalInputs: ['tests/runtimes/llama_cpp/qsa-selection/CMakeLists.txt', 'tests/runtimes/llama_cpp/qsa-selection/model.cpp'] });
      const build = join(context.outputRoot, 'build/native', name, 'browser');
      const prepared = await prepareSources({ inputs: captured.inputs, output: join(build, 'prepared'), python: selection.python, options: { target: 'native', webgpu: false, gpuProfile: false, fileRows: true, qsa, mtmdMode: 'cpu' } });
      try {
        const built = await buildPrepared({ context, source: captured, prepared, selection, configuration, chosen: { profile, variant } });
        assert.equal(built.kind, 'built');
      } finally {
        for (const phase of ['configure', 'compile']) {
          const log = join(build, `${phase}.log`);
          if (await optionalStat({ path: log })) {
            await copyFile(log, join(logs, `${name}-${phase}.txt`));
          }
        }
      }
      const test = join(context.outputRoot, `test-${name}`);
      await command({ name: `${name}-test-configure`, args: ['cmake', '-S', join(captured.inputs.compileRoot, 'tests/runtimes/llama_cpp/qsa-selection'), '-B', test, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', `-DLCB_LLAMA_SOURCE=${prepared.root}`, `-DLCB_NATIVE_BUILD=${build}`, `-DCMAKE_C_COMPILER=${process.env['CC'] ?? 'clang'}`, `-DCMAKE_CXX_COMPILER=${process.env['CXX'] ?? 'clang++'}`] });
      await command({ name: `${name}-test-compile`, args: ['cmake', '--build', test, '--parallel', String(jobs)] });
      executables[name] = join(test, 'qsa-model');
      status.libraries[name] = await fileIdentity({ path: join(build, 'llama/src/libllama.a') });
    }
    assert.notDeepEqual(status.libraries['reference'], status.libraries['optimized'], 'Reference and optimized libraries unexpectedly have the same identity');
    const reference = executables['reference'];
    const optimized = executables['optimized'];
    assert.ok(reference !== undefined && optimized !== undefined);
    for (const quant of ['F16', 'Q3_K', 'Q4_K']) {
      const fixture = join(results, `${quant}.gguf`);
      await command({ name: `generate-${quant}`, args: [reference, 'generate', fixture, quant] });
      status.fixtures[quant] = await fileIdentity({ path: fixture });
      for (const maskMode of ['fused', 'unfused'] as const) {
        const trace = maskMode === 'fused' ? 'trace' : 'trace-unfused';
        const traceRoot = maskMode === 'fused' ? join(results, quant) : join(results, quant, 'unfused');
        for (const [name, executable] of [['reference', reference], ['optimized', optimized]]) {
          assert.ok(name !== undefined && executable !== undefined);
          await command({ name: `trace-${quant}-${maskMode}-${name}`, args: [executable, trace, fixture, join(traceRoot, name)] });
        }
        const comparison = await compareQsaTraces({ reference: join(traceRoot, 'reference'), optimized: join(traceRoot, 'optimized'), maskMode });
        await writeFile(join(logs, `compare-${quant}-${maskMode}.json`), `${JSON.stringify(comparison, undefined, 2)}\n`);
      }
      for (const name of ['reference', 'optimized']) {
        const fused = await fileIdentity({ path: join(results, quant, name, 'logits.f32') });
        assert.deepEqual(fused, await fileIdentity({ path: join(results, quant, 'unfused', name, 'logits.f32') }), `Fused/unfused outputs differ: ${quant}/${name}`);
        status.maskParity.push({ quant, mode: name, ...fused });
      }
    }
    const flashRoot = join(results, 'F16/flash');
    for (const maskMode of ['fused', 'unfused'] as const) {
      const trace = maskMode === 'fused' ? 'trace-flash' : 'trace-flash-unfused';
      for (const [name, executable] of [['reference', reference], ['optimized', optimized]]) {
        assert.ok(name !== undefined && executable !== undefined);
        await command({ name: `trace-F16-flash-${maskMode}-${name}`, args: [executable, trace, join(results, 'F16.gguf'), join(flashRoot, maskMode, name)] });
      }
      const comparison = await compareQsaTraces({ reference: join(flashRoot, maskMode, 'reference'), optimized: join(flashRoot, maskMode, 'optimized'), maskMode });
      await writeFile(join(logs, `compare-F16-flash-${maskMode}.json`), `${JSON.stringify(comparison, undefined, 2)}\n`);
    }
    for (const name of ['reference', 'optimized']) {
      const fused = await fileIdentity({ path: join(flashRoot, 'fused', name, 'logits.f32') });
      assert.deepEqual(fused, await fileIdentity({ path: join(flashRoot, 'unfused', name, 'logits.f32') }), `Flash F16 fused/unfused outputs differ: ${name}`);
      status.flashMaskParity.push({ mode: name, maskType: 'F16', ...fused });
    }
    status.success = true;
  } finally {
    await save();
  }
}

export async function runQsaCli({ argv }: { argv: readonly string[] }): Promise<void> {
  await yargs([...argv]).strict().help().exitProcess(false).parserConfiguration({ 'camel-case-expansion': false })
    .option('checkout-root', { type: 'string', demandOption: true })
    .option('tooling-root', { type: 'string', demandOption: true })
    .option('output-root', { type: 'string', demandOption: true })
    .option('toolchain-root', { type: 'string', demandOption: true })
    .option('expected-commit', { type: 'string' })
    .option('local', { type: 'boolean', default: false })
    .option('jobs', { type: 'number', default: 2 })
    .command('$0', 'Run the independent native QSA parity fixture', builder => builder, async args => {
      if (args.local === (args['expected-commit'] !== undefined)) {
        throw new Error('Select exactly one of --local or --expected-commit');
      }
      const mode: SourceMode = args.local ? { kind: 'local' } : { kind: 'release', expectedCommit: args['expected-commit'] ?? '' };
      const context = { checkoutRoot: resolve(args['checkout-root']), toolingRoot: resolve(args['tooling-root']), outputRoot: resolve(args['output-root']), toolchainRoot: resolve(args['toolchain-root']) };
      const metrics = createMetrics({ env: process.env, checkoutRoot: context.checkoutRoot });
      await metrics.observeEnvironment({ observation: environmentObservation({ env: process.env, workRoot: context.checkoutRoot }) });
      await metrics.measure({ phase: 'qsa.native', packageKind: undefined, operation: async () => runQsaRegression({ context, mode, jobs: args.jobs }) });
    }).parseAsync();
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await runQsaCli({ argv: process.argv.slice(2) });
}
