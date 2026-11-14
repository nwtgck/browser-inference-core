import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import { cacheObservation, createMetrics, printObservation } from '#tools/ci/metrics';
import { runObservedCommand } from '#tools/ci/command';
import { ninjaSummary, readRecords, summarize, type Summary } from '#tools/ci/summary';
import { directory as ensureDirectory, isContained } from '#tools/shared/files';
import { READ_ENV_BIC_METRICS_MODE } from '#tools/shared/node-env';

async function writeSummary({ directory, checkoutRoot, report }: { directory: string; checkoutRoot: string; report: Summary }): Promise<void> {
  const target = resolve(directory);
  if (isContained({ root: checkoutRoot, path: target })) {
    throw new Error('Metrics summary must be outside the source tree');
  }
  await ensureDirectory({ path: target, create: true });
  await using file = await open(join(target, 'summary.json'), constants.O_WRONLY | constants.O_CREAT | constants.O_NOFOLLOW | constants.O_NONBLOCK, 0o600);
  const info = await file.stat();
  if (!info.isFile() || info.nlink !== 1) {
    throw new Error('Invalid summary file');
  }
  await file.truncate(0);
  await file.writeFile(`${JSON.stringify(report, undefined, 2)}\n`);
}

function consoleSummary({ report }: { report: Summary }): void {
  const rows = Object.entries(report.phases).sort(([left], [right]) => left.localeCompare(right));
  printObservation({
    event: 'summary',
    values: {
      invalidOrMissingRecords: report.invalidOrMissingRecords,
      phases: rows.length,
      note: 'Inclusive phase times overlap; do not sum as workflow wall time.',
    },
  });
  for (const [index, [phase, value]] of rows.entries()) {
    if (index >= 100) {
      break;
    }
    printObservation({
      event: 'phase_summary',
      values: {
        phase,
        started: value.started,
        ended: value.ended,
        failed: value.failed,
        incomplete: value.incomplete,
        inclusiveSeconds: value.inclusiveNs === null ? null : Math.round(value.inclusiveNs / 1000) / 1e6,
        exclusiveSeconds: value.exclusiveNs === null ? null : Math.round(value.exclusiveNs / 1000) / 1e6,
      },
    });
  }
  if (rows.length > 100) {
    printObservation({ event: 'summary_truncated', values: { remainingPhases: rows.length - 100 } });
  }
}

export async function main({ args, checkoutRoot }: { args: readonly string[]; checkoutRoot: string }): Promise<void> {
  const metrics = createMetrics({ checkoutRoot });
  const enabled = READ_ENV_BIC_METRICS_MODE() === 'basic';
  await yargs([...args])
    .scriptName('bicore-metrics').exitProcess(false).version(false).help().strict().demandCommand(1)
    .parserConfiguration({ 'populate--': true, 'camel-case-expansion': false, 'parse-numbers': false })
    .fail((message, error) => {
      throw error ?? new Error(message);
    })
    .command('run', 'Measure an external command', builder => builder.option('phase', { type: 'string', demandOption: true }), async options => {
      const trailing: unknown = options['--'];
      if (!Array.isArray(trailing) || trailing.some((value: unknown) => typeof value !== 'string')) {
        throw new Error('Expected command arguments after --');
      }
      const commandLine: string[] = [];
      for (const value of trailing) {
        if (typeof value === 'string') {
          commandLine.push(value);
        }
      }
      const command = commandLine[0];
      if (!command) {
        throw new Error('A command after -- is required');
      }
      const outcome = await runObservedCommand({ metrics, phase: options.phase, command, args: commandLine.slice(1), cwd: process.cwd(), env: undefined });
      if (outcome.signal) {
        process.kill(process.pid, outcome.signal);
      } else {
        process.exitCode = outcome.exitCode ?? 1;
      }
    })
    .command('cache', 'Record an Actions cache observation', builder => builder
      .option('kind', { choices: ['dawn', 'em', 'cc'] as const, demandOption: true })
      .option('requested', { type: 'string', default: '' })
      .option('matched', { type: 'string', default: '' })
      .option('hit', { type: 'string', default: '' })
      .option('outcome', { type: 'string', default: '' })
      .option('save-eligible', { choices: ['true', 'false'] as const, default: 'false' as const }), async options => {
      await metrics.observeCache({
        observation: cacheObservation({
          kind: options.kind,
          requested: options.requested,
          matched: options.matched,
          hit: options.hit,
          outcome: options.outcome,
          saveEligible: options['save-eligible'] === 'true',
        }),
      });
    })
    .command('cache-save', 'Record an Actions cache save outcome', builder => builder
      .option('kind', { choices: ['dawn', 'em', 'cc'] as const, demandOption: true })
      .option('outcome', { type: 'string', default: '' }), async options => {
      await metrics.observeCacheSave({ kind: options.kind, outcome: options.outcome });
    })
    .command('environment', 'Record whether compiler caches are configured', builder => builder, async () => {
      await metrics.observeEnvironment();
    })
    .command('summary', 'Summarize optional observations', builder => builder
      .option('directory', { type: 'string' }).option('ninja', { type: 'string' }), async options => {
      if (!enabled) {
        return;
      }
      try {
        if (options.ninja) {
          await metrics.observeNinja({ observation: await ninjaSummary({ path: options.ninja }) });
        }
        if (options.directory) {
          if (isContained({ root: checkoutRoot, path: resolve(options.directory) })) {
            throw new Error('Metrics must be outside the source tree');
          }
          const { records, errors } = await readRecords({ directory: options.directory });
          const report = summarize({ records, errors });
          await writeSummary({ directory: options.directory, checkoutRoot, report });
          consoleSummary({ report });
        }
      } catch {
        process.stderr.write('[pipeline-metrics] Optional telemetry unavailable or truncated.\n');
      }
    })
    .parseAsync();
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await main({ args: process.argv.slice(2), checkoutRoot: process.cwd() });
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
