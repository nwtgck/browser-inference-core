import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import { requiredString } from '#tools/toolchain/cli-options';
import { prepareToolchain } from '#tools/toolchain/prepare';
import { READ_ENV_BIC_METRICS_MODE } from '#tools/shared/node-env';

/** Only the optional observer may fail open; the preparation error is preserved. */
async function measuredPreparation({ checkoutRoot, operation }: { checkoutRoot: string; operation: () => Promise<void> }): Promise<void> {
  if (READ_ENV_BIC_METRICS_MODE() !== 'basic') {
    return operation();
  }
  let observer: ReturnType<typeof import('#tools/ci/metrics').createMetrics>;
  try {
    const { createMetrics } = await import('#tools/ci/metrics');
    observer = createMetrics({ checkoutRoot });
  } catch {
    try {
      process.stderr.write('[pipeline-metrics] Optional telemetry unavailable.\n');
    } catch { /* Optional output. */ }
    return operation();
  }
  await observer.measure({ phase: 'cli.toolchain.prepare', packageKind: undefined, operation });
}

/** This entry has no dependency on either runtime or the aggregate CLI. */
export function createToolchainCli({ prepare }: { prepare: typeof prepareToolchain }) {
  return yargs()
    .scriptName('bicore-toolchain').exitProcess(false).version(false).help().strict().demandCommand(1).showHelpOnFail(false)
    .parserConfiguration({ 'camel-case-expansion': false, 'parse-numbers': false, 'greedy-arrays': false })
    .fail((message, error) => {
      throw error ?? new Error(message);
    })
    .command('prepare', 'Prepare an already acquired toolchain', parser => parser
      .option('checkout-root', requiredString)
      .option('tools-root', requiredString)
      .option('dawn-only', { type: 'boolean', default: false }), async options => {
      const checkoutRoot = resolve(options['checkout-root']);
      const toolsRoot = resolve(options['tools-root']);
      await measuredPreparation({
        checkoutRoot,
        operation: async () => {
          await prepare({ checkoutRoot, toolsRoot, dawnOnly: options['dawn-only'] });
          process.stdout.write(`${JSON.stringify({ toolsRoot, status: 'complete' }, undefined, 2)}\n`);
        },
      });
    });
}

export async function main({ args }: { args: readonly string[] }): Promise<void> {
  await createToolchainCli({ prepare: prepareToolchain }).parseAsync([...args]);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await main({ args: process.argv.slice(2) });
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
