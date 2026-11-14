import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import type { SourceMode } from '#tools/source/capture';
import { cachePlanOptions, runCachePlan, writeCachePlan } from '#tools/toolchain/cli-options';
import { createMetrics, environmentObservation } from '#tools/ci/metrics';
import { buildRuntime, prepareBuildInputs } from './build.js';
import { loadConfiguration, selectConfiguration } from './config.js';

const outputRootOption = { type: 'string', demandOption: true, description: 'Private compiler inputs and build outputs' } as const;
const sourceOptions = { 'expected-commit': { type: 'string', description: 'Require this release checkout HEAD and clean inputs' }, local: { type: 'boolean', default: false, description: 'Use local inputs; never stage a publishable runtime' } } as const;

function sourceMode({ local, expectedCommit }: { local: boolean; expectedCommit: string | undefined }): SourceMode {
  if (local === (expectedCommit !== undefined)) {
    throw new Error('Select --expected-commit or --local');
  }
  if (expectedCommit !== undefined) {
    return { kind: 'release', expectedCommit };
  }
  return { kind: 'local' };
}

export async function main({ args }: { args: readonly string[] }): Promise<void> {
  await yargs(args)
    .scriptName('sd-runtime')
    .exitProcess(false)
    .strict()
    .help()
    .options({
      'checkout-root': { type: 'string', demandOption: true, description: 'Checkout owning this emitted tooling and static configuration' },
    })
    .command('prepare', 'Prepare coherent private compiler inputs and upstream sources', (command) => command.options({ ...sourceOptions, 'output-root': outputRootOption }), async (argv) => {
      const checkoutRoot = path.resolve(argv['checkout-root']);
      const outputRoot = path.resolve(argv['output-root']);
      const configuration = await loadConfiguration({ checkoutRoot });
      const { prepared } = await prepareBuildInputs({
        checkoutRoot,
        snapshotRoot: path.join(outputRoot, 'source-inputs'),
        preparedRoot: path.join(outputRoot, 'prepared'),
        mode: sourceMode({ local: argv.local, expectedCommit: argv['expected-commit'] }),
        declarations: configuration.upstreams,
      });
      writeFileSync(process.stdout.fd, `${JSON.stringify(prepared.facts.receipt)}\n`);
    })
    .command('cache-plan', 'Plan isolated runtime compiler caches', (command) => command.options(cachePlanOptions), async (argv) => {
      const checkoutRoot = path.resolve(argv['checkout-root']);
      const configuration = await loadConfiguration({ checkoutRoot });
      const selected = selectConfiguration({ configuration, profile: argv.profile, variant: argv.variant });
      const value = await runCachePlan({ runtime: 'stable-diffusion-cpp', checkoutRoot, workRoot: argv['work-root'], profile: argv.profile, variant: argv.variant, profileConfiguration: selected.profile, variantConfiguration: selected.variant, sourceCommit: argv['source-commit'], ccacheVersion: argv['ccache-version'], environment: argv.environment, os: argv.os, arch: argv.arch });
      await writeCachePlan({ value, output: argv.output });
    })
    .command('build', 'Build and stage one profile from this checkout', (command) => command.options({
      ...sourceOptions,
      'output-root': outputRootOption,
      'tooling-root': { type: 'string', demandOption: true, description: 'Normal emit directory belonging to this checkout' },
      'toolchain-root': { type: 'string', demandOption: true, description: 'Acquired pinned Emscripten and Dawn inputs' },
      profile: { type: 'string', demandOption: true },
      variant: { type: 'string', demandOption: true },
      jobs: { type: 'number', default: 4 },
      fresh: { type: 'boolean', default: false },
      kind: { choices: ['wasm', 'native'] as const, default: 'wasm' as const },
      'include-toolchain-notices': { type: 'boolean', default: false },
    }), async (argv) => {
      const context = { checkoutRoot: path.resolve(argv['checkout-root']), toolingRoot: path.resolve(argv['tooling-root']), outputRoot: path.resolve(argv['output-root']), toolchainRoot: path.resolve(argv['toolchain-root']) };
      const metrics = createMetrics({ env: process.env, checkoutRoot: context.checkoutRoot });
      await metrics.observeEnvironment({ observation: environmentObservation({ env: process.env, workRoot: context.checkoutRoot }) });
      const result = await metrics.measure({
        phase: 'build.total',
        packageKind: undefined,
        operation: async () => buildRuntime({
          context,
          selection: { profile: argv.profile, variant: argv.variant, kind: argv.kind, jobs: argv.jobs, fresh: argv.fresh, includeToolchainNotices: argv['include-toolchain-notices'] },
          mode: sourceMode({ local: argv.local, expectedCommit: argv['expected-commit'] }),
        }),
      });
      writeFileSync(process.stdout.fd, `${JSON.stringify(result)}\n`);
    })
    .demandCommand(1)
    .parseAsync();
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main({ args: process.argv.slice(2) }).catch((error: unknown) => {
    writeFileSync(process.stderr.fd, `${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
