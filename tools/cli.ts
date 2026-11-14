import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import yargs from 'yargs';
import type { RuntimeName } from '#dto/shared';
import { runtimeNames } from '#tools/shared/contracts';
import { ProcessError, runProcess } from '#tools/shared/process';
import { READ_ENV_BIC_METRICS_MODE, READ_ENV_GITHUB_JOB, READ_ENV_GITHUB_RUN_ATTEMPT, READ_ENV_GITHUB_RUN_ID, READ_ENV_LCB_SOURCE_COMMIT, READ_ENV_SDCB_TEST_WEBGPU } from '#tools/shared/node-env';
import type { PackageKind } from '#tools/ci/metrics';
import { cachePlanOptions, requiredString } from '#tools/toolchain/cli-options';

const executingCheckoutRoot = dirname(dirname(fileURLToPath(import.meta.resolve('#source/toolchain.json'))));

function runtimePackageKind({ runtime }: { runtime: RuntimeName | undefined }): PackageKind {
  switch (runtime) {
  case 'llama-cpp':
    return 'llama';
  case 'stable-diffusion-cpp':
    return 'image';
  case undefined:
    return 'root';
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
}

async function jsonOutput({ value, output }: { value: unknown; output: string | undefined }): Promise<void> {
  const text = JSON.stringify(value, undefined, 2) + '\n';
  if (output !== undefined) {
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, text);
  }
  process.stdout.write(text);
}

async function runtimeConfiguration({ runtime, checkoutRoot, profile, variant }: { runtime: RuntimeName; checkoutRoot: string; profile: string; variant: string }) {
  switch (runtime) {
  case 'llama-cpp': {
    const module = await import('#runtimes/llama-cpp/tools/config');
    return module.selectConfiguration({ configuration: await module.loadConfiguration({ checkoutRoot }), profile, variant });
  }
  case 'stable-diffusion-cpp': {
    const module = await import('#runtimes/stable-diffusion-cpp/tools/config');
    return module.selectConfiguration({ configuration: await module.loadConfiguration({ checkoutRoot }), profile, variant });
  }
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
}

/** Module loading is the dispatch boundary; no selected command means no tool runs. */
export type CliModules = {
  toolchainPrepare: () => Promise<Pick<typeof import('#tools/toolchain/prepare'), 'prepareToolchain'>>;
  cachePlanCli: () => Promise<Pick<typeof import('#tools/toolchain/cli-options'), 'runCachePlan'>>;
  runtimeConfiguration: typeof runtimeConfiguration;
  runtimePackage: ({ runtime }: { runtime: RuntimeName }) => Promise<Pick<typeof import('#runtimes/llama-cpp/tools/package'), 'buildPackage'> | Pick<typeof import('#runtimes/stable-diffusion-cpp/tools/package'), 'buildPackage'>>;
  runtimeVerify: ({ runtime }: { runtime: RuntimeName }) => Promise<Pick<typeof import('#runtimes/llama-cpp/tools/verify'), 'verifyRuntimePackage'> | Pick<typeof import('#runtimes/stable-diffusion-cpp/tools/verify'), 'verifyRuntimePackage'>>;
  aggregate: () => Promise<Pick<typeof import('#tools/package/aggregate'), 'assemblePackage' | 'validatePackage'>>;
  publication: () => Promise<Pick<typeof import('#tools/publication/artifacts'), 'freezePackage' | 'confirmPublication'>>;
  publicationSources: () => Promise<Pick<typeof import('#tools/publication/source'), 'verifyPublicationSources'>>;
  npmResolution: () => Promise<Pick<typeof import('#tools/publication/npm-lock'), 'prepareNpmResolution' | 'readNpmResolution'>>;
  report: () => Promise<Pick<typeof import('#tools/publication/consumer'), 'writeConfirmedReport'>>;
  metrics: () => Promise<Pick<typeof import('#tools/ci/metrics'), 'createMetrics'>>;
  validation: () => Promise<Pick<typeof import('#tools/validation/runtime'), 'validateSavedRuntimePackage'>>;
};

const defaultModules: CliModules = {
  toolchainPrepare: () => import('#tools/toolchain/prepare'),
  cachePlanCli: () => import('#tools/toolchain/cli-options'),
  runtimeConfiguration,
  runtimePackage: async ({ runtime }) => {
    switch (runtime) {
    case 'llama-cpp':
      return import('#runtimes/llama-cpp/tools/package');
    case 'stable-diffusion-cpp':
      return import('#runtimes/stable-diffusion-cpp/tools/package');
    default: {
      const exhaustive: never = runtime;
      throw new Error(String(exhaustive));
    }
    }
  },
  runtimeVerify: async ({ runtime }) => {
    switch (runtime) {
    case 'llama-cpp':
      return import('#runtimes/llama-cpp/tools/verify');
    case 'stable-diffusion-cpp':
      return import('#runtimes/stable-diffusion-cpp/tools/verify');
    default: {
      const exhaustive: never = runtime;
      throw new Error(String(exhaustive));
    }
    }
  },
  aggregate: () => import('#tools/package/aggregate'),
  publication: () => import('#tools/publication/artifacts'),
  publicationSources: () => import('#tools/publication/source'),
  npmResolution: () => import('#tools/publication/npm-lock'),
  report: () => import('#tools/publication/consumer'),
  metrics: () => import('#tools/ci/metrics'),
  validation: () => import('#tools/validation/runtime'),
};
const checkoutRootOption = { ...requiredString, description: 'Checkout that owns the executing tools and configuration' } as const;
const runtimeOption = { type: 'string', choices: runtimeNames, demandOption: true } as const;
const booleanOption = { type: 'boolean', default: false } as const;

/** Ordinary yargs builders infer each handler's options without a second schema. */
export function createCli({ modules, writeJson }: { modules: CliModules | undefined; writeJson: typeof jsonOutput | undefined }) {
  const tools = modules ?? defaultModules;
  const emit = writeJson ?? jsonOutput;
  let metrics: Promise<ReturnType<typeof import('#tools/ci/metrics').createMetrics> | undefined> | undefined;
  async function measured<T>({ phase, packageKind, operation }: { phase: string; packageKind: PackageKind | undefined; operation: () => Promise<T> }): Promise<T> {
    if (READ_ENV_BIC_METRICS_MODE() !== 'basic') {
      return operation();
    }
    metrics ??= Promise.resolve().then(() => tools.metrics()).then(({ createMetrics }) => createMetrics({ checkoutRoot: executingCheckoutRoot })).catch(() => {
      try {
        process.stderr.write('[pipeline-metrics] Optional telemetry unavailable.\n');
      } catch { /* Optional output. */ }
      return undefined;
    });
    const observer = await metrics;
    return observer ? observer.measure({ phase, packageKind, operation }) : operation();
  }
  return yargs()
    .scriptName('bicore')
    .exitProcess(false)
    .version(false)
    .help()
    .strict()
    .demandCommand(1)
    .showHelpOnFail(false)
    .parserConfiguration({ 'camel-case-expansion': false, 'parse-numbers': false, 'greedy-arrays': false })
    .check(options => {
      if (Array.isArray(options.runtime)) {
        throw new Error('--runtime accepts one value');
      }
      return true;
    })
    .fail((message, error) => {
      throw error ?? new Error(message);
    })
    .epilog('Acquisition, upload, and Git remote operations belong in workflow YAML.')
    .command('toolchain', 'Prepare local tools and plan caches', command => command
      .demandCommand(1)
      .command('prepare', 'Prepare the checked-out toolchain', command => command
        .option('checkout-root', checkoutRootOption)
        .option('tools-root', requiredString)
        .option('dawn-only', booleanOption), async options => measured({
        phase: 'cli.toolchain.prepare',
        packageKind: undefined,
        operation: async () => {
          const checkoutRoot = resolve(options['checkout-root']);
          const toolsRoot = resolve(options['tools-root']);
          const { prepareToolchain } = await tools.toolchainPrepare();
          await prepareToolchain({ checkoutRoot, toolsRoot, dawnOnly: options['dawn-only'] });
          await emit({ value: { toolsRoot, status: 'complete' }, output: undefined });
        },
      }))
      .command('cache-plan', 'Plan and configure an offline build cache', command => command
        .option('runtime', runtimeOption)
        .options(cachePlanOptions), async options => measured({
        phase: 'cli.toolchain.cache-plan',
        packageKind: undefined,
        operation: async () => {
          const checkoutRoot = resolve(options['checkout-root']);
          const workRoot = resolve(options['work-root']);
          const configuration = await tools.runtimeConfiguration({ runtime: options.runtime, checkoutRoot, profile: options.profile, variant: options.variant });
          const { runCachePlan } = await tools.cachePlanCli();
          const plan = await runCachePlan({ runtime: options.runtime, checkoutRoot, workRoot, profile: options.profile, variant: options.variant, profileConfiguration: configuration.profile, variantConfiguration: configuration.variant, sourceCommit: options['source-commit'], ccacheVersion: options['ccache-version'], environment: options.environment, os: options.os, arch: options.arch });
          await emit({ value: plan, output: resolve(options.output) });
        },
      })), () => {})
    .command('package', 'Assemble and verify local runtime packages', command => command
      .demandCommand(1)
      .command('runtime', 'Package one runtime', command => command
        .option('runtime', runtimeOption)
        .option('checkout-root', checkoutRootOption)
        .option('build-root', requiredString)
        .option('output', requiredString)
        .option('license-root', { type: 'string', demandOption: true, requiresArg: true, array: true, nargs: 1 })
        .option('profile', { type: 'string', array: true, requiresArg: true, nargs: 1 })
        .option('defer-npm-pack', booleanOption), async options => measured({
        phase: 'cli.package.runtime',
        packageKind: runtimePackageKind({ runtime: options.runtime }),
        operation: async () => {
          const module = await tools.runtimePackage({ runtime: options.runtime });
          const destination = resolve(options.output);
          const result = await module.buildPackage({ sourceRoot: resolve(options['checkout-root']), buildRoot: resolve(options['build-root']), destination, profiles: options.profile, licenseRoots: options['license-root'].map(value => resolve(value)), checkNpmPack: !options['defer-npm-pack'] });
          await emit({ value: { runtime: options.runtime, sourceCommit: result.manifest.sourceCommit, output: destination }, output: undefined });
        },
      }))
      .command('assemble', 'Assemble both runtime packages', command => command
        .option('checkout-root', checkoutRootOption)
        .option('inputs', requiredString)
        .option('output', requiredString)
        .option('defer-npm-pack', booleanOption), async options => measured({
        phase: 'cli.package.assemble',
        packageKind: 'root',
        operation: async () => {
          const { assemblePackage } = await tools.aggregate();
          const destination = resolve(options.output);
          const result = await assemblePackage({ sourceRoot: resolve(options['checkout-root']), inputs: resolve(options.inputs), destination, checkNpmPack: !options['defer-npm-pack'] });
          await emit({ value: { sourceCommit: result.manifest.sourceCommit, output: destination }, output: undefined });
        },
      }))
      .command('verify', 'Verify one runtime or the aggregate', command => command
        .option('checkout-root', checkoutRootOption)
        .option('directory', requiredString)
        .option('runtime', { type: 'string', choices: runtimeNames }), async options => measured({
        phase: 'cli.package.verify',
        packageKind: runtimePackageKind({ runtime: options.runtime }),
        operation: async () => {
          const sourceRoot = resolve(options['checkout-root']);
          const directory = resolve(options.directory);
          if (options.runtime === undefined) {
            const { validatePackage } = await tools.aggregate();
            const result = await validatePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: true });
            await emit({ value: { sourceCommit: result.manifest.sourceCommit, directory, status: 'complete' }, output: undefined });
          } else {
            const module = await tools.runtimeVerify({ runtime: options.runtime });
            const result = await module.verifyRuntimePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: true });
            await emit({ value: { runtime: options.runtime, sourceCommit: result.manifest.sourceCommit, directory, status: 'complete' }, output: undefined });
          }
        },
      })), () => {})
    .command('publication', 'Freeze, confirm, and report local publication state', command => command
      .demandCommand(1)
      .command('freeze', 'Freeze a verified package and its source facts', command => command
        .option('checkout-root', checkoutRootOption)
        .option('directory', requiredString)
        .option('work-root', { ...requiredString, description: 'Fresh artifact snapshot directory' })
        .option('receipt', requiredString)
        .option('branch', requiredString)
        .option('source-commit', requiredString)
        .option('source-work-root', { ...requiredString, description: 'Fresh compiler snapshot and upstream preparation directory' })
        .option('python', { ...requiredString, description: 'Python executable for the upstream WGSL generator' }), async options => measured({
        phase: 'cli.publication.freeze',
        packageKind: 'root',
        operation: async () => {
          const sourceRoot = resolve(options['checkout-root']);
          const { freezePackage } = await tools.publication();
          const { verifyPublicationSources } = await tools.publicationSources();
          const frozen = await freezePackage({ sourceRoot, packageDirectory: resolve(options.directory), work: resolve(options['work-root']), receipt: resolve(options.receipt), branch: options.branch, expectedSource: options['source-commit'], verifySource: ({ sourceCommit, manifests }) => verifyPublicationSources({ sourceRoot, workRoot: resolve(options['source-work-root']), sourceCommit, manifests, python: options.python }) });
          await emit({ value: frozen, output: undefined });
        },
      }))
      .command('confirm', 'Confirm the published commit against this job\'s frozen receipt', command => command
        .option('work-root', requiredString)
        .option('receipt', requiredString)
        .option('commit', requiredString), async options => measured({
        phase: 'cli.publication.confirm',
        packageKind: 'root',
        operation: async () => {
          const { confirmPublication } = await tools.publication();
          const publication = await confirmPublication({ work: resolve(options['work-root']), receipt: resolve(options.receipt), commit: options.commit });
          await emit({ value: { commit: publication.commit, sourceCommit: publication.frozen.sourceCommit, tree: publication.frozen.tree }, output: undefined });
        },
      }))
      .command('npm-fixture', 'Prepare local files for the workflow-owned npm resolution', command => command
        .option('directory', requiredString)
        .option('repository', requiredString)
        .option('commit', requiredString)
        .option('version', requiredString), async options => measured({
        phase: 'cli.publication.npm-fixture',
        packageKind: 'root',
        operation: async () => {
          const { prepareNpmResolution } = await tools.npmResolution();
          await emit({ value: await prepareNpmResolution({ directory: resolve(options.directory), repository: options.repository, commit: options.commit, version: options.version }), output: undefined });
        },
      }))
      .command('report', 'Write a report from the confirmed frozen state', command => command
        .option('work-root', requiredString)
        .option('receipt', requiredString)
        .option('commit', requiredString)
        .option('repository', requiredString)
        .option('npm-fixture', requiredString)
        .option('output', requiredString)
        .option('summary', { type: 'string', requiresArg: true })
        .option('run-id', { type: 'number', requiresArg: true })
        .option('run-attempt', { type: 'number', requiresArg: true })
        .implies('run-id', 'run-attempt')
        .implies('run-attempt', 'run-id')
        .check(options => {
          for (const value of [options['run-id'], options['run-attempt']]) {
            if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
              throw new Error('--run-id and --run-attempt require positive safe integers');
            }
          }
          return true;
        }), async options => measured({
        phase: 'cli.publication.report',
        packageKind: 'root',
        operation: async () => {
          const { confirmPublication } = await tools.publication();
          const { readNpmResolution } = await tools.npmResolution();
          const { writeConfirmedReport } = await tools.report();
          const publication = await confirmPublication({ work: resolve(options['work-root']), receipt: resolve(options.receipt), commit: options.commit });
          const resolution = await readNpmResolution({ directory: resolve(options['npm-fixture']), expected: { repository: options.repository, commit: options.commit, version: publication.frozen.report.data.package.version } });
          const output = resolve(options.output);
          const run = options['run-id'] === undefined || options['run-attempt'] === undefined ? undefined : { id: options['run-id'], attempt: options['run-attempt'] };
          await writeConfirmedReport({ publication, repository: options.repository, resolution, output, run, summary: options.summary === undefined ? undefined : resolve(options.summary) });
          await emit({ value: { output, artifactCommit: publication.commit, sourceCommit: publication.frozen.sourceCommit }, output: undefined });
        },
      })), () => {})
    .command('validation', 'Validate saved runtime packages', command => command
      .demandCommand(1)
      .command('runtime', 'Validate a saved package with its browser fixture', command => command
        .option('runtime', runtimeOption)
        .option('checkout-root', checkoutRootOption)
        .option('package', requiredString)
        .option('receipt', requiredString), async options => measured({
        phase: 'cli.validation.runtime',
        packageKind: runtimePackageKind({ runtime: options.runtime }),
        operation: async () => {
          const { validateSavedRuntimePackage } = await tools.validation();
          const receipt = await validateSavedRuntimePackage({ checkoutRoot: resolve(options['checkout-root']), runtime: options.runtime, packageDirectory: resolve(options.package), receiptPath: resolve(options.receipt), settings: { selectedSourceCommit: READ_ENV_LCB_SOURCE_COMMIT(), runId: READ_ENV_GITHUB_RUN_ID(), runAttempt: READ_ENV_GITHUB_RUN_ATTEMPT(), job: READ_ENV_GITHUB_JOB(), testWebGpu: READ_ENV_SDCB_TEST_WEBGPU() }, childEnvironment: undefined, runProcess });
          await emit({ value: receipt, output: undefined });
        },
      })), () => {});
}

export async function main({ argv, modules, writeJson, writeText }: { argv: readonly string[]; modules: CliModules | undefined; writeJson: typeof jsonOutput | undefined; writeText: (({ stream, text }: { stream: 'stdout' | 'stderr'; text: string }) => void) | undefined }): Promise<number> {
  const write = writeText ?? (({ stream, text }: { stream: 'stdout' | 'stderr'; text: string }) => {
    switch (stream) {
    case 'stdout': process.stdout.write(text);
      break;
    case 'stderr': process.stderr.write(text);
      break;
    default: {
      const exhaustive: never = stream;
      throw new Error(`Unhandled output stream: ${String(exhaustive)}`);
    }
    }
  });
  try {
    await createCli({ modules, writeJson }).parseAsync(argv, {}, (error, _options, output) => {
      if (error) {
        throw error;
      }
      if (output) {
        write({ stream: 'stdout', text: output + '\n' });
      }
    });
    return 0;
  } catch (error) {
    write({ stream: 'stderr', text: `${error instanceof Error ? error.message : String(error)}\n` });
    return error instanceof ProcessError && error.exitCode !== null && error.exitCode > 0 ? error.exitCode : 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = await main({ argv: process.argv.slice(2), modules: undefined, writeJson: undefined, writeText: undefined });
}
