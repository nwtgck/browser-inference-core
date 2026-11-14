import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import yargs from 'yargs';
import { createMetrics, environmentObservation } from '#tools/ci/metrics';
import type { SourceMode } from '#tools/source/capture';
import { cachePlanOptions, runCachePlan, writeCachePlan } from '#tools/toolchain/cli-options';
import { loadConfiguration, selectConfiguration } from '#runtimes/llama-cpp/tools/config';
import { buildRuntime } from '#runtimes/llama-cpp/tools/build';
import { prepareSources } from '#runtimes/llama-cpp/tools/prepare';

function enabled({ value }: { value: string }): boolean {
  return value === 'ON' || value === 'true' || value === '1';
}

function sourceMode({ local, expectedCommit }: { local: boolean; expectedCommit: string | undefined }): SourceMode {
  if (local && expectedCommit === undefined) {
    return { kind: 'local' };
  }
  if (!local && expectedCommit !== undefined) {
    return { kind: 'release', expectedCommit };
  }
  throw new Error('Select exactly one of --local or --expected-commit');
}

export async function main({ argv }: { argv: readonly string[] }): Promise<void> {
  const booleanStrings = ['ON', 'OFF', 'true', 'false', '1', '0', ''] as const;
  await yargs([...argv]).scriptName('llama-cpp').strict().demandCommand(1).help().exitProcess(false)
    .parserConfiguration({ 'camel-case-expansion': false })
    .command('cache-plan', 'Plan caches for this runtime', parser => parser.options(cachePlanOptions), async args => {
      const checkoutRoot = resolve(args['checkout-root']);
      const configuration = await loadConfiguration({ checkoutRoot });
      const selected = selectConfiguration({ configuration, profile: args.profile, variant: args.variant });
      const plan = await runCachePlan({ runtime: 'llama-cpp', checkoutRoot, workRoot: resolve(args['work-root']), profile: args.profile, variant: args.variant, profileConfiguration: selected.profile, variantConfiguration: selected.variant, sourceCommit: args['source-commit'], ccacheVersion: args['ccache-version'], environment: args.environment, os: args.os, arch: args.arch });
      await writeCachePlan({ value: plan, output: args.output });
    })
    .command('prepare', 'Prepare coherent compiler sources for CMake or native fixtures', parser => parser
      .option('compile-root', { type: 'string', demandOption: true })
      .option('upstream', { type: 'string', demandOption: true })
      .option('output', { type: 'string', demandOption: true })
      .option('target', { choices: ['wasm', 'native'] as const, demandOption: true })
      .option('python', { type: 'string', default: 'python3' })
      .option('webgpu', { choices: booleanStrings, default: 'OFF' as const })
      .option('gpu-profile', { choices: booleanStrings, default: 'OFF' as const })
      .option('file-rows', { choices: booleanStrings, default: 'OFF' as const })
      .option('qsa', { choices: booleanStrings, default: 'ON' as const })
      .option('bf16-projector', { choices: booleanStrings, default: 'ON' as const }), async args => {
      const webgpu = enabled({ value: args.webgpu });
      const prepared = await prepareSources({
        inputs: { compileRoot: resolve(args['compile-root']), upstreamRoots: { llama: resolve(args.upstream) } },
        output: resolve(args.output),
        python: args.python,
        options: { target: args.target, webgpu, gpuProfile: enabled({ value: args['gpu-profile'] }), fileRows: enabled({ value: args['file-rows'] }), qsa: enabled({ value: args.qsa }), mtmdMode: webgpu && enabled({ value: args['bf16-projector'] }) ? 'webgpu' : 'cpu' },
      });
      process.stdout.write(`${prepared.root}\n`);
    })
    .command('build', 'Build and stage one profile from this checkout', parser => parser
      .option('checkout-root', { type: 'string', demandOption: true })
      .option('tooling-root', { type: 'string', demandOption: true })
      .option('output-root', { type: 'string', demandOption: true })
      .option('toolchain-root', { type: 'string', demandOption: true })
      .option('expected-commit', { type: 'string' })
      .option('local', { type: 'boolean', default: false })
      .option('profile', { type: 'string', demandOption: true })
      .option('variant', { type: 'string', default: 'browser' })
      .option('kind', { choices: ['wasm', 'native'] as const, default: 'wasm' as const })
      .option('jobs', { type: 'number', default: 4 })
      .option('fresh', { type: 'boolean', default: false })
      .option('configure-only', { type: 'boolean', default: false })
      .option('include-toolchain-notices', { type: 'boolean', default: false })
      .option('python', { type: 'string', default: 'python3' }), async args => {
      const context = { checkoutRoot: resolve(args['checkout-root']), toolingRoot: resolve(args['tooling-root']), outputRoot: resolve(args['output-root']), toolchainRoot: resolve(args['toolchain-root']) };
      const metrics = createMetrics({ env: process.env, checkoutRoot: context.checkoutRoot });
      await metrics.observeEnvironment({ observation: environmentObservation({ env: process.env, workRoot: context.checkoutRoot }) });
      const result = await metrics.measure({
        phase: 'build.total',
        packageKind: undefined,
        operation: async () => buildRuntime({
          context,
          mode: sourceMode({ local: args.local, expectedCommit: args['expected-commit'] }),
          selection: { profile: args.profile, variant: args.variant, kind: args.kind, jobs: args.jobs, fresh: args.fresh, includeToolchainNotices: args['include-toolchain-notices'], configureOnly: args['configure-only'], python: args.python },
        }),
      });
      switch (result.kind) {
      case 'configured': process.stdout.write(`${result.compilationDatabase}\n`);
        break;
      case 'built': process.stdout.write(`${result.runtimeDirectory}\n`);
        break;
      default: {
        const _ex: never = result;
        throw new Error(`Unknown build result: ${String(_ex)}`);
      }
      }
    }).parseAsync();
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main({ argv: process.argv.slice(2) });
}
