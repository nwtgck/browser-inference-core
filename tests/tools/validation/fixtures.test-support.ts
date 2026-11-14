import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { BrowserSession, RuntimeName } from '#dto/shared';
import { decodeLlamaBrowserEnvelope, decodeLlamaManifest, type LlamaBrowserResult } from '#dto/llama-cpp';
import { decodeStableDiffusionBrowserEnvelope, decodeStableDiffusionManifest, type StableDiffusionBrowserResult } from '#dto/stable-diffusion-cpp';
import { loadConfiguration as llamaConfiguration } from '#runtimes/llama-cpp/tools/config';
import { loadConfiguration as imageConfiguration } from '#runtimes/stable-diffusion-cpp/tools/config';
import { verifyRuntimePackage as verifyLlama, verifyRuntimeInput as finishLlama } from '#runtimes/llama-cpp/tools/verify';
import { verifyRuntimePackage as verifyImage, verifyRuntimeInput as finishImage } from '#runtimes/stable-diffusion-cpp/tools/verify';
import { imageSmokeScope } from '#runtimes/stable-diffusion-cpp/tools/browser-smoke-contract';
import { parseJson } from '#tools/shared/json';
import type { runProcess } from '#tools/shared/process';
import { runtimeTestInputs, testFilesForRuntime } from '#tools/validation/runtime';
import { clearValidationReceipt, validateSnapshot, type ValidationReceipt } from '#tools/validation/snapshot';
import { manifestIo, type ManifestIo } from '#tools/validation/manifest';
import { checkedImageResults, addImageValidation } from '#tools/validation/image-results';
import { checkedLlamaResults, addLlamaValidation } from '#tools/validation/llama-results';
import { createSyntheticModel } from '#tools/validation/synthetic-model';
import { makeRuntime, fixtureSource } from '#tests/tools/package/fixtures.test-support';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

type Result = LlamaBrowserResult | StableDiffusionBrowserResult;
type Envelope = { session: BrowserSession; results: readonly Result[] };
export type ValidationFixture = {
  root: string; checkoutRoot: string; packageDirectory: string; receiptPath: string; runtime: RuntimeName;
  profiles: string[]; variants: string[]; testFiles: Record<string, string>; environment: NodeJS.ProcessEnv;
  commands: { phase: 'node_asyncify' | 'chromium_smoke'; args: readonly string[]; resultPath: string }[];
  finalCalls: number; childFailure: Error | undefined; finalFailure: Error | undefined; writeResults: boolean; resultText: string | undefined;
  resultMutator: (({ session, results }: { session: BrowserSession; results: readonly Result[] }) => unknown) | undefined;
  afterProcess: (() => Promise<void>) | undefined; afterFinal: (() => Promise<void>) | undefined;
  io: ManifestIo;
  validate(): Promise<ValidationReceipt<RuntimeName, Result>>;
  [Symbol.asyncDispose](): Promise<void>;
};

export function imageResults({ profiles, variants, gpu }: { profiles: readonly string[]; variants: readonly string[]; gpu: boolean }): StableDiffusionBrowserResult[] {
  const results: StableDiffusionBrowserResult[] = [];
  const backends = gpu ? ['CPU', 'WebGPU'] as const : ['CPU'] as const;
  for (const profile of profiles) {
    for (const variant of variants) {
      results.push({
        profile,
        variant,
        passed: true,
        scope: imageSmokeScope({ testWebGpu: gpu }),
        timestep: variant === 'test' ? backends.map((backend) => ({ backend, passed: true })) : [],
        bf16Weights: variant === 'test' ? backends.map((backend) => ({ backend, passed: true })) : [],
        ...(variant === 'test' ? { graphWalk: true, conv3dBias: backends.map((backend) => ({ backend, passed: true })), webgpuPerformance: backends.map((backend) => ({ backend, passed: true })) } : {}),
      });
    }
  }
  return results;
}

export function llamaResults({ profiles, variants }: { profiles: readonly string[]; variants: readonly string[] }): LlamaBrowserResult[] {
  const results: LlamaBrowserResult[] = [];
  for (const profile of profiles) {
    for (const variant of variants) {
      results.push({ profile, variant, passed: true, ...(profile.startsWith('cpu-') ? { syntheticModel: true } : { mockedAdapter: true, suspension: true, hostNavigatorInjection: variant === 'browser', defaultNavigatorFallback: true, independentFactories: variant === 'browser' }) });
    }
  }
  return results;
}

/** Real package/verification/npm; the browser process alone is a controlled double. */
export async function validationFixture({ runtime }: { runtime: RuntimeName }): Promise<ValidationFixture> {
  const temporary = await temporaryDirectory();
  try {
    const sourceRoot = process.cwd();
    await makeRuntime({ root: temporary.root, runtime, selection: 'configured' });
    const checkoutRoot = path.join(temporary.root, 'harness');
    const configuration = await (runtime === 'llama-cpp' ? llamaConfiguration : imageConfiguration)({ checkoutRoot: sourceRoot });
    for (const name of [...runtimeTestInputs[runtime], `runtimes/${runtime}/config/profiles.json`, `runtimes/${runtime}/config/variants.json`]) {
      const target = path.join(checkoutRoot, name);
      await mkdir(path.dirname(target), { recursive: true });
      await copyFile(path.join(sourceRoot, name), target);
    }
    await mkdir(path.join(checkoutRoot, 'runtimes', runtime), { recursive: true });
    const state: ValidationFixture = {
      root: temporary.root,
      checkoutRoot,
      runtime,
      packageDirectory: path.join(temporary.root, 'package-inputs', runtime),
      receiptPath: path.join(temporary.root, 'receipt.json'),
      profiles: Object.keys(configuration.profiles),
      variants: Object.keys(configuration.variants),
      testFiles: testFilesForRuntime({ checkoutRoot, runtime }),
      environment: { ...process.env, LCB_SOURCE_COMMIT: fixtureSource, GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '2', GITHUB_JOB: 'fixture', SDCB_TEST_WEBGPU: '0' },
      commands: [],
      finalCalls: 0,
      childFailure: undefined,
      finalFailure: undefined,
      writeResults: true,
      resultText: undefined,
      resultMutator: undefined,
      afterProcess: undefined,
      afterFinal: undefined,
      io: manifestIo,
      async validate() {
        state.commands.length = 0;
        state.finalCalls = 0;
        await clearValidationReceipt({ packageDirectory: state.packageDirectory, receiptPath: state.receiptPath });
        const run: typeof runProcess = async ({ command, args, cwd, env }) => {
          if (state.childFailure !== undefined) {
            throw state.childFailure;
          }
          if (command !== process.execPath || cwd !== path.join(checkoutRoot, 'runtimes', runtime) || env === undefined) {
            throw new Error('Wrong smoke invocation');
          }
          const browser = args[0]?.endsWith('/browser-smoke.mjs') === true;
          if (!browser && !args[0]?.endsWith('/asyncify-rewind.mjs')) {
            throw new Error('Unexpected smoke process');
          }
          if (env.BIC_BROWSER_RESULTS_FILE === undefined) {
            throw new Error('Missing fresh result path');
          }
          state.commands.push({ phase: browser ? 'chromium_smoke' : 'node_asyncify', args, resultPath: env.BIC_BROWSER_RESULTS_FILE });
          if (browser && state.writeResults) {
            const rawSession: unknown = JSON.parse(env.BIC_BROWSER_SESSION_JSON ?? '');
            let envelope: Envelope;
            switch (runtime) {
            case 'llama-cpp': envelope = decodeLlamaBrowserEnvelope({ input: { session: rawSession, results: llamaResults({ profiles: state.profiles, variants: state.variants }) } });
              break;
            case 'stable-diffusion-cpp': envelope = decodeStableDiffusionBrowserEnvelope({ input: { session: rawSession, results: imageResults({ profiles: state.profiles, variants: state.variants, gpu: state.environment.SDCB_TEST_WEBGPU === '1' }) } });
              break;
            default: {
              const exhaustive: never = runtime;
              throw new Error(String(exhaustive));
            }
            }
            const value = state.resultMutator === undefined ? envelope : state.resultMutator(envelope);
            await writeFile(env.BIC_BROWSER_RESULTS_FILE, state.resultText ?? JSON.stringify(value));
          }
          await state.afterProcess?.();
          return { stdout: Buffer.alloc(0), stderr: Buffer.alloc(0), exitCode: 0 };
        };
        switch (runtime) {
        case 'llama-cpp': {
          const validated = await verifyLlama({ sourceRoot, directory: state.packageDirectory, requireClean: true, checkNpmPack: false });
          return validateSnapshot({
            checkoutRoot,
            receiptPath: state.receiptPath,
            environment: state.environment,
            runProcess: run,
            io: state.io,
            plan: {
              validated,
              profiles: state.profiles,
              variants: state.variants,
              testFiles: state.testFiles,
              generateFixture: createSyntheticModel,
              nodeAsyncify: true,
              testWebGpu: false,
              decodeEnvelope: decodeLlamaBrowserEnvelope,
              checkedResults: ({ results }) => checkedLlamaResults({ profiles: state.profiles, variants: state.variants, results }),
              updateManifest: addLlamaValidation,
              finalVerify: async ({ input }) => {
                state.finalCalls++;
                if (state.finalFailure !== undefined) {
                  throw state.finalFailure;
                }
                const value = await finishLlama({ sourceRoot, input, requireClean: true, checkNpmPack: true });
                await state.afterFinal?.();
                return value;
              },
            },
          });
        }
        case 'stable-diffusion-cpp': {
          const validated = await verifyImage({ sourceRoot, directory: state.packageDirectory, requireClean: true, checkNpmPack: false });
          const gpu = state.environment.SDCB_TEST_WEBGPU === '1';
          return validateSnapshot({
            checkoutRoot,
            receiptPath: state.receiptPath,
            environment: state.environment,
            runProcess: run,
            io: state.io,
            plan: {
              validated,
              profiles: state.profiles,
              variants: state.variants,
              testFiles: state.testFiles,
              generateFixture: undefined,
              nodeAsyncify: false,
              testWebGpu: gpu,
              decodeEnvelope: decodeStableDiffusionBrowserEnvelope,
              checkedResults: ({ results }) => checkedImageResults({ profiles: state.profiles, variants: state.variants, results, testWebGpu: gpu }),
              updateManifest: addImageValidation,
              finalVerify: async ({ input }) => {
                state.finalCalls++;
                if (state.finalFailure !== undefined) {
                  throw state.finalFailure;
                }
                const value = await finishImage({ sourceRoot, input, requireClean: true, checkNpmPack: true });
                await state.afterFinal?.();
                return value;
              },
            },
          });
        }
        default: {
          const exhaustive: never = runtime;
          throw new Error(String(exhaustive));
        }
        }
      },
      async [Symbol.asyncDispose]() {
        await temporary[Symbol.asyncDispose]();
      },
    };
    return state;
  } catch (error) {
    await temporary[Symbol.asyncDispose]();
    throw error;
  }
}

export async function readFixtureManifest({ fixture }: { fixture: ValidationFixture }) {
  const text = await readFile(path.join(fixture.packageDirectory, 'manifest.json'), 'utf8');
  switch (fixture.runtime) {
  case 'llama-cpp': return parseJson({ text, decode: decodeLlamaManifest });
  case 'stable-diffusion-cpp': return parseJson({ text, decode: decodeStableDiffusionManifest });
  default: {
    const exhaustive: never = fixture.runtime;
    throw new Error(String(exhaustive));
  }
  }
}
