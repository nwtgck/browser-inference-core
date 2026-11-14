import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RuntimeName } from '#dto/shared';
import { decodeLlamaBrowserEnvelope, type LlamaBrowserResult } from '#dto/llama-cpp';
import { decodeStableDiffusionBrowserEnvelope, type StableDiffusionBrowserResult } from '#dto/stable-diffusion-cpp';
import { createSyntheticModel } from './synthetic-model.js';
import { clearValidationReceipt, manifestIo, validateSnapshot, type ValidationReceipt } from './snapshot.js';

export type { ValidationReceipt } from './snapshot.js';

/** Include indirectly loaded harnesses: results must bind to those exact bytes. */
export const runtimeTestInputs = {
  'llama-cpp': [
    'tests/runtimes/llama_cpp/asyncify-rewind.mjs', 'tests/runtimes/llama_cpp/browser-smoke.mjs',
    'tests/runtimes/llama_cpp/chat-surface.mjs', 'tests/runtimes/llama_cpp/file-mount-lifetime.mjs',
    'runtimes/llama-cpp/vendor/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja', 'tools/validation/synthetic-model.ts',
  ],
  'stable-diffusion-cpp': [
    'tests/runtimes/stable_diffusion_cpp/browser-smoke.mjs', 'tests/runtimes/stable_diffusion_cpp/gguf-fixture.mjs',
    'tests/runtimes/stable_diffusion_cpp/model-io-fixtures.mjs', 'tests/runtimes/stable_diffusion_cpp/browser-smoke-contract.mjs',
    'tests/runtimes/stable_diffusion_cpp/core-contract.mjs',
    'runtimes/stable-diffusion-cpp/config/browser-smoke-contract.json', 'runtimes/stable-diffusion-cpp/tools/browser-smoke-contract.ts', 'tests/runtimes/stable_diffusion_cpp/arithmetic-probes.mjs',
  ],
} as const;

export function testFilesForRuntime({ checkoutRoot, runtime }: { checkoutRoot: string; runtime: RuntimeName }): Record<string, string> {
  const files: Record<string, string> = {};
  for (const name of runtimeTestInputs[runtime]) {
    files[name] = path.join(checkoutRoot, name);
  }
  for (const name of ['profiles.json', 'variants.json']) {
    const relative = `runtimes/${runtime}/config/${name}`;
    files[relative] = path.join(checkoutRoot, relative);
  }
  switch (runtime) {
  case 'llama-cpp':
    files['executed/synthetic-model.js'] = fileURLToPath(import.meta.resolve('#tools/validation/synthetic-model'));
    break;
  case 'stable-diffusion-cpp':
    files['executed/browser-smoke-contract.js'] = fileURLToPath(import.meta.resolve('#runtimes/stable-diffusion-cpp/tools/browser-smoke-contract'));
    break;
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
  return files;
}

/** Saved artifacts may be old, incomplete or from another run; decode once here. */
export async function validateSavedRuntimePackage({ checkoutRoot, runtime, packageDirectory, receiptPath, environment, runProcess }: { checkoutRoot: string; runtime: RuntimeName; packageDirectory: string; receiptPath: string; environment: NodeJS.ProcessEnv; runProcess: typeof import('#tools/shared/process').runProcess }): Promise<ValidationReceipt<'llama-cpp', LlamaBrowserResult> | ValidationReceipt<'stable-diffusion-cpp', StableDiffusionBrowserResult>> {
  await clearValidationReceipt({ packageDirectory, receiptPath });
  switch (runtime) {
  case 'llama-cpp': {
    const [{ loadConfiguration }, { verifyRuntimePackage, verifyRuntimeInput }, { checkedLlamaResults, addLlamaValidation }] = await Promise.all([
      import('#runtimes/llama-cpp/tools/config'), import('#runtimes/llama-cpp/tools/verify'), import('./llama-results.js'),
    ]);
    const configuration = await loadConfiguration({ checkoutRoot });
    const validated = await verifyRuntimePackage({ sourceRoot: checkoutRoot, directory: packageDirectory, requireClean: true, checkNpmPack: false });
    const profiles = Object.keys(configuration.profiles);
    const variants = Object.keys(configuration.variants);
    return validateSnapshot({
      checkoutRoot,
      receiptPath,
      environment,
      runProcess,
      io: manifestIo,
      plan: {
        validated,
        profiles,
        variants,
        testFiles: testFilesForRuntime({ checkoutRoot, runtime }),
        generateFixture: createSyntheticModel,
        nodeAsyncify: true,
        testWebGpu: false,
        decodeEnvelope: decodeLlamaBrowserEnvelope,
        checkedResults: ({ results }) => checkedLlamaResults({ profiles, variants, results }),
        updateManifest: addLlamaValidation,
        finalVerify: ({ input }) => verifyRuntimeInput({ sourceRoot: checkoutRoot, input, requireClean: true, checkNpmPack: true }),
      },
    });
  }
  case 'stable-diffusion-cpp': {
    const [{ loadConfiguration }, { verifyRuntimePackage, verifyRuntimeInput }, { checkedImageResults, addImageValidation }] = await Promise.all([
      import('#runtimes/stable-diffusion-cpp/tools/config'), import('#runtimes/stable-diffusion-cpp/tools/verify'), import('./image-results.js'),
    ]);
    const configuration = await loadConfiguration({ checkoutRoot });
    const validated = await verifyRuntimePackage({ sourceRoot: checkoutRoot, directory: packageDirectory, requireClean: true, checkNpmPack: false });
    const profiles = Object.keys(configuration.profiles);
    const variants = Object.keys(configuration.variants);
    const testWebGpu = environment.SDCB_TEST_WEBGPU === '1';
    return validateSnapshot({
      checkoutRoot,
      receiptPath,
      environment,
      runProcess,
      io: manifestIo,
      plan: {
        validated,
        profiles,
        variants,
        testFiles: testFilesForRuntime({ checkoutRoot, runtime }),
        generateFixture: undefined,
        nodeAsyncify: false,
        testWebGpu,
        decodeEnvelope: decodeStableDiffusionBrowserEnvelope,
        checkedResults: ({ results }) => checkedImageResults({ profiles, variants, results, testWebGpu }),
        updateManifest: addImageValidation,
        finalVerify: ({ input }) => verifyRuntimeInput({ sourceRoot: checkoutRoot, input, requireClean: true, checkNpmPack: true }),
      },
    });
  }
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
}
