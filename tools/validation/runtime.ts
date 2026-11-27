import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RuntimeName } from '#dto/shared';
import { decodeLlamaBrowserEnvelope, type LlamaBrowserResult } from '#dto/llama-cpp';
import { decodeSpeechBrowserEnvelope, type SpeechBrowserResult } from '#dto/speech-cpp';
import { decodeStableDiffusionBrowserEnvelope, type StableDiffusionBrowserResult } from '#dto/stable-diffusion-cpp';
import { createSyntheticModel } from './synthetic-model.js';
import { clearValidationReceipt, manifestIo, validateSnapshot, type ExecutionSettings, type ValidationPhase, type ValidationReceipt } from './snapshot.js';
import type { ChildEnvironment } from '#tools/shared/process';
import { addSpeechValidation, checkedSpeechResults } from './speech-cpp-results.js';

export type { ValidationPhase, ValidationReceipt } from './snapshot.js';

const sharedTestInputs = ['tests/runtimes/browser-smoke-harness.mjs', 'tools/shared/node-env.ts'];
const llamaRegressionInputs = [
  'tests/runtimes/llama_cpp/asyncify-rewind.mjs', 'tests/runtimes/llama_cpp/chat-surface.mjs',
  'tests/runtimes/llama_cpp/file-mount-lifetime.mjs', 'runtimes/llama-cpp/vendor/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja',
];
const imageRegressionInputs = ['tests/runtimes/stable_diffusion_cpp/arithmetic-probes.mjs'];
const browserHarnesses = {
  'llama-cpp': 'tests/runtimes/llama_cpp/browser-smoke.mjs',
  'speech-cpp': 'tests/runtimes/speech-cpp/browser-smoke.mjs',
  'stable-diffusion-cpp': 'tests/runtimes/stable_diffusion_cpp/browser-smoke.mjs',
} as const;

/** Include indirectly loaded harnesses: results must bind to those exact bytes. */
export const runtimeTestInputs = {
  'llama-cpp': [
    ...sharedTestInputs, browserHarnesses['llama-cpp'], ...llamaRegressionInputs, 'tools/validation/synthetic-model.ts',
  ],
  'speech-cpp': [
    ...sharedTestInputs, browserHarnesses['speech-cpp'], 'tests/runtimes/speech-cpp/browser-smoke-worker.mjs',
    'tests/runtimes/speech-cpp/startup-contract.mjs',
  ],
  'stable-diffusion-cpp': [
    ...sharedTestInputs, browserHarnesses['stable-diffusion-cpp'], 'tests/runtimes/stable_diffusion_cpp/gguf-fixture.mjs',
    'tests/runtimes/stable_diffusion_cpp/model-io-fixtures.mjs', 'tests/runtimes/stable_diffusion_cpp/browser-smoke-contract.mjs',
    'tests/runtimes/stable_diffusion_cpp/core-contract.mjs',
    'runtimes/stable-diffusion-cpp/config/browser-smoke-contract.json', 'runtimes/stable-diffusion-cpp/tools/browser-smoke-contract.ts', ...imageRegressionInputs,
  ],
} as const;

export function testFilesForRuntime({ checkoutRoot, runtime, phase }: { checkoutRoot: string; runtime: RuntimeName; phase: ValidationPhase }): Record<string, string> {
  const files: Record<string, string> = {};
  files['executed/node-env.js'] = fileURLToPath(import.meta.resolve('#tools/shared/node-env'));
  for (const name of runtimeTestInputs[runtime]) {
    if (phase === 'startup' && (llamaRegressionInputs.includes(name) || imageRegressionInputs.includes(name))) {
      continue;
    }
    files[name] = path.join(checkoutRoot, name);
  }
  for (const name of ['profiles.json', 'variants.json']) {
    const relative = `runtimes/${runtime}/config/${name}`;
    files[relative] = path.join(checkoutRoot, relative);
  }
  switch (runtime) {
  case 'speech-cpp':
    break;
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

function smokeVariants({ phase, configured }: { phase: ValidationPhase; configured: readonly string[] }): readonly string[] {
  switch (phase) {
  case 'startup': return ['browser'];
  case 'regression': return ['test'];
  case 'full': return configured;
  default: {
    const exhaustive: never = phase;
    throw new Error(`Unknown validation phase: ${String(exhaustive)}`);
  }
  }
}

/** Saved artifacts may be old, incomplete or from another run; decode once here. */
export async function validateSavedRuntimePackage({ checkoutRoot, runtime, packageDirectory, receiptPath, phase: requestedPhase, settings, childEnvironment, runProcess }: { checkoutRoot: string; runtime: RuntimeName; packageDirectory: string; receiptPath: string; phase?: ValidationPhase | undefined; settings: ExecutionSettings & { testWebGpu?: string | undefined }; childEnvironment: ChildEnvironment | undefined; runProcess: typeof import('#tools/shared/process').runProcess }): Promise<ValidationReceipt<'llama-cpp', LlamaBrowserResult> | ValidationReceipt<'stable-diffusion-cpp', StableDiffusionBrowserResult> | ValidationReceipt<'speech-cpp', SpeechBrowserResult>> {
  await clearValidationReceipt({ packageDirectory, receiptPath });
  const phase = requestedPhase ?? 'full';
  const verifyNpmPack = phase === 'full';
  const basePlan = { phase, browserHarness: browserHarnesses[runtime], testFiles: testFilesForRuntime({ checkoutRoot, runtime, phase }) };
  switch (runtime) {
  case 'speech-cpp': {
    switch (phase) {
    case 'startup': break;
    case 'regression':
    case 'full':
      throw new Error('Use --phase startup for packaged speech browser validation');
    default: {
      const exhaustive: never = phase;
      throw new Error(`Unknown validation phase: ${String(exhaustive)}`);
    }
    }
    const [{ loadConfiguration }, { verifyRuntimePackage }] = await Promise.all([
      import('#runtimes/speech-cpp/tools/config'), import('#runtimes/speech-cpp/tools/verify'),
    ]);
    const configuration = await loadConfiguration({ checkoutRoot });
    const validated = await verifyRuntimePackage({ sourceRoot: checkoutRoot, directory: packageDirectory, requireClean: true, checkNpmPack: false });
    const profiles = Object.keys(configuration.profiles);
    const variants = Object.keys(configuration.variants);
    return validateSnapshot({
      checkoutRoot,
      receiptPath,
      settings,
      childEnvironment,
      runProcess,
      io: manifestIo,
      plan: {
        ...basePlan,
        validated,
        profiles,
        variants,
        generateFixture: undefined,
        nodeAsyncify: false,
        testWebGpu: false,
        decodeEnvelope: decodeSpeechBrowserEnvelope,
        checkedResults: ({ results }) => checkedSpeechResults({ profiles: configuration.profiles, results }),
        updateManifest: addSpeechValidation,
        finalVerify: undefined,
      },
    });
  }
  case 'llama-cpp': {
    const [{ loadConfiguration }, { verifyRuntimePackage, verifyRuntimeInput }, { checkedLlamaResults, addLlamaValidation }] = await Promise.all([
      import('#runtimes/llama-cpp/tools/config'), import('#runtimes/llama-cpp/tools/verify'), import('./llama-cpp-results.js'),
    ]);
    const configuration = await loadConfiguration({ checkoutRoot });
    const validated = await verifyRuntimePackage({ sourceRoot: checkoutRoot, directory: packageDirectory, requireClean: true, checkNpmPack: false });
    const profiles = Object.keys(configuration.profiles);
    const variants = Object.keys(configuration.variants);
    return validateSnapshot({
      checkoutRoot,
      receiptPath,
      settings,
      childEnvironment,
      runProcess,
      io: manifestIo,
      plan: {
        ...basePlan,
        validated,
        profiles,
        variants,
        generateFixture: createSyntheticModel,
        nodeAsyncify: phase !== 'startup',
        testWebGpu: false,
        decodeEnvelope: decodeLlamaBrowserEnvelope,
        checkedResults: ({ results }) => checkedLlamaResults({ profiles, variants: smokeVariants({ phase, configured: variants }), results }),
        updateManifest: addLlamaValidation,
        finalVerify: verifyNpmPack ? ({ input }) => verifyRuntimeInput({ sourceRoot: checkoutRoot, input, requireClean: true, checkNpmPack: true }) : undefined,
      },
    });
  }
  case 'stable-diffusion-cpp': {
    const [{ loadConfiguration }, { verifyRuntimePackage, verifyRuntimeInput }, { checkedStableDiffusionResults, addStableDiffusionValidation }] = await Promise.all([
      import('#runtimes/stable-diffusion-cpp/tools/config'), import('#runtimes/stable-diffusion-cpp/tools/verify'), import('./stable-diffusion-cpp-results.js'),
    ]);
    const configuration = await loadConfiguration({ checkoutRoot });
    const validated = await verifyRuntimePackage({ sourceRoot: checkoutRoot, directory: packageDirectory, requireClean: true, checkNpmPack: false });
    const profiles = Object.keys(configuration.profiles);
    const variants = Object.keys(configuration.variants);
    const testWebGpu = phase !== 'startup' && settings.testWebGpu === '1';
    return validateSnapshot({
      checkoutRoot,
      receiptPath,
      settings,
      childEnvironment,
      runProcess,
      io: manifestIo,
      plan: {
        ...basePlan,
        validated,
        profiles,
        variants,
        generateFixture: undefined,
        nodeAsyncify: false,
        testWebGpu,
        decodeEnvelope: decodeStableDiffusionBrowserEnvelope,
        checkedResults: ({ results }) => checkedStableDiffusionResults({ profiles, variants: smokeVariants({ phase, configured: variants }), results, testWebGpu }),
        updateManifest: addStableDiffusionValidation,
        finalVerify: verifyNpmPack ? ({ input }) => verifyRuntimeInput({ sourceRoot: checkoutRoot, input, requireClean: true, checkNpmPack: true }) : undefined,
      },
    });
  }
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
}
