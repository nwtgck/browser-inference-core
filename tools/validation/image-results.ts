import type { StableDiffusionBrowserResult, StableDiffusionManifest } from '#dto/stable-diffusion-cpp';
import { imageSmokeScope } from '#runtimes/stable-diffusion-cpp/tools/browser-smoke-contract';
import { requireResultPairs } from './pairs.js';

// Every named arithmetic probe is required. The shared producer supplies scope text.
const sharedProbes = ['timestep', 'bf16Weights'] as const;
const testOnlyProbes = ['conv3dBias', 'webgpuPerformance'] as const;

export function checkedImageResults({ profiles, variants, results, testWebGpu }: { profiles: readonly string[]; variants: readonly string[]; results: readonly StableDiffusionBrowserResult[]; testWebGpu: boolean }): void {
  requireResultPairs({ profiles, variants, results });
  const backends = testWebGpu ? ['CPU', 'WebGPU'] : ['CPU'];
  for (const result of results) {
    if (result.scope !== imageSmokeScope({ testWebGpu })) {
      throw new Error('Wrong image smoke scope');
    }
    if (result.variant === 'browser') {
      if ('graphWalk' in result) {
        throw new Error('Wrong image graph walk evidence');
      }
      for (const key of sharedProbes) {
        if (result[key].length !== 0) {
          throw new Error(`Wrong image ${key} evidence`);
        }
      }
      for (const key of testOnlyProbes) {
        if (key in result) {
          throw new Error(`Wrong image ${key} evidence`);
        }
      }
    } else {
      if (result.graphWalk !== true) {
        throw new Error('Wrong image graph walk evidence');
      }
      for (const key of [...sharedProbes, ...testOnlyProbes]) {
        const evidence = result[key];
        if (evidence?.length !== backends.length) {
          throw new Error(`Wrong image ${key} evidence`);
        }
        for (const [index, backend] of backends.entries()) {
          if (evidence[index]?.backend !== backend || evidence[index]?.passed !== true) {
            throw new Error(`Wrong image ${key} evidence`);
          }
        }
      }
    }
  }
}

export function addImageValidation({ manifest, results }: { manifest: StableDiffusionManifest; results: readonly StableDiffusionBrowserResult[] }): StableDiffusionManifest {
  const updated = structuredClone(manifest);
  for (const result of results) {
    const provenance = updated.profiles[result.profile]?.variants[result.variant];
    if (provenance === undefined) {
      throw new Error('Browser result does not name a packaged profile/variant');
    }
    provenance.validation.browserSmoke = true;
    provenance.validation.browserSmokeScope = result.scope;
    provenance.validation.browserSmokeEvidence = structuredClone({
      timestep: result.timestep,
      bf16Weights: result.bf16Weights,
      ...(result.graphWalk === undefined ? {} : { graphWalk: result.graphWalk }),
      ...(result.conv3dBias === undefined ? {} : { conv3dBias: result.conv3dBias }),
      ...(result.webgpuPerformance === undefined ? {} : { webgpuPerformance: result.webgpuPerformance }),
    });
  }
  return updated;
}
