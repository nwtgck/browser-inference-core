import type { LlamaBrowserResult, LlamaManifest } from '#dto/llama-cpp';
import { requireResultPairs } from './pairs.js';

export function checkedLlamaResults({ profiles, variants, results }: { profiles: readonly string[]; variants: readonly string[]; results: readonly LlamaBrowserResult[] }): void {
  requireResultPairs({ profiles, variants, results });
  for (const result of results) {
    if (result.profile.startsWith('cpu-')) {
      if (result.syntheticModel !== true) {
        throw new Error('Wrong CPU smoke scope');
      }
    } else {
      if (result.mockedAdapter !== true || result.suspension !== true || result.syntheticModel === true) {
        throw new Error('Wrong GPU smoke scope');
      }
      const browser = result.variant === 'browser';
      if (result.defaultNavigatorFallback !== true || result.hostNavigatorInjection !== browser || result.independentFactories !== browser) {
        throw new Error('Wrong GPU host navigator smoke scope');
      }
    }
  }
}

export function addLlamaValidation({ manifest, results }: { manifest: LlamaManifest; results: readonly LlamaBrowserResult[] }): LlamaManifest {
  const updated = structuredClone(manifest);
  for (const result of results) {
    const provenance = updated.profiles[result.profile]?.variants[result.variant];
    if (provenance === undefined) {
      throw new Error('Browser result does not name a packaged profile/variant');
    }
    provenance.validation.browserSmoke = true;
    if (result.profile.startsWith('cpu-')) {
      provenance.validation.syntheticUntrainedModel = structuredClone(result);
    } else {
      provenance.validation.mockedAdapterSuspension = structuredClone(result);
    }
  }
  return updated;
}
