import type { SpeechBrowserResult, SpeechManifest } from '#dto/speech-cpp';
import { requireResultPairs } from './pairs.js';

export function checkedSpeechResults({ profiles, results }: { profiles: Readonly<Record<string, { memory64: boolean }>>; results: readonly SpeechBrowserResult[] }): void {
  requireResultPairs({ profiles: Object.keys(profiles), variants: ['browser'], results });
  for (const result of results) {
    const versions = [result.apiVersionMajor, result.apiVersionMinor];
    if (result.pointerBytes !== (profiles[result.profile]?.memory64 ? 8 : 4) || versions.some(version => !Number.isSafeInteger(version) || version < 0)) {
      throw new Error('Wrong speech startup evidence');
    }
  }
}

export function addSpeechValidation({ manifest, results }: { manifest: SpeechManifest; results: readonly SpeechBrowserResult[] }): SpeechManifest {
  const updated = structuredClone(manifest);
  for (const result of results) {
    const provenance = updated.profiles[result.profile]?.variants[result.variant];
    if (provenance === undefined) {
      throw new Error('Browser result does not name a packaged profile/variant');
    }
    provenance.validation.browserSmoke = true;
    provenance.validation.browserSmokeScope = 'Real Wasm in a Worker, pointer width and public speech API version';
  }
  return updated;
}
