import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Upstreams } from '#dto/stable-diffusion-cpp';
import profileDefinitions from '#runtimes/stable-diffusion-cpp/config/profiles.json' with { type: 'json' };
import variantDefinitions from '#runtimes/stable-diffusion-cpp/config/variants.json' with { type: 'json' };
import upstreamDefinitions from '#runtimes/stable-diffusion-cpp/config/upstreams.json' with { type: 'json' };

export const runtimeDirectory = 'runtimes/stable-diffusion-cpp';
export const publicRuntimeDirectory = 'runtimes/stable-diffusion-cpp-browser-core';
export type Profile = (typeof profileDefinitions)[keyof typeof profileDefinitions];
export type Variant = (typeof variantDefinitions)[keyof typeof variantDefinitions];
export type UpstreamDeclarations = typeof upstreamDefinitions;
export type Configuration = { profiles: Record<string, Profile>; variants: Record<string, Variant>; upstreams: UpstreamDeclarations };

/** Static data and tooling belong to the same executing checkout. */
export async function loadConfiguration({ checkoutRoot }: { checkoutRoot: string }): Promise<Configuration> {
  const imported = fileURLToPath(import.meta.resolve('#runtimes/stable-diffusion-cpp/config/profiles.json'));
  if (path.resolve(imported) !== path.resolve(checkoutRoot, runtimeDirectory, 'config/profiles.json')) {
    throw new Error('Image tooling does not belong to the selected checkout; run the emitted tooling from that checkout');
  }
  return { profiles: profileDefinitions, variants: variantDefinitions, upstreams: upstreamDefinitions };
}

/** Compare source selection facts while retaining unrelated saved metadata. */
export function upstreamFacts({ upstreams }: { upstreams: Upstreams }) {
  const source = upstreams['stable-diffusion'];
  const ggml = upstreams.ggml;
  return { 'stable-diffusion': { path: source.path, repository: source.repository, commit: source.commit }, ggml: { path: ggml.path, repository: ggml.repository, sourceSubdirectory: ggml.sourceSubdirectory, commit: ggml.commit } };
}

export function selectConfiguration({ configuration, profile, variant }: { configuration: Configuration; profile: string; variant: string }): { profile: Profile; variant: Variant } {
  const selectedProfile = configuration.profiles[profile];
  const selectedVariant = configuration.variants[variant];
  if (selectedProfile === undefined || selectedVariant === undefined) {
    throw new Error(`Unknown image-runtime profile or variant: ${profile}/${variant}`);
  }
  if (selectedProfile.jspi === selectedProfile.asyncify) {
    throw new Error('Select exactly one suspension mode');
  }
  if (selectedProfile.memory64 && !selectedProfile.jspi) {
    throw new Error('Memory64 requires JSPI');
  }
  if (!Number.isSafeInteger(selectedProfile.maximumMemory) || selectedProfile.maximumMemory <= 0 || (!selectedProfile.memory64 && selectedProfile.maximumMemory > 2 ** 32)) {
    throw new Error('Selected memory exceeds its address space');
  }
  if (selectedProfile.pthreads) {
    throw new Error('Image-runtime native pthread builds require an explicitly supported build graph');
  }
  return { profile: selectedProfile, variant: selectedVariant };
}
