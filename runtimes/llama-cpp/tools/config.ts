import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { LlamaProvenance } from '#dto/llama-cpp';
import { selectedGitlinks } from '#tools/source/capture';
import { loadToolchainConfiguration, type ToolchainConfiguration } from '#tools/toolchain/config';
import profiles from '#runtimes/llama-cpp/config/profiles.json' with { type: 'json' };
import variants from '#runtimes/llama-cpp/config/variants.json' with { type: 'json' };
import runtimeToolchain from '#runtimes/llama-cpp/config/toolchain.json' with { type: 'json' };

export const runtimePath = 'runtimes/llama-cpp';
export const upstreamPath = `${runtimePath}/vendor/llama.cpp`;

// The executing checkout owns these JSON modules. CMake checks
// the chosen execution flags; adding an unselected profile cannot reject a build.
export type Profile = (typeof profiles)[keyof typeof profiles] & { gpuProfile?: boolean };
export type Variant = (typeof variants)[keyof typeof variants];
export type Configuration = {
  profiles: Readonly<Record<string, Profile>>;
  variants: Readonly<Record<string, Variant>>;
  toolchain: ToolchainConfiguration & typeof runtimeToolchain;
};

/** The caller must identify the checkout that owns this tool and its config. */
export async function loadConfiguration({ checkoutRoot }: { checkoutRoot: string }): Promise<Configuration> {
  const importedProfiles = fileURLToPath(import.meta.resolve('#runtimes/llama-cpp/config/profiles.json'));
  if (resolve(importedProfiles) !== resolve(checkoutRoot, runtimePath, 'config/profiles.json')) {
    throw new Error('Configuration must belong to the executing checkout');
  }
  return { profiles, variants, toolchain: { ...await loadToolchainConfiguration({ checkoutRoot }), ...runtimeToolchain } };
}

export async function upstreamCommit({ checkoutRoot, revision }: { checkoutRoot: string; revision: string | undefined }): Promise<string> {
  const links = await selectedGitlinks({ sourceRoot: checkoutRoot, revision, paths: { llama: upstreamPath } });
  const commit = links['llama'];
  if (commit === undefined) {
    throw new Error('Selected revision has no llama.cpp gitlink');
  }
  return commit;
}

export function selectConfiguration({ configuration, profile, variant }: { configuration: Configuration; profile: string; variant: string }): { profile: Profile; variant: Variant } {
  const selectedProfile = Object.hasOwn(configuration.profiles, profile) ? configuration.profiles[profile] : undefined;
  const selectedVariant = Object.hasOwn(configuration.variants, variant) ? configuration.variants[variant] : undefined;
  if (selectedProfile === undefined || selectedVariant === undefined) {
    throw new Error(`Unknown llama.cpp profile/variant: ${profile}/${variant}`);
  }
  return { profile: selectedProfile, variant: selectedVariant };
}

/** Inputs consumed by CMake and the native compiler, independent of Node tooling. */
export function sourceCaptureSelection({ kind }: { kind: 'native' | 'wasm' }): { include: readonly string[]; exclude: readonly string[] } {
  const include = [`${runtimePath}/CMakeLists.txt`, `${runtimePath}/cmake`, `${runtimePath}/native`, `${runtimePath}/host/emscripten`, `${runtimePath}/upstream-patches-only-as-a-last-resort-with-explicit-user-approval`, `${runtimePath}/config/variants.json`, 'tools/abi/Bindings.cmake', 'tools/abi/ConfigureFingerprint.cmake', 'tools/abi/fingerprint.h.in', 'tools/shared/TargetSources.cmake' ];
  switch (kind) {
  case 'native': include.push('tests/runtimes/llama_cpp/model-graph/boundary.cpp');
    break;
  case 'wasm': break;
  default: {
    const _ex: never = kind;
    throw new Error(`Unknown build kind: ${String(_ex)}`);
  }
  }
  return { include, exclude: [] };
}

/** Runtime capability advertised by builds and checked at artifact reception. */
export function visionBf16StorageMetadata(): NonNullable<LlamaProvenance['visionBf16Storage']> {
  return { version: 1, modes: ['legacy_expand', 'retain_checked'], default: 'legacy_expand' };
}
