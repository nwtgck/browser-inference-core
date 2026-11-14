import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { RuntimeName } from '#dto/shared';
import type { ToolchainConfiguration } from '#tools/toolchain/config';

const cachePolicy = 'bic-ts-v1';

function digest({ value }: { value: object }): string {
  const input = JSON.stringify(value, (_key: string, item: unknown) => {
    if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
      return Object.fromEntries(Object.entries(item).sort(([left], [right]) => left.localeCompare(right)));
    }
    return item;
  });
  return createHash('sha256').update(input).digest('hex');
}

/** Cache keys describe a partition, not an assertion that cached objects are correct. */
export function cachePlan({ runtime, profile, variant, profileConfiguration, variantConfiguration, sourceCommit, toolchain, ccacheVersion, platform, workspace }: {
  runtime: RuntimeName; profile: string; variant: string;
  profileConfiguration: object; variantConfiguration: object;
  sourceCommit: string; toolchain: ToolchainConfiguration; ccacheVersion: string;
  platform: { os: string; arch: string; environment: string }; workspace: string;
}) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(profile) || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(variant)) {
    throw new Error('Invalid cache path');
  }
  if (!/^[a-f0-9]{40}$/u.test(sourceCommit)) {
    throw new Error('Invalid source commit for cache key');
  }
  if (!ccacheVersion || !platform.os || !platform.arch || !platform.environment || !workspace) {
    throw new Error('Incomplete cache partition');
  }
  const identity = digest({
    value: {
      emsdkVersion: toolchain.emsdkVersion,
      emscriptenRelease: toolchain.emscriptenRelease,
      emscriptenAsyncifyBigIntPatch: toolchain.emscriptenAsyncifyBigIntPatch,
      emscriptenTypeScriptPatch: toolchain.emscriptenTypeScriptPatch,
      dawnSha256: toolchain.dawnSha256,
      ccacheVersion,
      platform,
      workspace,
      runtime,
      profile,
      variant,
      profileConfiguration,
      variantConfiguration,
    },
  }).slice(0, 32);
  const partition = `${cachePolicy}-${runtime}-${profile}-${variant}-${identity}`;
  const emRestorePrefix = `${partition}-em-`;
  const ccRestorePrefix = `${partition}-cc-`;
  return {
    identity,
    dawnKey: `${cachePolicy}-dawn-${toolchain.dawnSha256}`,
    emKey: `${emRestorePrefix}${sourceCommit}`,
    ccKey: `${ccRestorePrefix}${sourceCommit}`,
    emRestorePrefix,
    ccRestorePrefix,
    emStagingPath: `.cache/restored/${runtime}/${profile}/${variant}/em`,
    ccStagingPath: `.cache/restored/${runtime}/${profile}/${variant}/cc`,
    emCachePath: '.tools/emsdk/upstream/emscripten/cache',
    ccCachePath: `.cache/ccache/${runtime}/${profile}/${variant}`,
  };
}

/** Configure standard ccache; a missing restored cache remains an ordinary cold build. */
export async function configureCache({ workRoot, plan }: { workRoot: string; plan: ReturnType<typeof cachePlan> }): Promise<Record<string, string>> {
  const root = resolve(workRoot);
  const configPath = join(root, '.cache/ccache.conf');
  await mkdir(join(root, '.cache'), { recursive: true });
  await writeFile(configPath, `compiler_check = content\ndirect_mode = false\nsloppiness =\nhard_link = false\nfile_clone = false\nmax_size = 256M\ncompression = true\nnamespace = ${plan.identity}\n`);
  return {
    EM_CACHE: join(root, plan.emCachePath),
    EM_COMPILER_WRAPPER: 'ccache',
    CCACHE_DIR: join(root, plan.ccCachePath),
    CCACHE_CONFIGPATH: configPath,
    BIC_CACHE_ID: plan.identity,
  };
}

/** A failed or partial restore must never replace a fresh cache. */
export function cacheRestoreDecision({ requestedKey, matchedKey, outcome, restorePrefix }: {
  requestedKey: string; matchedKey: string | undefined; outcome: 'success' | 'miss' | 'failure'; restorePrefix?: string;
}): 'accept' | 'fresh' {
  if (outcome !== 'success' || !matchedKey) {
    return 'fresh';
  }
  if (matchedKey === requestedKey) {
    return 'accept';
  }
  if (restorePrefix && requestedKey.startsWith(restorePrefix) && matchedKey.startsWith(restorePrefix) &&
      /^[a-f0-9]{40}$/u.test(requestedKey.slice(restorePrefix.length)) && /^[a-f0-9]{40}$/u.test(matchedKey.slice(restorePrefix.length))) {
    return 'accept';
  }
  return 'fresh';
}
