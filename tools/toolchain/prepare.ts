import { copyFile, mkdtemp, mkdir, readdir, rename, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { decodeEmscriptenReleases } from '#dto/shared';
import { directory, fileIdentity, regularFile, relativePath } from '#tools/shared/files';
import { readJson } from '#tools/shared/json';
import { withOutputDirectories } from '#tools/shared/output';
import { runProcess } from '#tools/shared/process';
import { loadToolchainConfiguration, type AsyncifyPatchConfig } from '#tools/toolchain/config';

const asyncifySource = 'src/lib/libasync.js';
const patchPath = 'toolchain/patches/emscripten-asyncify-bigint.patch';

export async function verifyAsyncifyBigIntPatch({ emscriptenRoot, patchConfig }: { emscriptenRoot: string; patchConfig: AsyncifyPatchConfig }): Promise<void> {
  const source = join(emscriptenRoot, asyncifySource);
  if ((await fileIdentity({ path: source })).sha256 !== patchConfig.patchedSha256) {
    throw new Error('Asyncify requires the pinned bigint rewind patch; prepare the local toolchain');
  }
}

/** Apply the already approved four-guard change to a private copy first. */
export async function applyAsyncifyBigIntPatch({ checkoutRoot, emscriptenRoot, patchConfig }: { checkoutRoot: string; emscriptenRoot: string; patchConfig: AsyncifyPatchConfig }): Promise<void> {
  const target = await regularFile({ path: join(emscriptenRoot, asyncifySource) });
  const input = await fileIdentity({ path: target });
  if (input.sha256 === patchConfig.patchedSha256) {
    return;
  }
  if (input.sha256 !== patchConfig.sourceSha256) {
    throw new Error('Unexpected Emscripten Asyncify source');
  }
  const patch = await regularFile({ path: join(checkoutRoot, patchPath) });
  const temporary = await mkdtemp(join(dirname(target), '.asyncify-patch-'));
  try {
    const copied = join(temporary, asyncifySource);
    await mkdir(dirname(copied), { recursive: true });
    await copyFile(target, copied);
    await runProcess({ command: 'patch', args: ['--batch', '--forward', '--fuzz=0', '-p1', '-i', patch], cwd: temporary });
    if ((await fileIdentity({ path: copied })).sha256 !== patchConfig.patchedSha256) {
      throw new Error('Patched Emscripten Asyncify source does not match its pin');
    }
    if ((await fileIdentity({ path: target })).sha256 !== input.sha256) {
      throw new Error('Emscripten Asyncify source changed while applying patch');
    }
    await rename(copied, target);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

/** Verify and extract the exact pinned release, without downloading it. */
export async function prepareDawn({ toolsRoot, dawnTag, dawnSha256 }: { toolsRoot: string; dawnTag: string; dawnSha256: string }): Promise<string> {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(dawnTag)) {
    throw new Error('Invalid Dawn release tag');
  }
  const tools = await directory({ path: toolsRoot });
  const source = await regularFile({ path: join(tools, 'downloads', `emdawnwebgpu_pkg-${dawnTag}.zip`) });
  const temporary = await mkdtemp(join(tmpdir(), 'dawn-archive-'));
  try {
    const archive = join(temporary, 'input.zip');
    await copyFile(source, archive);
    if ((await fileIdentity({ path: archive })).sha256 !== dawnSha256) {
      throw new Error('Dawn archive checksum mismatch');
    }
    const listing = (await runProcess({ command: 'unzip', args: ['-Z', '-1', archive], cwd: temporary })).stdout.toString('utf8').split('\n').filter(Boolean);
    const details = (await runProcess({ command: 'zipinfo', args: ['-l', archive], cwd: temporary })).stdout.toString('utf8').split('\n');
    const modes = details.filter(line => /^[bcdlps-][rwxStTs-]{9}\s/u.test(line));
    if (modes.length !== listing.length || modes.some(line => line[0] !== '-' && line[0] !== 'd')) {
      throw new Error('Dawn archive contains unsupported entries');
    }
    const names = new Set<string>();
    for (const entry of listing) {
      const name = entry.endsWith('/') ? entry.slice(0, -1) : entry;
      relativePath({ value: name });
      if ((name !== 'emdawnwebgpu_pkg' && !name.startsWith('emdawnwebgpu_pkg/')) || names.has(name)) {
        throw new Error(`Invalid Dawn archive member: ${entry}`);
      }
      names.add(name);
    }
    if (!names.has('emdawnwebgpu_pkg/emdawnwebgpu.port.py')) {
      throw new Error('Dawn archive does not contain the expected port');
    }
    const result = join(tools, 'emdawnwebgpu_pkg');
    await withOutputDirectories({
      destinations: [result],
      replace: true,
      preserveUnchanged: true,
      prepare: async ({ paths: [output] }) => {
        if (!output) {
          throw new Error('Missing Dawn output');
        }
        const unpack = join(dirname(output), 'unpacked');
        await mkdir(unpack);
        await runProcess({ command: 'unzip', args: ['-q', archive, '-d', unpack], cwd: temporary });
        const packageRoot = await directory({ path: join(unpack, 'emdawnwebgpu_pkg') });
        await regularFile({ path: join(packageRoot, 'emdawnwebgpu.port.py') });
        for (const name of await readdir(packageRoot)) {
          await rename(join(packageRoot, name), join(output, name));
        }
      },
    });
    return result;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

export async function prepareToolchain({ checkoutRoot, toolsRoot, dawnOnly }: { checkoutRoot: string; toolsRoot: string; dawnOnly: boolean }): Promise<void> {
  const config = await loadToolchainConfiguration({ checkoutRoot });
  if (!dawnOnly) {
    const sdk = await directory({ path: join(toolsRoot, 'emsdk') });
    const releases = await readJson({ path: join(sdk, 'emscripten-releases-tags.json'), decode: decodeEmscriptenReleases });
    if (releases.releases[config.emsdkVersion] !== config.emscriptenRelease) {
      throw new Error('emsdk release identity differs from pinned configuration');
    }
    await applyAsyncifyBigIntPatch({ checkoutRoot, emscriptenRoot: join(sdk, 'upstream/emscripten'), patchConfig: config.emscriptenAsyncifyBigIntPatch });
  }
  await prepareDawn({ toolsRoot, dawnTag: config.dawnTag, dawnSha256: config.dawnSha256 });
}
