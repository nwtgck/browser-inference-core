import { copyFile, mkdir, rename } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { containedPath, fileIdentity, regularFile } from '#tools/shared/files';
import { ProcessError, runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tools/shared/temporary-directory';
import type { SdkPatch } from '#tools/toolchain/config';

async function checkDirection({ root, patchPath, direction }: { root: string; patchPath: string; direction: 'forward' | 'reverse' }): Promise<boolean> {
  let directionFlag: string;
  switch (direction) {
  case 'forward': directionFlag = '--forward';
    break;
  case 'reverse': directionFlag = '--reverse';
    break;
  default: {
    const _ex: never = direction;
    throw new Error(String(_ex));
  }
  }
  try {
    await runProcess({ command: 'patch', args: ['--batch', '--force', '--fuzz=0', '--reject-file=-', '--dry-run', directionFlag, '-p1', '-i', patchPath], cwd: root });
    return true;
  } catch (error) {
    if (error instanceof ProcessError && error.exitCode === 1) {
      return false;
    }
    throw error;
  }
}

/** Work only on a private file copy; source hashes detect concurrent writes,
 * never admit a particular upstream file. The patch context owns compatibility. */
async function sdkPatch({ checkoutRoot, emscriptenRoot, source, patch, operation }: { checkoutRoot: string; emscriptenRoot: string; source: string; patch: SdkPatch; operation: 'apply' | 'verify' }): Promise<void> {
  const target = await regularFile({ path: containedPath({ root: emscriptenRoot, relative: source }) });
  const before = await fileIdentity({ path: target });
  const patchSource = await regularFile({ path: containedPath({ root: checkoutRoot, relative: patch.file }) });
  let parent: string;
  switch (operation) {
  case 'apply': parent = dirname(target);
    break;
  case 'verify': parent = tmpdir();
    break;
  default: {
    const _ex: never = operation;
    throw new Error(String(_ex));
  }
  }
  await using temporary = await temporaryDirectory({ prefix: join(parent, '.sdk-patch-') });
  const work = temporary.root;
  const copied = join(work, source);
  const patchPath = join(work, 'change.patch');
  await mkdir(dirname(copied), { recursive: true });
  await copyFile(target, copied);
  await copyFile(patchSource, patchPath);
  const observedPatch = await fileIdentity({ path: patchPath });
  if (observedPatch.bytes !== patch.bytes || observedPatch.sha256 !== patch.sha256) {
    throw new Error('Emscripten SDK patch changed after configuration was loaded');
  }
  const forward = await checkDirection({ root: work, patchPath, direction: 'forward' });
  const reverse = await checkDirection({ root: work, patchPath, direction: 'reverse' });
  if (!forward && reverse) {
    return;
  }
  if (!forward || reverse) {
    throw new Error('Emscripten SDK patch context is missing or ambiguous');
  }
  switch (operation) {
  case 'verify': throw new Error('Emscripten SDK patch is not applied; prepare the local toolchain');
  case 'apply': break;
  default: {
    const _ex: never = operation;
    throw new Error(String(_ex));
  }
  }
  await runProcess({ command: 'patch', args: ['--batch', '--fuzz=0', '--reject-file=-', '--forward', '-p1', '-i', patchPath], cwd: work });
  if (!await checkDirection({ root: work, patchPath, direction: 'reverse' })) {
    throw new Error('Emscripten SDK patch did not produce the expected reverse context');
  }
  const current = await fileIdentity({ path: target });
  if (current.bytes !== before.bytes || current.sha256 !== before.sha256) {
    throw new Error('Emscripten SDK source changed while applying patch');
  }
  await rename(copied, target);
}

export async function applySdkPatch({ checkoutRoot, emscriptenRoot, source, patch }: { checkoutRoot: string; emscriptenRoot: string; source: string; patch: SdkPatch }): Promise<void> {
  await sdkPatch({ checkoutRoot, emscriptenRoot, source, patch, operation: 'apply' });
}

export async function verifySdkPatch({ checkoutRoot, emscriptenRoot, source, patch }: { checkoutRoot: string; emscriptenRoot: string; source: string; patch: SdkPatch }): Promise<void> {
  await sdkPatch({ checkoutRoot, emscriptenRoot, source, patch, operation: 'verify' });
}
