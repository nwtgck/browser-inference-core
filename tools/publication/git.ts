import { createHash } from 'node:crypto';
import { BinaryReader } from '#tools/shared/streams';
import { runProcess, startProcess } from '#tools/shared/process';
import type { FileEntry } from '#dto/shared';
import { relativePath } from '#tools/shared/files';
import { bytesIdentity, assertSame } from '#tools/package/files';

export function gitObjectId({ value }: { value: string }): string {
  if (!/^[a-f0-9]{40}$/u.test(value)) {
    throw new Error('Expected a complete Git object identity');
  }
  return value;
}

/** Local Git only; transport remains visible in the workflow. */
export function localGitEnvironment(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (key.startsWith('GIT_')) {
      delete env[key];
    }
  }
  return { ...env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null', GIT_ALLOW_PROTOCOL: '', GIT_NO_LAZY_FETCH: '1', GIT_NO_REPLACE_OBJECTS: '1', GIT_TERMINAL_PROMPT: '0' };
}

export async function localGit({ cwd, args, input }: { cwd: string; args: readonly string[]; input?: string | Uint8Array }): Promise<Buffer> {
  const verb = args[0];
  if (!verb || !['init', 'config', 'add', 'write-tree', 'ls-tree', 'cat-file', 'rev-parse', 'show', 'check-ref-format', 'diff-tree', 'commit-tree'].includes(verb)) {
    throw new Error('Publication accepts local Git operations only');
  }
  return (await runProcess({ command: 'git', args: ['--no-replace-objects', ...args], cwd, env: localGitEnvironment(), ...(input === undefined ? {} : { input }) })).stdout;
}

export async function gitBlob({ work, tree, path }: { work: string; tree: string; path: string }): Promise<Buffer> {
  gitObjectId({ value: tree });
  relativePath({ value: path });
  return localGit({ cwd: work, args: ['show', `${tree}:${path}`] });
}

export async function verifyGitTree({ work, tree, files }: { work: string; tree: string; files: ReadonlyMap<string, FileEntry> }): Promise<void> {
  gitObjectId({ value: tree });
  const listing = await localGit({ cwd: work, args: ['ls-tree', '-r', '-z', '--full-tree', tree] });
  const objects = new Map<string, string>();
  for (const record of listing.toString('utf8').split('\0').filter(Boolean)) {
    const match = /^(100644|100755) blob ([a-f0-9]{40})\t(.+)$/u.exec(record);
    const oid = match?.[2];
    const path = match?.[3];
    if (!oid || !path || objects.has(path)) {
      throw new Error('Non-regular or duplicate entry in publication Git tree');
    }
    relativePath({ value: path });
    objects.set(path, oid);
  }
  assertSame({ actual: [...objects.keys()].sort(), expected: [...files.keys()].sort(), message: 'Publication Git tree differs from validated package' });
  const child = startProcess({ command: 'git', args: ['--no-replace-objects', 'cat-file', '--batch'], cwd: work, env: localGitEnvironment() });
  const reader = new BinaryReader({ stream: child.stdout });
  try {
    for (const [path, oid] of objects) {
      const entry = files.get(path);
      if (!entry) {
        throw new Error('Missing Git payload identity');
      }
      child.stdin.write(oid + '\n');
      if (await reader.line({ limit: 256 }) !== `${oid} blob ${entry.bytes}`) {
        throw new Error(`Git blob type/size differs from package: ${path}`);
      }
      const digest = createHash('sha256');
      let remaining = entry.bytes;
      while (remaining) {
        const chunk = await reader.take({ count: Math.min(remaining, 1024 * 1024) });
        digest.update(chunk);
        remaining -= chunk.length;
      }
      if ((await reader.take({ count: 1 }))[0] !== 10 || digest.digest('hex') !== entry.sha256) {
        throw new Error(`Git blob differs from package bytes: ${path}`);
      }
    }
    child.stdin.end();
    await reader.expectEnd();
    await child.completed;
  } catch (error) {
    child.stdout.destroy();
    child.kill();
    await child.completed.catch(() => undefined);
    throw error;
  }
}
