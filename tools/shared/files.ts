import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { lstat, mkdir, open, readdir, readlink } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import type { FileIdentity, SourceIdentity } from '#dto/shared';
export type { FileIdentity, SourceIdentity } from '#dto/shared';

export function relativePath({ value }: { value: string }): string {
  if (!value || value.includes('\\') || Array.from(value).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) || isAbsolute(value) || value.split('/').some((part) => !part || part === '.' || part === '..' || part.toLowerCase() === '.git')) {
    throw new Error(`Invalid relative source path: ${JSON.stringify(value)}`);
  }
  return value;
}

export function isContained({ root, path }: { root: string; path: string }): boolean {
  const name = relative(resolve(root), resolve(path));
  return name === '' || (!isAbsolute(name) && name !== '..' && !name.startsWith(`..${sep}`));
}

export function containedPath({ root, relative: name }: { root: string; relative: string }): string {
  return join(resolve(root), relativePath({ value: name }));
}

export async function optionalStat({ path }: { path: string }) {
  try {
    return await lstat(path);
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return undefined;
    }
    throw error;
  }
}

/** Check every parent before use; resolving first would conceal linked roots. */
export async function directory({ path, create }: { path: string; create?: boolean }): Promise<string> {
  const absolute = resolve(path);
  const parents: string[] = [];
  let current = absolute;
  for (;;) {
    parents.push(current);
    const parent = dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  for (const parent of parents.reverse()) {
    let mode = await optionalStat({ path: parent });
    if (mode === undefined && create) {
      await mkdir(parent).catch(async (error: unknown) => {
        if ((await optionalStat({ path: parent }))?.isDirectory()) {
          return;
        }
        throw error;
      });
      mode = await optionalStat({ path: parent });
    }
    if (!mode?.isDirectory() || mode.isSymbolicLink()) {
      throw new Error(`Linked, missing or non-directory source path: ${parent}`);
    }
  }
  return absolute;
}

export async function regularFile({ path }: { path: string }): Promise<string> {
  const absolute = resolve(path);
  await directory({ path: dirname(absolute) });
  if (!(await optionalStat({ path: absolute }))?.isFile()) {
    throw new Error(`Linked, missing or non-regular source file: ${absolute}`);
  }
  return absolute;
}

export async function fileIdentity({ path }: { path: string }): Promise<FileIdentity> {
  const absolute = await regularFile({ path });
  const file = await open(absolute, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const before = await file.stat({ bigint: true });
    if (!before.isFile()) {
      throw new Error(`Non-regular opened source file: ${absolute}`);
    }
    const hash = createHash('sha256');
    let bytes = 0;
    for await (const value of file.createReadStream({ autoClose: false, start: 0 })) {
      const chunk: unknown = value;
      if (!Buffer.isBuffer(chunk)) {
        throw new Error('Expected a binary file stream');
      }
      bytes += chunk.length;
      hash.update(chunk);
    }
    const after = await file.stat({ bigint: true });
    if (before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs || BigInt(bytes) !== after.size) {
      throw new Error(`Source file changed while reading: ${absolute}`);
    }
    return { bytes, sha256: hash.digest('hex') };
  } finally {
    await file.close();
  }
}

export async function fileIdentities({ root, exclude }: { root: string; exclude?: readonly string[] }): Promise<Record<string, SourceIdentity>> {
  await directory({ path: root });
  const exclusions = exclude ?? [];
  for (const value of exclusions) {
    relativePath({ value });
  }
  const files = new Map<string, SourceIdentity>();
  async function visit({ path, prefix }: { path: string; prefix: string }): Promise<void> {
    for (const name of (await readdir(path)).sort()) {
      if (name.toLowerCase() === '.git') {
        continue;
      }
      const relative = prefix + name;
      relativePath({ value: relative });
      if (exclusions.some((part) => relative === part || relative.startsWith(`${part}/`))) {
        continue;
      }
      const file = join(path, name);
      const mode = await lstat(file);
      if (mode.isSymbolicLink()) {
        files.set(relative, { symlink: await readlink(file) });
      } else if (mode.isDirectory()) {
        await visit({ path: file, prefix: `${relative}/` });
      } else if (mode.isFile()) {
        files.set(relative, await fileIdentity({ path: file }));
      } else {
        throw new Error(`Non-regular captured source entry: ${file}`);
      }
    }
  }
  await visit({ path: root, prefix: '' });
  return Object.fromEntries(files);
}

/** Observe a regular-file-only boundary directly, without reparsing typed values. */
export async function regularFileIdentities({ root, exclude }: { root: string; exclude?: readonly string[] }): Promise<Record<string, FileIdentity>> {
  const files = await fileIdentities({ root, ...(exclude === undefined ? {} : { exclude }) });
  const regular = new Map<string, FileIdentity>();
  for (const [name, identity] of Object.entries(files)) {
    if ('symlink' in identity) {
      throw new Error(`Expected a regular captured file: ${name}`);
    }
    regular.set(name, identity);
  }
  return Object.fromEntries(regular);
}

export function treeIdentity({ files }: { files: Readonly<Record<string, SourceIdentity>> }): { files: number; sha256: string } {
  const sorted = Object.fromEntries(Object.entries(files).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0));
  return { files: Object.keys(sorted).length, sha256: createHash('sha256').update(JSON.stringify(sorted)).digest('hex') };
}
