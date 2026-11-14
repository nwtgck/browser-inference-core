import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { cp, open, readdir, lstat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { directory, fileIdentities, fileIdentity, regularFile, optionalStat, relativePath, type FileIdentity } from '#tools/shared/files';
import { parseJson } from '#tools/shared/json';
import { runProcess } from '#tools/shared/process';
import { inventory } from '#tools/shared/contracts';
import { decodeNpmPack, type FileEntry } from '#dto/shared';

export const maxArtifactBytes = 100 * 1024 ** 2;

export function bytesIdentity({ data }: { data: Uint8Array }): FileIdentity {
  return { bytes: data.byteLength, sha256: createHash('sha256').update(data).digest('hex') };
}

export async function readMetadataBytes({ path }: { path: string }): Promise<Buffer> {
  return readRegularBytes({ path, maxBytes: maxArtifactBytes - 1 });
}

/** Bound allocation and verify the same opened regular file throughout the read. */
export async function readRegularBytes({ path, maxBytes }: { path: string; maxBytes: number }): Promise<Buffer> {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
    throw new Error('Invalid bounded file size');
  }
  await using file = await open(await regularFile({ path }), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  const before = await file.stat({ bigint: true });
  if (!before.isFile()) {
    throw new Error(`Non-regular opened package input: ${path}`);
  }
  if (before.size > BigInt(maxBytes)) {
    throw new Error(`Oversized package metadata: ${path}`);
  }
  const chunks: Buffer[] = [];
  let bytes = 0;
  for (;;) {
    const chunk = Buffer.alloc(Math.min(64 * 1024, maxBytes + 1 - bytes));
    const { bytesRead } = await file.read(chunk, 0, chunk.length, null);
    if (!bytesRead) {
      break;
    }
    bytes += bytesRead;
    if (bytes > maxBytes) {
      throw new Error(`Oversized package metadata: ${path}`);
    }
    chunks.push(chunk.subarray(0, bytesRead));
  }
  const after = await file.stat({ bigint: true });
  if (before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) {
    throw new Error(`Package input changed while reading: ${path}`);
  }
  return Buffer.concat(chunks, bytes);
}

export async function readMetadata<T>({ path, decode }: { path: string; decode: ({ input }: { input: unknown }) => T }): Promise<T> {
  return parseJson({ text: (await readMetadataBytes({ path })).toString('utf8'), decode });
}

export async function packageFiles({ root, exclude }: { root: string; exclude: readonly string[] }): Promise<Map<string, FileEntry>> {
  const values = await fileIdentities({ root, exclude });
  const entries = Object.entries(values).map(([path, value]) => {
    if ('symlink' in value) {
      throw new Error(`Linked package input: ${path}`);
    }
    if (value.bytes >= maxArtifactBytes) {
      throw new Error(`Single-file artifact size limit exceeded: ${path}`);
    }
    return { path, ...value };
  });
  const result = inventory({ entries });
  // Detect directory aliases as well as colliding file paths.
  const folded = new Set<string>();
  async function visit({ path, prefix }: { path: string; prefix: string }): Promise<void> {
    for (const name of await readdir(path)) {
      const relative = prefix + name;
      if (exclude.some(part => relative === part || relative.startsWith(part + '/'))) {
        continue;
      }
      relativePath({ value: relative });
      const key = relative.toLowerCase();
      if (folded.has(key)) {
        throw new Error(`Case-colliding package path: ${relative}`);
      }
      folded.add(key);
      if ((await lstat(join(path, name))).isDirectory()) {
        await visit({ path: join(path, name), prefix: relative + '/' });
      }
    }
  }
  await visit({ path: root, prefix: '' });
  return result;
}

export function assertSame({ actual, expected, message }: { actual: unknown; expected: unknown; message: string }): void {
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error(message);
  }
}

export async function copyRegular({ source, destination }: { source: string; destination: string }): Promise<void> {
  await copyBoundFile({ source, destination, identity: await fileIdentity({ path: source }) });
}

async function copyBoundFile({ source, destination, identity }: { source: string; destination: string; identity: FileIdentity }): Promise<void> {
  const permissions = (await lstat(source)).mode & 0o777;
  await directory({ path: dirname(destination), create: true });
  if (await optionalStat({ path: destination })) {
    await regularFile({ path: destination });
  }
  await cp(await regularFile({ path: source }), destination, { dereference: false, preserveTimestamps: true });
  assertSame({ actual: await fileIdentity({ path: destination }), expected: { bytes: identity.bytes, sha256: identity.sha256 }, message: `Copied input changed: ${source}` });
  if (((await lstat(destination)).mode & 0o777) !== permissions) {
    throw new Error(`Copied input permissions changed: ${source}`);
  }
}

export async function copyRegularTree({ source, destination }: { source: string; destination: string }): Promise<void> {
  const before = await packageFiles({ root: source, exclude: [] });
  await copyBoundTree({ source, destination, files: before });
}

/** A validated inventory crosses a copy boundary without rehashing its source. */
export async function copyBoundTree({ source, destination, files }: { source: string; destination: string; files: ReadonlyMap<string, FileEntry> }): Promise<void> {
  await directory({ path: source });
  await directory({ path: destination, create: true });
  for (const [name, identity] of files) {
    relativePath({ value: name });
    await copyBoundFile({ source: join(source, name), destination: join(destination, name), identity });
  }
  assertSame({ actual: [...await listPackagePaths({ root: destination, exclude: [] })].sort(), expected: [...files.keys()].sort(), message: 'Copied package tree changed' });
}

export async function checkNpmPack({ directory: root, files }: { directory: string; files: ReadonlyMap<string, FileEntry> }): Promise<void> {
  const result = await runProcess({ command: 'npm', args: ['pack', '--dry-run', '--json', '--ignore-scripts', '--offline'], cwd: root, env: { kind: 'inherit', removePrefixes: [], overrides: { npm_config_ignore_scripts: 'true', npm_config_offline: 'true', npm_config_audit: 'false', npm_config_fund: 'false' } }, capture: 'file' });
  const packed = parseJson({ text: result.stdout.toString('utf8'), decode: decodeNpmPack });
  if (packed.length !== 1) {
    throw new Error('npm pack returned more than one package');
  }
  const entries = packed[0]?.files;
  if (!entries || new Set(entries.map(entry => entry.path)).size !== entries.length) {
    throw new Error('Duplicate npm pack entry');
  }
  assertSame({ actual: entries.map(entry => entry.path).sort(), expected: [...files.keys()].sort(), message: 'npm pack tree mismatch' });
  for (const entry of entries) {
    if (entry.size !== files.get(entry.path)?.bytes) {
      throw new Error(`npm pack size mismatch: ${entry.path}`);
    }
  }
}

export async function readBoundFile({ root, entry }: { root: string; entry: FileEntry }): Promise<Buffer> {
  const raw = await readRegularBytes({ path: join(root, entry.path), maxBytes: Math.min(entry.bytes, maxArtifactBytes - 1) });
  assertSame({ actual: bytesIdentity({ data: raw }), expected: { bytes: entry.bytes, sha256: entry.sha256 }, message: `Artifact hash/size mismatch: ${entry.path}` });
  return raw;
}

export async function requireFiles({ files, names }: { files: ReadonlyMap<string, FileEntry>; names: readonly string[] }): Promise<void> {
  for (const name of names) {
    if (!files.has(name)) {
      throw new Error(`Missing runtime artifact: ${name}`);
    }
  }
}

/** Inspect names and file kinds without hashing the payload for a second owner. */
export async function listPackagePaths({ root, exclude }: { root: string; exclude: readonly string[] }): Promise<Set<string>> {
  await directory({ path: root });
  const files = new Set<string>();
  const folded = new Set<string>();
  async function visit({ path, prefix }: { path: string; prefix: string }): Promise<void> {
    for (const name of await readdir(path)) {
      const relative = prefix + name;
      if (exclude.some(part => relative === part || relative.startsWith(part + '/'))) {
        continue;
      }
      relativePath({ value: relative });
      const key = relative.toLowerCase();
      if (folded.has(key)) {
        throw new Error(`Case-colliding package entry: ${relative}`);
      }
      folded.add(key);
      const mode = await lstat(join(path, name));
      if (mode.isDirectory()) {
        await visit({ path: join(path, name), prefix: relative + '/' });
      } else if (mode.isFile()) {
        files.add(relative);
      } else {
        throw new Error(`Linked or non-regular package entry: ${relative}`);
      }
    }
  }
  await visit({ path: root, prefix: '' });
  return files;
}
