import { constants } from 'node:fs';
import { copyFile, lstat, mkdtemp, mkdir, open, readdir, readlink, rename, rm, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { directory, isContained, optionalStat } from './files.js';
import { runProcess } from './process.js';

type DirectoryIdentity = { dev: number; ino: number };
type Output = { destination: string; temporary: string; original: DirectoryIdentity | undefined; moved: boolean; installed: DirectoryIdentity | undefined; reuse: boolean; retain: boolean };

async function identity({ path }: { path: string }): Promise<DirectoryIdentity | undefined> {
  const mode = await optionalStat({ path });
  if (!mode) {
    return undefined;
  }
  await directory({ path });
  return { dev: mode.dev, ino: mode.ino };
}

function sameIdentity({ left, right }: { left: DirectoryIdentity | undefined; right: DirectoryIdentity | undefined }): boolean {
  return left?.dev === right?.dev && left?.ino === right?.ino;
}

async function checkOriginal({ output }: { output: Output }): Promise<void> {
  if (!sameIdentity({ left: await identity({ path: output.destination }), right: output.original })) {
    throw new Error(`Output destination changed, appeared or disappeared during preparation: ${output.destination}`);
  }
}

/** Ordinary failures roll back the whole group. This is not a concurrent-reader transaction. */
export async function withOutputDirectories<T>({ destinations, replace, preserveUnchanged, prepare }: { destinations: readonly string[]; replace: boolean; preserveUnchanged?: boolean; prepare: ({ paths }: { paths: readonly string[] }) => Promise<T> }): Promise<T> {
  const paths = destinations.map((path) => resolve(path));
  if (!paths.length || paths.some((path, index) => paths.slice(index + 1).some((other) => isContained({ root: path, path: other }) || isContained({ root: other, path })))) {
    throw new Error('Output destinations must be nonempty, distinct and non-overlapping');
  }
  const outputs: Output[] = [];
  try {
    for (const destination of paths) {
      await directory({ path: dirname(destination), create: true });
      const original = await identity({ path: destination });
      if (original && !replace) {
        throw new Error(`Output destination must not already exist: ${destination}`);
      }
      const temporary = await mkdtemp(join(dirname(destination), 'bicore-output-'));
      outputs.push({ destination, temporary, original, moved: false, installed: undefined, reuse: false, retain: false });
      await mkdir(join(temporary, 'prepared'), { mode: 0o700 });
    }
    const value = await prepare({ paths: outputs.map((output) => join(output.temporary, 'prepared')) });
    for (const output of outputs) {
      await checkOriginal({ output });
    }
    if (preserveUnchanged) {
      for (const output of outputs) {
        if (output.original) {
          output.reuse = await preserveUnchangedTree({ work: join(output.temporary, 'prepared'), previous: output.destination });
        }
      }
    }
    for (const output of outputs) {
      await checkOriginal({ output });
    }
    try {
      for (const output of outputs) {
        await checkOriginal({ output });
        if (output.reuse) {
          continue;
        }
        const prepared = join(output.temporary, 'prepared');
        const candidate = await identity({ path: prepared });
        if (!candidate) {
          throw new Error(`Prepared output disappeared: ${prepared}`);
        }
        if (output.original) {
          await rename(output.destination, join(output.temporary, 'previous'));
          output.moved = true;
        }
        await rename(prepared, output.destination);
        output.installed = candidate;
      }
    } catch (error) {
      const retained: string[] = [];
      for (const output of [...outputs].reverse()) {
        if (!output.installed && !output.moved) {
          continue;
        }
        try {
          if (output.installed) {
            if (!sameIdentity({ left: await identity({ path: output.destination }), right: output.installed })) {
              throw new Error('Activated output changed before recovery');
            }
            await rename(output.destination, join(output.temporary, 'prepared'));
          }
          if (output.moved) {
            if (await optionalStat({ path: output.destination })) {
              throw new Error('Output destination appeared before recovery');
            }
            await rename(join(output.temporary, 'previous'), output.destination);
          }
        } catch {
          output.retain = true;
          retained.push(join(output.temporary, 'previous'));
        }
      }
      if (retained.length) {
        throw new Error(`Output activation and recovery failed; previous output retained at ${retained.join(', ')}`, { cause: error });
      }
      throw error;
    }
    return value;
  } finally {
    for (const output of outputs) {
      if (!output.retain) {
        await rm(output.temporary, { recursive: true, force: true });
      }
    }
  }
}

export async function sameFileBytes({ left, right }: { left: string; right: string }): Promise<boolean> {
  if ((await stat(left)).size !== (await stat(right)).size) {
    return false;
  }
  await using a = await open(left, 'r');
  await using b = await open(right, 'r');
  const x = Buffer.allocUnsafe(1024 * 1024);
  const y = Buffer.allocUnsafe(x.length);
  for (;;) {
    const read = await a.read(x, 0, x.length, null);
    if (!read.bytesRead) {
      return (await b.read(y, 0, 1, null)).bytesRead === 0;
    }
    let remaining = read.bytesRead;
    let offset = 0;
    while (remaining) {
      const next = await b.read(y, offset, remaining, null);
      if (!next.bytesRead) {
        return false;
      }
      offset += next.bytesRead;
      remaining -= next.bytesRead;
    }
    if (!x.subarray(0, read.bytesRead).equals(y.subarray(0, read.bytesRead))) {
      return false;
    }
  }
}

async function treeEntries({ root }: { root: string }): Promise<Map<string, string>> {
  const result = new Map<string, string>();
  async function visit({ path, prefix }: { path: string; prefix: string }): Promise<void> {
    for (const name of (await readdir(path)).sort()) {
      const relative = prefix + name;
      const file = join(path, name);
      result.set(relative, file);
      if ((await lstat(file)).isDirectory()) {
        await visit({ path: file, prefix: `${relative}/` });
      }
    }
  }
  await visit({ path: root, prefix: '' });
  return result;
}

export async function preserveUnchangedTree({ work, previous }: { work: string; previous: string }): Promise<boolean> {
  const fresh = await treeEntries({ root: work });
  const old = await treeEntries({ root: previous });
  let identical = fresh.size === old.size && (await lstat(work)).mode === (await lstat(previous)).mode;
  const equalFiles: string[] = [];
  for (const [name, path] of fresh) {
    const earlier = old.get(name);
    if (!earlier) {
      identical = false;
      continue;
    }
    const before = await lstat(earlier);
    const after = await lstat(path);
    let equal = before.mode === after.mode;
    if (after.isFile()) {
      equal = equal && await sameFileBytes({ left: path, right: earlier });
    } else if (after.isSymbolicLink()) {
      equal = equal && await readlink(path) === await readlink(earlier);
    } else if (!after.isDirectory()) {
      equal = false;
    }
    if (equal && after.isFile()) {
      equalFiles.push(name);
    } else if (!equal) {
      identical = false;
    }
  }
  if (!identical) {
    // Node's utimes rounds nanoseconds. GNU cp preserves the actual timestamps
    // without copying bytes, linking previous outputs, or spawning per file.
    let batch: string[] = [];
    let bytes = 0;
    async function preserve(): Promise<void> {
      if (!batch.length) {
        return;
      }
      await runProcess({ command: 'cp', args: ['--attributes-only', '--preserve=timestamps', '--parents', '--no-dereference', '--', ...batch, work], cwd: previous });
      batch = [];
      bytes = 0;
    }
    for (const name of equalFiles) {
      if (bytes + Buffer.byteLength(name) > 32 * 1024) {
        await preserve();
      }
      batch.push(name);
      bytes += Buffer.byteLength(name) + 1;
    }
    await preserve();
  }
  return identical;
}

/** Publish a completed file without touching an equal previous file's mtime. */
export async function replaceFileUnchanged({ candidate, output }: { candidate: string; output: string }): Promise<void> {
  await directory({ path: dirname(output), create: true });
  const old = await optionalStat({ path: output });
  if (old && !old.isFile()) {
    throw new Error(`Linked or non-regular output: ${output}`);
  }
  if (old && await sameFileBytes({ left: candidate, right: output })) {
    return;
  }
  const temporary = await mkdtemp(join(dirname(output), 'bicore-file-'));
  try {
    const file = join(temporary, 'prepared');
    await copyFile(candidate, file, constants.COPYFILE_FICLONE);
    await rename(file, output);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}
