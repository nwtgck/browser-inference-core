import { constants } from 'node:fs';
import { chmod, copyFile, lstat, mkdir, open, readdir, readlink, symlink } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import type { SourceIdentity } from '#dto/shared';
import { containedPath, directory, fileIdentities, isContained, optionalStat, regularFile, relativePath } from '#tools/shared/files';
import { withOutputDirectories } from '#tools/shared/output';
import { runProcess, startProcess } from '#tools/shared/process';
import { BinaryReader } from '#tools/shared/streams';

const objectId = /^[a-f0-9]{40}$/u;
const ignoredNames = new Set(['.git', 'build', '.github']);

export function gitEnvironment({ ceiling }: { ceiling?: string | undefined }): NodeJS.ProcessEnv {
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !name.startsWith('GIT_')));
  return { ...env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null', GIT_TERMINAL_PROMPT: '0', GIT_NO_LAZY_FETCH: '1', GIT_OPTIONAL_LOCKS: '0', GIT_NO_REPLACE_OBJECTS: '1', ...(ceiling === undefined ? {} : { GIT_CEILING_DIRECTORIES: ceiling }) };
}

const gitOptions = ['-c', 'protocol.allow=never', '-c', 'core.fsmonitor=false'];

export async function gitRead({ sourceRoot, args }: { sourceRoot: string; args: readonly string[] }): Promise<Buffer> {
  return (await runProcess({ command: 'git', args: [...gitOptions, '-C', sourceRoot, ...args], cwd: sourceRoot, env: gitEnvironment({ ceiling: undefined }) })).stdout;
}

async function repositoryRoot({ sourceRoot }: { sourceRoot: string }): Promise<string> {
  const root = await directory({ path: sourceRoot });
  const actual = (await gitRead({ sourceRoot: root, args: ['rev-parse', '--show-toplevel'] })).toString('utf8').trimEnd();
  if (resolve(actual) !== root) {
    throw new Error(`Initialize the exact source checkout: ${root}`);
  }
  return root;
}

async function checkCommit({ sourceRoot, revision }: { sourceRoot: string; revision: string }): Promise<void> {
  if (!objectId.test(revision)) {
    throw new Error('Expected a complete pinned source commit');
  }
  if ((await gitRead({ sourceRoot, args: ['cat-file', '-t', revision] })).toString('utf8').trim() !== 'commit') {
    throw new Error('Source identity is not a commit');
  }
}

export async function selectedGitlinks({ sourceRoot, paths, revision }: { sourceRoot: string; paths: Readonly<Record<string, string>>; revision: string | undefined }): Promise<Record<string, string>> {
  sourceRoot = await repositoryRoot({ sourceRoot });
  if (revision !== undefined) {
    await checkCommit({ sourceRoot, revision });
  }
  const selections = Object.entries(paths);
  if (!selections.length || new Set(Object.values(paths)).size !== selections.length) {
    throw new Error('Source roles must select distinct gitlinks');
  }
  const result = new Map<string, string>();
  for (const [role, relative] of selections) {
    relativePath({ value: relative });
    const args = revision === undefined ? ['ls-files', '--stage', '-z', '--', relative] : ['ls-tree', '-z', revision, '--', relative];
    const raw = (await gitRead({ sourceRoot, args })).toString('utf8');
    const rows = raw.split('\0').filter(Boolean);
    const pattern = revision === undefined ? /^160000 ([a-f0-9]{40}) 0\t(.+)$/u : /^160000 commit ([a-f0-9]{40})\t(.+)$/u;
    const match = rows.length === 1 ? pattern.exec(rows[0] ?? '') : null;
    if (!match || match[2] !== relative || !match[1]) {
      throw new Error(`Missing or ambiguous committed source gitlink: ${role}: ${relative}`);
    }
    result.set(role, match[1]);
  }
  return Object.fromEntries(result);
}

type Entry = { mode: string; kind: 'blob' | 'commit'; oid: string; name: string };

async function entries({ sourceRoot, revision, include, exclude }: { sourceRoot: string; revision: string; include: readonly string[] | undefined; exclude: readonly string[] }): Promise<Entry[]> {
  await checkCommit({ sourceRoot, revision });
  for (const value of [...include ?? [], ...exclude]) {
    relativePath({ value });
  }
  if (include?.length === 0 || include && new Set(include).size !== include.length) {
    throw new Error('Expected nonempty distinct source selections');
  }
  const selected = new Set<string>();
  const result: Entry[] = [];
  const raw = await gitRead({ sourceRoot, args: ['ls-tree', '-rz', '--full-tree', revision] });
  for (const row of new TextDecoder('utf-8', { fatal: true }).decode(raw).split('\0').filter(Boolean)) {
    const match = /^(\d+) (blob|commit) ([a-f0-9]{40})\t([\s\S]+)$/u.exec(row);
    if (!match) {
      throw new Error('Unsupported source tree entry');
    }
    const [, mode, kind, oid, name] = match;
    if (mode === undefined || oid === undefined || name === undefined || (kind !== 'blob' && kind !== 'commit')) {
      throw new Error('Invalid source tree entry');
    }
    relativePath({ value: name });
    const matching = include?.filter((prefix) => name === prefix || name.startsWith(`${prefix}/`));
    if (matching && !matching.length) {
      continue;
    }
    if (matching) {
      for (const value of matching) {
        selected.add(value);
      }
    }
    if (exclude.some((prefix) => name === prefix || name.startsWith(`${prefix}/`)) || name.split('/').some((part) => ignoredNames.has(part))) {
      continue;
    }
    if (!((kind === 'blob' && ['100644', '100755', '120000'].includes(mode)) || kind === 'commit' && mode === '160000')) {
      throw new Error(`Unsupported source tree entry: ${name}: ${mode} ${kind}`);
    }
    result.push({ mode, kind, oid, name });
  }
  if (include?.some((name) => !selected.has(name))) {
    throw new Error(`Missing committed source selection: ${include.filter((name) => !selected.has(name)).join(', ')}`);
  }
  return result;
}

async function resolveSourceLink({ path }: { path: string }): Promise<string> {
  let pending = resolve(path).split('/').filter(Boolean);
  let current = '/';
  let links = 0;
  while (pending.length) {
    const part = pending.shift();
    if (part === undefined) {
      break;
    }
    if (part === '.' || part === '') {
      continue;
    }
    if (part === '..') {
      current = dirname(current);
      continue;
    }
    current = resolve(current, part);
    const mode = await optionalStat({ path: current });
    if (mode?.isSymbolicLink()) {
      if (++links > 40) {
        throw new Error(`Invalid cyclic source link: ${path}`);
      }
      const target = await readlink(current);
      pending = [...target.split('/'), ...pending];
      current = isAbsolute(target) ? '/' : dirname(current);
    }
  }
  return current;
}

async function validateLinks({ root }: { root: string }): Promise<void> {
  async function visit({ path }: { path: string }): Promise<void> {
    for (const name of await readdir(path)) {
      const file = join(path, name);
      const mode = await lstat(file);
      if (mode.isSymbolicLink()) {
        if (isAbsolute(await readlink(file)) || !isContained({ root, path: await resolveSourceLink({ path: file }) })) {
          throw new Error(`Source link escapes its tree: ${file}`);
        }
      } else if (mode.isDirectory()) {
        await visit({ path: file });
      }
    }
  }
  await visit({ path: root });
}

async function exportTree({ sourceRoot, revision, destination, include, exclude }: { sourceRoot: string; revision: string; destination: string; include: readonly string[] | undefined; exclude: readonly string[] }): Promise<void> {
  sourceRoot = await repositoryRoot({ sourceRoot });
  const selected = await entries({ sourceRoot, revision, include, exclude });
  const child = startProcess({ command: 'git', args: [...gitOptions, '-C', sourceRoot, 'cat-file', '--batch'], cwd: sourceRoot, env: gitEnvironment({ ceiling: undefined }) });
  const reader = new BinaryReader({ stream: child.stdout });
  try {
    for (const entry of selected) {
      const target = containedPath({ root: destination, relative: entry.name });
      await directory({ path: dirname(target), create: true });
      switch (entry.kind) {
      case 'commit':
        await mkdir(target);
        await exportTree({ sourceRoot: join(sourceRoot, entry.name), revision: entry.oid, destination: target, include: undefined, exclude: [] });
        continue;
      case 'blob':
        break;
      default: {
        const _ex: never = entry.kind;
        throw new Error(`Unhandled source tree entry kind: ${_ex}`);
      }
      }
      child.stdin.write(`${entry.oid}\n`);
      const header = /^(\w+) blob (\d+)$/u.exec(await reader.line({ limit: 256 }));
      if (!header || header[1] !== entry.oid || header[2] === undefined) {
        throw new Error(`Missing or invalid local source blob: ${entry.name}`);
      }
      let remaining = Number(header[2]);
      if (!Number.isSafeInteger(remaining)) {
        throw new Error(`Invalid source blob size: ${entry.name}`);
      }
      if (entry.mode === '120000') {
        if (remaining === 0 || remaining > 65536) {
          throw new Error(`Invalid source link: ${entry.name}`);
        }
        const chunks: Buffer[] = [];
        while (remaining) {
          const chunk = await reader.take({ count: remaining });
          chunks.push(chunk);
          remaining -= chunk.length;
        }
        const link = new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks));
        if (link.includes('\0') || isAbsolute(link)) {
          throw new Error(`Source link escapes its tree: ${entry.name}`);
        }
        await symlink(link, target);
      } else {
        const file = await open(target, 'wx', entry.mode === '100755' ? 0o755 : 0o644);
        try {
          while (remaining) {
            const chunk = await reader.take({ count: Math.min(remaining, 1024 * 1024) });
            await file.writeFile(chunk);
            remaining -= chunk.length;
          }
        } finally {
          await file.close();
        }
        await chmod(target, entry.mode === '100755' ? 0o755 : 0o644);
      }
      if ((await reader.take({ count: 1 }))[0] !== 10) {
        throw new Error(`Invalid source object boundary: ${entry.name}`);
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
  await validateLinks({ root: destination });
}

export async function exportCommit({ sourceRoot, revision, destination, include, exclude }: { sourceRoot: string; revision: string; destination: string; include?: readonly string[]; exclude?: readonly string[] }): Promise<void> {
  // Selection errors must be reported before even a temporary output is installed.
  await repositoryRoot({ sourceRoot });
  const exclusions = exclude ?? [];
  await entries({ sourceRoot, revision, include, exclude: exclusions });
  await withOutputDirectories({
    destinations: [destination],
    replace: false,
    prepare: async ({ paths }) => {
      const output = paths[0];
      if (!output) {
        throw new Error('Missing private source output');
      }
      await exportTree({ sourceRoot, revision, destination: output, include, exclude: exclusions });
    },
  });
}

async function copyLocal({ sourceRoot, destination, include, exclude }: { sourceRoot: string; destination: string; include: readonly string[]; exclude: readonly string[] }): Promise<void> {
  await repositoryRoot({ sourceRoot });
  if (!include.length || new Set(include).size !== include.length) {
    throw new Error('Expected nonempty distinct source selections');
  }
  for (const value of [...include, ...exclude]) {
    relativePath({ value });
  }
  const raw = await gitRead({ sourceRoot, args: ['ls-files', '--cached', '--others', '--exclude-standard', '--deduplicate', '-z'] });
  const names = new TextDecoder('utf-8', { fatal: true }).decode(raw).split('\0').filter(Boolean);
  for (const name of names.sort()) {
    relativePath({ value: name });
    if (!include.some((prefix) => name === prefix || name.startsWith(`${prefix}/`)) || exclude.some((prefix) => name === prefix || name.startsWith(`${prefix}/`)) || name.split('/').some((part) => ignoredNames.has(part))) {
      continue;
    }
    const source = containedPath({ root: sourceRoot, relative: name });
    const mode = await optionalStat({ path: source });
    if (!mode) {
      continue;
    }
    const target = containedPath({ root: destination, relative: name });
    await directory({ path: dirname(target), create: true });
    if (mode.isSymbolicLink()) {
      await directory({ path: dirname(source) });
      await symlink(await readlink(source), target);
    } else if (mode.isFile()) {
      await copyFile(await regularFile({ path: source }), target, constants.COPYFILE_FICLONE);
    } else {
      throw new Error(`Local input must be an ordinary file; select upstream gitlinks separately: ${name}`);
    }
  }
  await validateLinks({ root: destination });
}

export type SourceMode = { kind: 'release'; expectedCommit: string } | { kind: 'local' };
export type SourceState =
  | { readonly mode: 'release'; readonly checkoutRoot: string; readonly head: string; readonly publishable: true }
  | { readonly mode: 'local'; readonly checkoutRoot: string; readonly head: string; readonly publishable: false };
export type SourceSnapshot = { compileRoot: string; files: Record<string, SourceIdentity>; source: SourceState };

async function checkSource({ source }: { source: SourceState }): Promise<void> {
  const head = (await gitRead({ sourceRoot: source.checkoutRoot, args: ['rev-parse', '--verify', 'HEAD'] })).toString('utf8').trim();
  if (head !== source.head) {
    throw new Error('Checkout HEAD changed after source selection');
  }
  if (source.publishable && (await gitRead({ sourceRoot: source.checkoutRoot, args: ['status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'] })).length) {
    throw new Error('Release source checkout is dirty');
  }
}

/** The runtime obtains checkoutRoot from its own static configuration owner. */
export async function observeSource({ checkoutRoot, mode }: { checkoutRoot: string; mode: SourceMode }): Promise<SourceState> {
  checkoutRoot = await repositoryRoot({ sourceRoot: checkoutRoot });
  const head = (await gitRead({ sourceRoot: checkoutRoot, args: ['rev-parse', '--verify', 'HEAD'] })).toString('utf8').trim();
  await checkCommit({ sourceRoot: checkoutRoot, revision: head });
  let source: SourceState;
  switch (mode.kind) {
  case 'release':
    if (!objectId.test(mode.expectedCommit) || head !== mode.expectedCommit) {
      throw new Error('Checkout HEAD differs from the expected release commit');
    }
    source = { mode: 'release', checkoutRoot, head, publishable: true };
    break;
  case 'local':
    source = { mode: 'local', checkoutRoot, head, publishable: false };
    break;
  default: {
    const _ex: never = mode;
    throw new Error(`Unhandled source mode: ${_ex}`);
  }
  }
  await checkSource({ source });
  return source;
}

/** Stabilize owned compiler inputs; executable tools and dependencies stay in the checkout. */
export async function snapshotCompileInputs({ source, destination, include, exclude, replace }: { source: SourceState; destination: string; include: readonly string[]; exclude: readonly string[]; replace: boolean }): Promise<SourceSnapshot> {
  destination = resolve(destination);
  const checkoutRoot = await repositoryRoot({ sourceRoot: source.checkoutRoot });
  const ignoredDestination = isContained({ root: checkoutRoot, path: destination }) && destination.slice(checkoutRoot.length + 1).split('/').some((name) => ignoredNames.has(name));
  const excludedDestination = exclude.some((name) => isContained({ root: containedPath({ root: checkoutRoot, relative: name }), path: destination }));
  if (include.some((name) => {
    const input = containedPath({ root: checkoutRoot, relative: name });
    return isContained({ root: destination, path: input }) || !ignoredDestination && !excludedDestination && isContained({ root: input, path: destination });
  })) {
    throw new Error('Compile snapshot must not overlap selected inputs');
  }
  await checkSource({ source });
  return withOutputDirectories({
    destinations: [destination],
    replace,
    preserveUnchanged: true,
    prepare: async ({ paths }) => {
      const work = paths[0];
      if (!work) {
        throw new Error('Missing compile-input snapshot');
      }
      switch (source.mode) {
      case 'release':
        await exportTree({ sourceRoot: checkoutRoot, revision: source.head, destination: work, include, exclude });
        break;
      case 'local':
        await copyLocal({ sourceRoot: checkoutRoot, destination: work, include, exclude });
        break;
      default: {
        const _ex: never = source;
        throw new Error(`Unhandled source snapshot mode: ${_ex}`);
      }
      }
      const files = await fileIdentities({ root: work });
      await checkSource({ source });
      return { compileRoot: destination, files, source };
    },
  });
}
