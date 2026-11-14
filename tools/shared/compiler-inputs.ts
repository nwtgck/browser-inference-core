import { readFile, readdir } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import { decodeCompilationDatabase, type CompileEntry, type CompilerInputs, type FileIdentity } from '#dto/shared';
import { fileIdentity, isContained, optionalStat, regularFile } from './files.js';
import { parseJson } from './json.js';
import { splitCommandLine } from './process.js';

export type { CompileEntry, CompilerInputs } from '#dto/shared';

export function compileArguments({ entry }: { entry: CompileEntry }): string[] {
  if (entry.arguments !== undefined) {
    return entry.arguments;
  }
  if (entry.command === undefined) {
    throw new Error('Compile entry requires arguments or command');
  }
  return splitCommandLine({ command: entry.command });
}

/** Record actual compdb/CMake target facts; do not reconstruct CMake source lists. */
export async function recordCompilerInputs({ buildRoot, roots }: { buildRoot: string; roots: Readonly<Record<string, string>> }): Promise<CompilerInputs> {
  const owned = Object.entries(roots).map(([label, root]) => ({ label, root: resolve(root) })).sort((a, b) => b.root.length - a.root.length);
  function identify({ path }: { path: string }): string {
    const owner = owned.find(({ root }) => isContained({ root, path }));
    if (!owner) {
      throw new Error(`Compiled input is outside the selected build roots: ${path}`);
    }
    return `${owner.label}/${relative(owner.root, path).split('\\').join('/')}`;
  }
  const entries = parseJson({ text: await readFile(await regularFile({ path: join(buildRoot, 'compile_commands.json') }), 'utf8'), decode: decodeCompilationDatabase });
  const translationUnits = new Map<string, FileIdentity>();
  const compiledPaths = new Set<string>();
  const identities = new Map<string, Promise<FileIdentity>>();
  function observe({ path }: { path: string }): Promise<FileIdentity> {
    let value = identities.get(path);
    if (value === undefined) {
      value = fileIdentity({ path });
      identities.set(path, value);
    }
    return value;
  }
  for (const entry of entries) {
    const path = resolve(entry.directory, entry.file);
    compiledPaths.add(path);
    translationUnits.set(identify({ path }), await observe({ path }));
  }
  const targetSources = new Map<string, Record<string, FileIdentity>>();
  for (const name of (await readdir(join(buildRoot, 'target-sources'))).sort()) {
    if (!name.endsWith('.txt')) {
      continue;
    }
    const [sourceDirectory, ...files] = (await readFile(await regularFile({ path: join(buildRoot, 'target-sources', name) }), 'utf8')).trimEnd().split('\n');
    if (!sourceDirectory) {
      throw new Error('Missing target source directory');
    }
    const [declaredDirectory, ...declarations] = (await readFile(await regularFile({ path: join(buildRoot, 'target-sources', `${name}.declared`) }), 'utf8')).trimEnd().split('\n');
    if (!declaredDirectory) {
      throw new Error('Missing declared target source directory');
    }
    const declaredPaths = new Set(declarations.filter(Boolean).map((source) => resolve(declaredDirectory, source)));
    const target = new Map<string, FileIdentity>();
    const actualPaths = files.filter(Boolean).map((source) => resolve(sourceDirectory, source));
    for (const path of new Set([...actualPaths, ...declaredPaths])) {
      // CMake appends virtual custom-command .rule sources while generating.
      // A declared file, real compiler input, or existing generated file is
      // always observed, including a legitimate file named *.rule.
      if (isContained({ root: buildRoot, path }) && path.endsWith('.rule') && !declaredPaths.has(path) && !compiledPaths.has(path) && !await optionalStat({ path })) {
        continue;
      }
      target.set(identify({ path }), await observe({ path }));
    }
    targetSources.set(name.slice(0, -4), Object.fromEntries(target));
  }
  return { translationUnits: Object.fromEntries(translationUnits), targetSources: Object.fromEntries(targetSources) };
}
