import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import type { PreparedSources, Upstreams } from '#dto/speech-cpp';
import { directory, fileIdentities, isContained, relativePath, treeIdentity, type SourceIdentity } from '#tools/shared/files';
import { withOutputDirectories } from '#tools/shared/output';
import { applyRecipe, type Patch } from '#tools/source/patches';
import { exportCommit, selectedGitlinks, type SourceSnapshot, type SourceState } from '#tools/source/capture';
import { runtimeDirectory, type UpstreamDeclarations } from './config.js';
import { patches, sourceSelections, type SourceSelection } from './recipe.js';

export type SourceRole = keyof UpstreamDeclarations;
export type CompileInputs = { compileRoot: string; upstreamRoots: Readonly<Record<SourceRole, string>> };
export type SourceFacts = {
  receipt: PreparedSources;
  files: Record<string, SourceIdentity>;
  identity: ReturnType<typeof treeIdentity>;
};
export type PreparedRuntime = { inputs: CompileInputs; root: string; facts: SourceFacts };

export async function resolveSources({ declarations: upstreams, source }: { declarations: UpstreamDeclarations; source: SourceState }): Promise<{ sources: Record<SourceRole, string>; upstreams: Upstreams }> {
  const { speech, ggml, ...unhandledSources } = upstreams;
  unhandledSources satisfies Record<PropertyKey, never>;
  relativePath({ value: speech.path });
  relativePath({ value: ggml.path });
  let revision: string | undefined;
  switch (source.mode) {
  case 'release': revision = source.head;
    break;
  case 'local': revision = undefined;
    break;
  default: {
    const exhaustive: never = source;
    throw new Error(String(exhaustive));
  }
  }
  const commits = await selectedGitlinks({
    sourceRoot: source.checkoutRoot,
    paths: {
      'speech': `${runtimeDirectory}/${speech.path}`,
      ggml: `${runtimeDirectory}/${ggml.path}`,
    },
    revision,
  });
  const speechCommit = commits['speech'];
  const ggmlCommit = commits.ggml;
  if (speechCommit === undefined || ggmlCommit === undefined) {
    throw new Error('Incomplete speech-runtime gitlink selection');
  }
  return {
    sources: {
      'speech': path.join(source.checkoutRoot, runtimeDirectory, speech.path),
      ggml: path.join(source.checkoutRoot, runtimeDirectory, ggml.path),
    },
    upstreams: {
      'speech': { ...speech, commit: speechCommit },
      ggml: { ...ggml, commit: ggmlCommit },
    },
  };
}

/** Export each selected upstream once and publish only the complete patched pair. */
export async function prepareSources({ sources, upstreams, output, patchRoot, recipe, replace, selection }: { sources: Record<SourceRole, string>; upstreams: Upstreams; output: string; patchRoot: string; recipe: readonly Patch[]; replace: boolean; selection: Readonly<Record<SourceRole, SourceSelection>> | undefined }): Promise<SourceFacts> {
  if (Object.keys(sources).sort().join(',') !== 'ggml,speech') {
    throw new Error('Unknown speech-runtime source set');
  }
  if (path.resolve(sources.ggml) === path.resolve(sources['speech'])) {
    throw new Error('Independent upstreams must not alias');
  }
  for (const source of Object.values(sources)) {
    await directory({ path: source });
    if (isContained({ root: source, path: output }) || isContained({ root: output, path: source })) {
      throw new Error('Prepared output must be outside and not contain an upstream checkout');
    }
  }
  return await withOutputDirectories({
    destinations: [output],
    replace,
    preserveUnchanged: true,
    prepare: async ({ paths }) => {
      const work = paths[0];
      if (work === undefined) {
        throw new Error('Missing prepared output directory');
      }
      const prepared = { 'speech': path.join(work, 'speech'), ggml: path.join(work, 'ggml') };
      for (const role of ['speech', 'ggml'] as const) {
        await exportCommit({ sourceRoot: sources[role], revision: upstreams[role].commit, destination: prepared[role], ...selection?.[role] });
      }
      await directory({ path: prepared.ggml });
      const records = await applyRecipe({ sources: prepared, patchRoot, patches: recipe });
      const receipt: PreparedSources = { runtime: 'speech-cpp', upstreams, patches: records, vendorCheckoutModified: false };
      // CMake consumes these upstream identities to compile version metadata.
      await writeFile(path.join(work, 'patch-provenance.json'), `${JSON.stringify(receipt, undefined, 2)}\n`);
      const files = await fileIdentities({ root: work });
      return { receipt, files, identity: treeIdentity({ files }) };
    },
  });
}

/** Configuration/recipe stay typed checkout values; patch bytes come from the snapshot. */
export async function prepareRuntimeSources({ snapshot, declarations, output, replace }: { snapshot: SourceSnapshot; declarations: UpstreamDeclarations; output: string; replace: boolean }): Promise<PreparedRuntime> {
  const selected = await resolveSources({ declarations, source: snapshot.source });
  const facts = await prepareSources({ ...selected, output, patchRoot: path.join(snapshot.compileRoot, runtimeDirectory, 'upstream-patches'), recipe: patches, replace, selection: sourceSelections });
  return {
    inputs: {
      compileRoot: snapshot.compileRoot,
      upstreamRoots: { 'speech': path.join(output, 'speech'), ggml: path.join(output, 'ggml') },
    },
    root: output,
    facts,
  };
}

export async function verifyPreparedSources({ preparedRoot, facts }: { preparedRoot: string; facts: SourceFacts }): Promise<void> {
  if (!isDeepStrictEqual(await fileIdentities({ root: preparedRoot }), facts.files)) {
    throw new Error('Prepared speech sources changed during build');
  }
}
