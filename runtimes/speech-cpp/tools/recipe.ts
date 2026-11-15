import type { Patch } from '#tools/source/patches';

export type SourceSelection = { include?: readonly string[]; exclude?: readonly string[] };

/** Keep the runtime and its independent ggml checkout separate. */
export const sourceSelections = {
  speech: { exclude: ['ggml', '.github', 'docs', 'models'] },
  ggml: { exclude: ['.github', 'docs', 'examples', 'tests'] },
} as const satisfies Record<'speech' | 'ggml', SourceSelection>;

export const patches: readonly Patch[] = [{
  file: '0001-external-ggml-source.patch',
  target: 'speech',
  reason: 'Compile the independently pinned ggml source without modifying either vendor checkout.',
  files: ['CMakeLists.txt'],
}];
