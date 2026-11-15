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
}, {
  file: '0002-webgpu-matmul-vec4-layout.patch',
  target: 'ggml',
  reason: 'Use scalar matrix kernels when binding offsets or active tensor strides cannot address vec4 elements exactly.',
  files: [
    'src/ggml-webgpu/ggml-webgpu.cpp',
    'src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    'src/ggml-webgpu/ggml-webgpu-mul-mat-layout.hpp',
  ],
}, {
  file: '0003-webgpu-buffer-address-range.patch',
  target: 'ggml',
  reason: 'Split weight buffers without size_t overflow and keep GPU buffer offsets within the native pointer address range.',
  files: ['src/ggml-backend.cpp', 'src/ggml-webgpu/ggml-webgpu.cpp'],
}];
