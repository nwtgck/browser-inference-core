import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { withOutputDirectories } from '#tools/shared/output';
import { copyRegular } from './files.js';
import { collectNotices } from './licenses.js';
import { apiFiles } from './runtime.js';
import type { LlamaProvenance } from '#dto/llama-cpp';
import type { StableDiffusionProvenance } from '#dto/stable-diffusion-cpp';

/** Stage the compiled profile and its provenance together for CI transfer. */
export async function stageProfile({ buildDirectory, generatedDirectory, runtimeFiles, provenance, noticeRoots, noticeDestination }: { buildDirectory: string; generatedDirectory: string; runtimeFiles: readonly { source: string; name: string }[]; provenance: LlamaProvenance | StableDiffusionProvenance; noticeRoots: readonly string[]; noticeDestination: string | undefined }): Promise<void> {
  const names = runtimeFiles.map(file => file.name);
  if (new Set(names).size !== names.length || ['core.mjs', 'core.wasm', 'core.d.ts'].some(name => !names.includes(name))) {
    throw new Error('Staging requires each compiled runtime member exactly once');
  }
  if (noticeRoots.length && !noticeDestination) {
    throw new Error('Toolchain notices require a destination');
  }
  const destinations = [buildDirectory, ...(noticeDestination === undefined ? [] : [noticeDestination])];
  await withOutputDirectories({
    destinations,
    replace: true,
    prepare: async ({ paths }) => {
      const output = paths[0];
      if (!output) {
        throw new Error('Missing profile staging output');
      }
      await mkdir(join(output, 'runtime'));
      await mkdir(join(output, 'generated'));
      for (const file of runtimeFiles) {
        if (!/^[a-zA-Z0-9_.-]+$/u.test(file.name)) {
          throw new Error('Invalid staged runtime filename');
        }
        await copyRegular({ source: file.source, destination: join(output, 'runtime', file.name) });
      }
      if (!runtimeFiles.some(file => file.name === 'abi.json')) {
        await copyRegular({ source: join(generatedDirectory, 'schema.json'), destination: join(output, 'runtime', 'abi.json') });
      }
      for (const name of apiFiles) {
        await copyRegular({ source: join(generatedDirectory, name), destination: join(output, 'generated', name) });
      }
      await writeFile(join(output, 'provenance.json'), JSON.stringify(provenance, undefined, 2) + '\n');
      const notices = paths[1];
      if (notices) {
        const names = noticeRoots.map(root => root.split('/').at(-1));
        if (new Set(names).size !== names.length) {
          throw new Error('Toolchain notice roots must have distinct names');
        }
        for (const root of noticeRoots) {
          await collectNotices({ source: root, destination: join(notices, root.split('/').at(-1) ?? ''), required: true });
        }
      }
    },
  });
}
