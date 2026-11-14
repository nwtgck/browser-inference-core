import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { runBindingsCli } from '#tools/abi/cli';
import { generateBindings, type BindingPolicy } from '#tools/abi/generate';

export const bindingPolicy: BindingPolicy = {
  runtime: 'stable-diffusion-cpp',
  astScope: undefined,
  publicFunction: ({ entry: { node, namespaces }, translationUnit: unit }) => !namespaces.length && node.kind === 'FunctionDecl' && !node.name?.startsWith('__') && !node.name?.startsWith('sdc_test_') && (!node.mangledName || node.mangledName === node.name)
    && (/^(?:ggml|gguf)_/u.test(node.name ?? '') || node.loc?.includedFrom?.file === unit),
  exclusions: {
    ggml_threadpool_get_n_threads: 'declared by upstream but not implemented in the pinned source set',
    ggml_fopen: 'FILE lifetime is outside the browser core',
    gguf_init_from_file_ptr: 'FILE ownership is outside the browser core',
  },
  acceptedDeprecated: [],
  synchronous: ['sd_set_preview_callback', 'sd_cancel_generation'],
};

export function generateRuntimeBindings({ request }: { request: Omit<Parameters<typeof generateBindings>[0], 'policy'> }) {
  return generateBindings({ ...request, policy: bindingPolicy });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await runBindingsCli({ policy: bindingPolicy });
}
