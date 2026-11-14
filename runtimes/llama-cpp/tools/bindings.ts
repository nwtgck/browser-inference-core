import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { runBindingsCli } from '#tools/abi/cli';
import { generateBindings, type BindingPolicy } from '#tools/abi/generate';

export const bindingPolicy: BindingPolicy = {
  runtime: 'llama-cpp',
  astScope: {
    filters: ['llama_', 'ggml_', 'gguf_', 'mtmd', 'Bicore', 'lcb_', 'bicore_', 'int', 'size_t', 'ptrdiff_t'],
    prefixes: ['llama_', 'ggml_', 'gguf_', 'mtmd_', 'Bicore', 'lcb_', 'bicore_'],
    aliases: ['int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'intptr_t', 'uintptr_t', 'size_t', 'ssize_t', 'ptrdiff_t', 'off_t'],
  },
  publicFunction: ({ entry: { node, namespaces } }) => !namespaces.length && node.kind === 'FunctionDecl' && (/^(?:llama|ggml|gguf|mtmd)_/u.test(node.name ?? '') || ['bicore_file_backed_rows_version', 'bicore_model_file_backed_shard'].includes(node.name ?? '')) && (!node.mangledName || node.mangledName === node.name),
  exclusions: {
    ggml_threadpool_get_n_threads: 'declared by upstream but not implemented in the pinned source set',
    mtmd_helper_video_init: 'requires subprocess video support, unavailable in the browser source set',
    mtmd_helper_video_init_from_buf: 'requires subprocess video support, unavailable in the browser source set',
    mtmd_helper_video_free: 'requires subprocess video support, unavailable in the browser source set',
    mtmd_helper_video_get_info: 'requires subprocess video support, unavailable in the browser source set',
    mtmd_helper_video_read_next: 'requires subprocess video support, unavailable in the browser source set',
  },
  acceptedDeprecated: ['mtmd_helper_model_can_chat', 'mtmd_helper_support_video'],
  synchronous: ['ggml_op_desc', 'ggml_backend_buffer_name', 'ggml_backend_buffer_is_host', 'ggml_backend_dev_count', 'ggml_backend_dev_get', 'ggml_backend_dev_name', 'ggml_backend_dev_supports_op', 'bicore_file_backed_rows_version', 'bicore_model_file_backed_shard'],
};

export function generateRuntimeBindings({ request }: { request: Omit<Parameters<typeof generateBindings>[0], 'policy'> }) {
  return generateBindings({ ...request, policy: bindingPolicy });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await runBindingsCli({ policy: bindingPolicy });
}
