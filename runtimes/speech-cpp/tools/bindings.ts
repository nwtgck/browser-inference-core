import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { runBindingsCli } from '#tools/abi/cli';
import { generateBindings, type BindingPolicy } from '#tools/abi/generate';

export const bindingPolicy: BindingPolicy = {
  runtime: 'speech-cpp',
  astScope: undefined,
  publicFunction: ({ entry: { node, namespaces } }) => !namespaces.length && node.kind === 'FunctionDecl' && node.name?.startsWith('speech_') === true && (!node.mangledName || node.mangledName === node.name),
  exclusions: {},
  acceptedDeprecated: [],
  // Cancellation must remain callable while a JSPI computation is suspended.
  synchronous: ['speech_request_cancel', 'speech_last_error', 'speech_last_error_option'],
};

export function generateRuntimeBindings({ request }: { request: Omit<Parameters<typeof generateBindings>[0], 'policy'> }) {
  return generateBindings({ ...request, policy: bindingPolicy });
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await runBindingsCli({ policy: bindingPolicy });
}
