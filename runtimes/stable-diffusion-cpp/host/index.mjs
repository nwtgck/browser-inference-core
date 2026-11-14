/** Optional, policy-free host helpers. The caller creates the module and owns its lifetime. */
import { createLowLevelFunctions } from '../api/functions.mjs';
import { attachCore as attachBindings } from './bindings.mjs';
export { createAbiCaller, selectAbiSchema, toNativePointer, fromNativePointer, toEmscriptenPointer, fromEmscriptenPointer } from './abi.mjs';
export { mountReadOnlyFile } from './read-only-file.mjs';
export { default as schema } from '../api/schema.mjs';

export function attachCore(module, metadata, options = {}) {
  return attachBindings(module, metadata, { ...options, createApi: createLowLevelFunctions });
}
