import schema from '../../api/schema.mjs';
import { attachCore } from './bindings.mjs';
export { attachCore } from './bindings.mjs';
export { mountReadOnlyFile } from './read-only-file.mjs';
export { schema };

/** No automatic download, model selection, fallback, or Worker creation. */
export async function createCore({ profile, variant = 'browser', baseURL = new URL('../../profiles/', import.meta.url), moduleOptions = {} } = {}) {
  if (!['cpu-wasm32', 'cpu-wasm64', 'webgpu-wasm32-asyncify', 'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi'].includes(profile)) {
    throw new TypeError('Select an explicit supported build profile');
  }
  if (!['browser', 'test'].includes(variant)) {
    throw new TypeError('Select browser or test as the runtime variant');
  }
  const root = new URL(`${profile}/${variant}/`, baseURL);
  const { default: factory } = await import(new URL('core.mjs', root).href);
  const module = await factory({
    locateFile: path => new URL(path, root).href,
    ...moduleOptions,
  });
  return attachCore(module, schema, {
    suspension: profile === 'webgpu-wasm32-asyncify' ? 'asyncify' : 'direct',
  });
}
