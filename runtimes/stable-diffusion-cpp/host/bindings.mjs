import { attachAbiCore } from './abi.mjs';

/** The host owns one suspension gate for all calls into this module instance. */
export function attachCore(module, schema, { suspension = 'direct', createApi } = {}) {
  return attachAbiCore({ module, schema, suspension, createApi });
}
