import { attachAbiCore } from './abi.mjs';

export function attachCore(module, schema, { suspension, createApi }) {
  return attachAbiCore({ module, schema, suspension, createApi });
}
