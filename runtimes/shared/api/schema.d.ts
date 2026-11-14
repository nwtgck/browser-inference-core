import type { AbiSchema } from '../host/abi.js';

/** Select the actual native target by its compiled fingerprint before use. */
declare const schema: AbiSchema & {
  targets: Record<string, AbiSchema>;
};
export default schema;
