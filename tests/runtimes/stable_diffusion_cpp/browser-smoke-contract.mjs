// Shared by the actual browser producer and its host-only contract tests.
// This describes evidence; it never fabricates a successful probe result.
import { readFileSync } from 'node:fs';

export const imageSmokeContract = JSON.parse(readFileSync(
  new URL('./browser-smoke-contract.json', import.meta.url), 'utf8'));
if (imageSmokeContract.schemaVersion !== 1) throw Error('Unknown image smoke contract');

export function imageSmokeScope(testWebGpu) {
  if (typeof testWebGpu !== 'boolean') throw TypeError('testWebGpu must be boolean');
  return imageSmokeContract.scopePrefix +
    (testWebGpu ? 'CPU and WebGPU' : 'CPU (no GPU inference)') + imageSmokeContract.scopeSuffix;
}
