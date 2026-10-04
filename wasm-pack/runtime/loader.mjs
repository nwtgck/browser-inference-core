/** gzip/Brotli entry. This module graph has no Zstandard implementation. */
export { sha256, verify, decompressNative } from './loader-core.mjs';
import { createCoreLoader } from './loader-core.mjs';
export function createWasmLoader(options) {
    if (options.plan?.codec === 'zstd') {
        throw new Error('Use plan.runtime.entry for the selected Zstandard decoder');
    }
    return createCoreLoader(options);
}
