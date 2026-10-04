/** Zstandard entry, selected only when the plan uses that codec. */
export { sha256, verify, decompressNative } from './loader-core.mjs';
import { createCoreLoader } from './loader-core.mjs';
import { createStreamingZstdDecoder } from './zstd-loader.mjs';
export function createWasmLoader(options) {
    return createCoreLoader({ ...options, createZstdDecoder: createStreamingZstdDecoder });
}
