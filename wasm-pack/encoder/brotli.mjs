// Build-time adapter uses Node's built-in codec, not an install-time download.
import { brotliCompressSync, brotliDecompressSync, constants } from 'node:zlib';
const limit = 64 * 1024 * 1024, chunks = [];
let length = 0;
for await (const chunk of process.stdin) {
    length += chunk.length;
    if (length > limit)
        throw Error('Input limit');
    chunks.push(chunk);
}
const raw = Buffer.concat(chunks), packed = brotliCompressSync(raw, { params: { [constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_SIZE_HINT]: raw.length } });
if (!brotliDecompressSync(packed, { maxOutputLength: Math.max(1, raw.length) }).equals(raw))
    throw Error('Brotli roundtrip');
process.stdout.write(packed);
