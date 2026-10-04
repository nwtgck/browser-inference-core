/** Stage-4 bounded streaming wrapper. Caller receives bytes only after validation. */
export async function createStreamingZstdDecoder(binary) {
    const { instance } = await WebAssembly.instantiate(binary, {}), e = instance.exports, CAP = 64 * 1024 * 1024;
    return {
        decode(input, base, size, { maxWindow = CAP, rangeStart = 0, rangeLength = size } = {}) {
            if (!(input instanceof Uint8Array) || !(base instanceof Uint8Array))
                throw TypeError('byte views');
            if (!Number.isSafeInteger(size) || size < 0 || size > CAP || input.length > CAP || base.length > CAP || !Number.isSafeInteger(maxWindow) || maxWindow < 1 || maxWindow > CAP)
                throw Error('limits');
            if (!Number.isSafeInteger(rangeStart) || !Number.isSafeInteger(rangeLength) || rangeStart < 0 || rangeLength < 0 || rangeStart > size || rangeLength > size - rangeStart)
                throw Error('range');
            e.clear();
            const p = e.prepare(0, input.length), b = e.prepare(1, base.length);
            if (!p || !b)
                throw Error('prepare');
            new Uint8Array(e.memory.buffer, p, input.length).set(input);
            new Uint8Array(e.memory.buffer, b, base.length).set(base);
            const status = e.start(size, maxWindow);
            if (status) {
                e.clear();
                throw Error(`zstd start ${status}`);
            }
            const out = new Uint8Array(rangeLength);
            let offset = 0;
            try {
                for (;;) {
                    const n = e.pull();
                    if (n < 0)
                        throw Error(`zstd pull ${n}`);
                    if (n === 0)
                        break;
                    if (n > size - offset)
                        throw Error('output overflow');
                    const lo = Math.max(offset, rangeStart), hi = Math.min(offset + n, rangeStart + rangeLength);
                    if (hi > lo)
                        out.set(new Uint8Array(e.memory.buffer, e.chunk_ptr() + lo - offset, hi - lo), lo - rangeStart);
                    offset += n;
                }
                if (offset !== size)
                    throw Error('output length');
                return out;
            }
            finally {
                e.clear();
            }
        }, memoryBytes: () => e.memory.buffer.byteLength, clear: () => e.clear()
    };
}
