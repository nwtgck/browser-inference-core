/** Artifact-owned decoder API. Consumers provide bytes, never interpret the recipe. */
import { validatePlan, MAX_BYTES } from './catalog.mjs';
export async function sha256(bytes) {
    const hash = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
    return Array.from(hash, v => v.toString(16).padStart(2, '0')).join('');
}
export async function verify(bytes, expected) {
    if (!(bytes instanceof Uint8Array) || bytes.length !== expected.bytes || await sha256(bytes) !== expected.sha256)
        throw Error('Asset length/hash mismatch');
    return bytes;
}
/** Consume the entire frame, bound allocation, and reject overflow before returning. */
export async function decompressNative(bytes, codec, expectedBytes, signal) {
    if (!['gzip', 'brotli'].includes(codec))
        throw Error('Unsupported native codec');
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 0 || expectedBytes > MAX_BYTES)
        throw Error('Invalid output bound');
    signal?.throwIfAborted();
    const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(codec)).getReader();
    const out = new Uint8Array(expectedBytes);
    let at = 0;
    const cancel = () => {
        void reader.cancel(signal.reason).catch(() => {
        });
    };
    signal?.addEventListener('abort', cancel, { once: true });
    try {
        for (;;) {
            signal?.throwIfAborted();
            const { value, done } = await reader.read();
            signal?.throwIfAborted();
            if (done)
                break;
            if (!(value instanceof Uint8Array) || value.length > expectedBytes - at)
                throw Error('Decompression overflow');
            out.set(value, at);
            at += value.length;
        }
        if (at !== expectedBytes)
            throw Error('Decompression length mismatch');
        return out;
    }
    catch (error) {
        await reader.cancel(error).catch(() => {
        });
        throw error;
    }
    finally {
        signal?.removeEventListener('abort', cancel);
        reader.releaseLock();
    }
}
/** No unbounded successful-output cache, network fallback, or shared caller cancellation.
 * Requests are serialized to bound decoder work; a failed request cannot poison the next.
 * `decompress` is an optional platform adapter for tests/embedded runtimes. All results
 * are still length/hash checked here. No raw-Wasm fallback is implemented.
 */
export function createCoreLoader({ plan: input, readAsset, decompress = decompressNative, createZstdDecoder }) {
    const plan = validatePlan(input);
    if (typeof readAsset !== 'function' || typeof decompress !== 'function')
        throw TypeError('Reader and decompressor required');
    let tail = Promise.resolve();
    async function execute(id, signal) {
        signal?.throwIfAborted();
        if (!Object.hasOwn(plan.targets, id))
            throw Error('Unavailable target');
        let zstd;
        async function decompressPayload(bytes, codec, n) {
            if (codec !== 'zstd')
                return decompress(bytes, codec, n, signal);
            if (!zstd) {
                const f = plan.runtime.files.find(f => f.role === 'zstd');
                if (!f || !f.decoded)
                    throw Error('Missing Zstandard decoder');
                const read = await readAsset({ ...f }, signal);
                if (!(read instanceof Uint8Array) || read.length !== f.bytes)
                    throw Error('Decoder length mismatch');
                const packed = read.slice();
                await verify(packed, f);
                signal?.throwIfAborted();
                const binary = await decompress(packed, 'gzip', f.decoded.bytes, signal);
                await verify(binary, f.decoded);
                signal?.throwIfAborted();
                if (typeof createZstdDecoder !== 'function')
                    throw Error('Use the selected codec entry point');
                zstd = await createZstdDecoder(binary);
            }
            signal?.throwIfAborted();
            return zstd.decode(bytes, new Uint8Array(), n, { maxWindow: 1048576 });
        }
        async function payload(key) {
            const a = plan.assets[key];
            signal?.throwIfAborted();
            // Snapshot caller-owned buffers before hashing and asynchronous use.
            const read = await readAsset({ ...a }, signal);
            if (!(read instanceof Uint8Array) || read.length !== a.bytes)
                throw Error('Asset length mismatch');
            const bytes = read.slice();
            signal?.throwIfAborted();
            await verify(bytes, a);
            const result = await decompressPayload(bytes, a.codec, a.decoded.bytes);
            signal?.throwIfAborted();
            return verify(result, a.decoded);
        }
        const target = plan.targets[id], rep = target.representation;
        let result;
        if (rep.kind === 'full')
            result = await payload(rep.payload);
        else {
            const base = plan.targets[rep.base];
            let dictionary = await payload(base.representation.payload);
            await verify(dictionary, base.raw);
            signal?.throwIfAborted();
            if (rep.prediction !== undefined) {
                const rules = await payload(rep.prediction);
                const { predict } = await import('./predict.mjs');
                signal?.throwIfAborted();
                dictionary = predict(dictionary, rules, { expectedBytes: rep.predicted.bytes });
                await verify(dictionary, rep.predicted);
                signal?.throwIfAborted();
            }
            const recipe = await payload(rep.payload);
            const { decodeDelta } = await import('./delta.mjs');
            signal?.throwIfAborted();
            result = decodeDelta(dictionary, recipe, { expectedTargetBytes: target.raw.bytes });
        }
        signal?.throwIfAborted();
        await verify(result, target.raw);
        signal?.throwIfAborted();
        return result;
    }
    return Object.freeze({
        plan,
        load(id, { signal } = {}) {
            if (typeof id !== 'string')
                return Promise.reject(TypeError('Target identifier required'));
            const work = tail.then(() => execute(id, signal));
            tail = work.catch(() => {
            });
            return work;
        },
    });
}
