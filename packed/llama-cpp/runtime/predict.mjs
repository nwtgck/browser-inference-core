/** Bounded byte-pattern predictor. Never parse or execute the predicted bytes.
 * Longest matching OLD pattern wins; replacements are NOT recursively rescanned.
 * A separate lossless patch must reconstruct the original, hash-checked target.
 */
export function predict(base, rules, { expectedBytes, maxBytes = 64 * 1024 * 1024 } = {}) {
    if (!(base instanceof Uint8Array) || !(rules instanceof Uint8Array))
        throw new TypeError('byte views required');
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 0 || base.length > 64 * 1024 * 1024 || rules.length > 4 * 1024 * 1024 || !Number.isSafeInteger(expectedBytes) || expectedBytes < 0 || expectedBytes > maxBytes || maxBytes > 64 * 1024 * 1024 || rules.length < 24)
        throw new Error('limits');
    const d = new DataView(rules.buffer, rules.byteOffset, rules.byteLength);
    if (d.getUint32(0, true) !== 0x31445250)
        throw new Error('format');
    const bs = d.getUint32(4, true), ts = d.getUint32(8, true), count = d.getUint32(12, true), oldLen = d.getUint32(16, true), newLen = d.getUint32(20, true);
    if (bs !== base.length || ts !== expectedBytes || count > 65536 || 24 + 2 * count + oldLen + newLen !== rules.length)
        throw new Error('size contract');
    const offsets = new Uint32Array(count), replacements = new Uint32Array(count), byPrefix = new Map(), lead = new Uint8Array(65536);
    let old = 24 + 2 * count, neo = old + oldLen;
    for (let i = 0; i < count; i++) {
        const n = rules[24 + i], m = rules[24 + count + i];
        if (n < 3 || n > 16 || m < 1 || m > 16 || old + n > 24 + 2 * count + oldLen || neo + m > rules.length)
            throw new Error('rule length');
        if (i) {
            const prev = offsets[i - 1], prevN = rules[24 + i - 1];
            if (prevN < n)
                throw new Error('noncanonical pattern order');
            if (prevN === n) {
                let j = 0;
                while (j < n && rules[prev + j] === rules[old + j])
                    j++;
                if (j === n || rules[prev + j] >= rules[old + j])
                    throw new Error('duplicate or unsorted pattern');
            }
        }
        offsets[i] = old;
        replacements[i] = neo;
        const key = rules[old] | rules[old + 1] << 8 | rules[old + 2] << 16;
        let bucket = byPrefix.get(key);
        if (!bucket) {
            bucket = [];
            byPrefix.set(key, bucket);
        }
        if (bucket.length >= 256)
            throw new Error('too many colliding rules');
        // Sorting locally makes the longest-match contract independent of file ordering.
        bucket.push(i);
        lead[rules[old] | rules[old + 1] << 8] = 1;
        old += n;
        neo += m;
    }
    if (old !== 24 + 2 * count + oldLen || neo !== rules.length)
        throw new Error('trailing rule bytes');
    const out = new Uint8Array(ts);
    let at = 0, plain = 0, i = 0, work = 0;
    const budget = 64 * base.length + 65536;
    for (; i + 2 < base.length;) {
        if (!lead[base[i] | base[i + 1] << 8]) {
            i++;
            continue;
        }
        const key = base[i] | base[i + 1] << 8 | base[i + 2] << 16, bucket = byPrefix.get(key);
        let found = -1;
        if (bucket)
            for (const r of bucket) {
                const n = rules[24 + r], p = offsets[r];
                if (i + n > base.length)
                    continue;
                let j = 3;
                for (; j < n; j++) {
                    if (++work > budget)
                        throw new Error('work limit');
                    if (base[i + j] !== rules[p + j])
                        break;
                }
                if (j === n) {
                    found = r;
                    break;
                }
            }
        if (found < 0) {
            i++;
            continue;
        }
        const n = rules[24 + found], m = rules[24 + count + found], gap = i - plain;
        if (gap > ts - at || m > ts - at - gap)
            throw new Error('prediction overflow');
        out.set(base.subarray(plain, i), at);
        at += gap;
        out.set(rules.subarray(replacements[found], replacements[found] + m), at);
        at += m;
        i += n;
        plain = i;
    }
    const remain = base.length - plain;
    if (remain !== ts - at)
        throw new Error('prediction length');
    out.set(base.subarray(plain), at);
    return out;
}
