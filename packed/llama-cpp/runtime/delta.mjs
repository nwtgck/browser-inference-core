/** Bounded byte-copy decoder. No WebAssembly parser or semantic transform.
 * LNP1: existing copy/literal lanes. WXP1: same lanes + sparse byte corrections.
 * Corrections are applied BEFORE a later copy may reference the corrected output.
 */
export function decodeDelta(base, patch, { expectedTargetBytes, maxTargetBytes = 64 * 1024 * 1024 } = {}) {
    if (!(base instanceof Uint8Array) || !(patch instanceof Uint8Array))
        throw TypeError('byte views required');
    const validSize = v => Number.isSafeInteger(v) && v >= 0 && v <= 64 * 1024 * 1024;
    if (!validSize(maxTargetBytes) || !validSize(expectedTargetBytes) || expectedTargetBytes > maxTargetBytes || patch.length > 128 * 1024 * 1024)
        throw Error('invalid limit');
    if (patch.length < 32)
        throw Error('short header');
    const d = new DataView(patch.buffer, patch.byteOffset, patch.byteLength);
    const magic = d.getUint32(0, true), sparse = magic === 0x31505857;
    if (!sparse && magic !== 0x31504e4c)
        throw Error('unknown format');
    const header = sparse ? 44 : 32;
    if (patch.length < header)
        throw Error('short sparse header');
    const mode = d.getUint32(4, true), additive = mode >= 4, sourceMode = mode % 4;
    if (![0, 2].includes(sourceMode) || mode > (sparse ? 6 : 2))
        throw Error('unknown mode');
    const bs = d.getUint32(8, true), ts = d.getUint32(12, true), count = d.getUint32(16, true);
    const l = d.getUint32(20, true), m = d.getUint32(24, true), o = d.getUint32(28, true);
    const controlEnd = header + l + m + o;
    if (bs !== base.length || ts !== expectedTargetBytes || ts > maxTargetBytes || count > ts + 1 || count > Math.min(l, m, o) || controlEnd > patch.length)
        throw Error('size contract');
    const lits = sparse ? d.getUint32(32, true) : patch.length - controlEnd;
    const corrections = sparse ? d.getUint32(36, true) : 0, gapBytes = sparse ? d.getUint32(40, true) : 0;
    const gapStart = controlEnd + lits, gapEnd = gapStart + gapBytes;
    if (lits > ts || corrections > ts || corrections > gapBytes || gapEnd + corrections !== patch.length)
        throw Error('correction bounds');
    const positions = [header, header + l, header + l + m, gapStart], ends = [header + l, header + l + m, controlEnd, gapEnd];
    function read(lane) {
        let value = 0, mul = 1;
        for (let i = 0; i < 5; i++) {
            if (positions[lane] >= ends[lane])
                throw Error('truncated integer');
            const b = patch[positions[lane]++];
            if (i === 4 && b > 15)
                throw Error('integer overflow');
            value += (b & 127) * mul;
            if (!(b & 128)) {
                if (i && b === 0)
                    throw Error('noncanonical integer');
                return value;
            }
            mul *= 128;
        }
        throw Error('integer overflow');
    }
    let changeIndex = 0, nextChange = -1;
    function advanceChange() {
        if (changeIndex === corrections) {
            nextChange = Infinity;
            return;
        }
        const gap = read(3);
        if (!gap)
            throw Error('zero correction gap');
        nextChange += gap;
        if (nextChange >= ts || !patch[gapEnd + changeIndex])
            throw Error('invalid correction');
    }
    advanceChange();
    const out = new Uint8Array(ts);
    let at = 0, lit = controlEnd, previousBase = 0;
    function correct(start, n) {
        if (nextChange < start)
            throw Error('correction in literal');
        while (nextChange < start + n) {
            const v = patch[gapEnd + changeIndex++];
            out[nextChange] = additive ? (out[nextChange] + v) & 255 : out[nextChange] ^ v;
            advanceChange();
        }
    }
    for (let i = 0; i < count; i++) {
        const ll = read(0), ml = read(1), code = read(2);
        if (!ll && !ml || ll > ts - at || ml > ts - at - ll || ll > gapStart - lit)
            throw Error('record bounds');
        if (nextChange < at + ll)
            throw Error('correction in literal');
        out.set(patch.subarray(lit, lit + ll), at);
        lit += ll;
        at += ll;
        if (!ml) {
            if (code !== 0)
                throw Error('unused reference');
            continue;
        }
        let offset;
        if (sourceMode === 0)
            offset = code;
        else if (code % 2 === 1)
            offset = (code - 1) / 2;
        else {
            if (code < 2)
                throw Error('bad base reference');
            const zig = (code - 2) / 2, delta = zig % 2 === 0 ? zig / 2 : -(zig + 1) / 2, index = previousBase + delta;
            if (index < 0 || index >= bs || ml > bs - index)
                throw Error('bad base range');
            previousBase = index + ml;
            offset = bs + at - index;
        }
        if (offset <= 0 || offset > bs + at)
            throw Error('invalid copy');
        let source = at - offset, left = ml;
        if (source < 0) {
            const n = Math.min(left, -source);
            out.set(base.subarray(bs + source, bs + source + n), at);
            correct(at, n);
            source += n;
            at += n;
            left -= n;
        }
        while (left) {
            const n = Math.min(left, at - source);
            if (n <= 0)
                throw Error('forward copy');
            out.set(out.subarray(source, source + n), at);
            correct(at, n);
            source += n;
            at += n;
            left -= n;
        }
    }
    if (at !== ts || lit !== gapStart || changeIndex !== corrections || positions.some((v, i) => v !== ends[i]))
        throw Error('trailing or missing bytes');
    return out;
}
