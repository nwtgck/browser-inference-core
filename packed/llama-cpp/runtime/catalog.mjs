/** Artifact-owned selection. No Node, network, filesystem, or codec dependency. */
export const FORMAT_VERSION = 1;
export const MAX_BYTES = 64 * 1024 * 1024;
const MAX_TARGETS = 128;
const codecs = new Set(['gzip', 'brotli', 'zstd']);
const idPattern = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$/;
const own = (o, k) => Object.hasOwn(o, k);
function record(o, name) {
    if (!o || typeof o !== 'object' || Array.isArray(o))
        throw Error(`Invalid ${name}`);
    return o;
}
function id(s) {
    if (typeof s !== 'string' || !idPattern.test(s))
        throw Error('Invalid identifier');
    return s;
}
function size(n, max = MAX_BYTES) {
    if (!Number.isSafeInteger(n) || n < 0 || n > max)
        throw Error('Invalid size');
    return n;
}
function digest(s) {
    if (typeof s !== 'string' || !/^[a-f0-9]{64}$/.test(s))
        throw Error('Invalid digest');
}
function path(s) {
    if (typeof s !== 'string' || s.length > 512 || s.startsWith('/') ||
        !/^[a-zA-Z0-9_./-]+$/.test(s) || s.split('/').some(v => !v || v === '.' || v === '..')) {
        throw Error('Unsafe asset path');
    }
}
function identity(o) {
    record(o, 'identity');
    size(o.bytes);
    digest(o.sha256);
}
function immutable(o) {
    if (o && typeof o === 'object') {
        for (const v of Object.values(o))
            immutable(v);
        Object.freeze(o);
    }
    return o;
}
export function validateCatalog(catalog) {
    const c = record(catalog, 'catalog');
    if (c.formatVersion !== FORMAT_VERSION || c.decoderApiVersion !== 1)
        throw Error('Unsupported catalog');
    const assets = record(c.assets, 'assets');
    if (Object.keys(assets).length > 10000)
        throw Error('Too many assets');
    const paths = new Set();
    for (const [key, a] of Object.entries(assets)) {
        id(key);
        record(a, 'asset');
        identity(a);
        path(a.path);
        if (paths.has(a.path.toLowerCase()))
            throw Error('Duplicate asset path');
        paths.add(a.path.toLowerCase());
        if (!codecs.has(a.codec))
            throw Error('Unsupported codec');
        identity(a.decoded);
    }
    const offeredCodecs = new Set();
    const targets = record(c.targets, 'targets');
    const names = Object.keys(targets);
    if (!names.length || names.length > MAX_TARGETS)
        throw Error('Invalid target count');
    for (const [key, t] of Object.entries(targets)) {
        id(key);
        record(t, 'target');
        record(t.identity, 'target identity');
        for (const field of ['runtime', 'source', 'profile', 'variant'])
            id(t.identity[field]);
        identity(t.raw);
        path(t.raw.path);
        if (!Array.isArray(t.representations) || !t.representations.length || t.representations.length > 1024)
            throw Error('Invalid representations');
        const seen = new Set();
        for (const rep of t.representations) {
            record(rep, 'representation');
            id(rep.id);
            if (seen.has(rep.id))
                throw Error('Duplicate representation');
            seen.add(rep.id);
            if (!own(assets, rep.payload))
                throw Error('Unknown payload');
            const payload = assets[rep.payload];
            if (rep.kind === 'full') {
                if (rep.base !== undefined || rep.prediction !== undefined || rep.predicted !== undefined)
                    throw Error('Full depends on a base');
                if (payload.decoded.bytes !== t.raw.bytes || payload.decoded.sha256 !== t.raw.sha256)
                    throw Error('Full identity mismatch');
            }
            else if (rep.kind === 'delta') {
                if (!own(targets, rep.base) || rep.base === key)
                    throw Error('Unknown or self base');
                if (targets[rep.base].identity?.runtime !== t.identity.runtime)
                    throw Error('Cross-runtime base');
                if (rep.prediction !== undefined) {
                    if (!own(assets, rep.prediction))
                        throw Error('Unknown prediction');
                    if (assets[rep.prediction].codec !== payload.codec)
                        throw Error('Mixed codec representation');
                    identity(rep.predicted);
                    if (assets[rep.prediction].decoded.bytes > 4 * 1024 * 1024)
                        throw Error('Large prediction');
                }
                else if (rep.predicted !== undefined)
                    throw Error('Missing prediction');
            }
            else
                throw Error('Unknown representation');
        }
        // Every offered codec must have an independent alternative. Subset selection
        // must never force an unselected target to become a hidden dictionary.
        const offered = new Set(t.representations.map(r => assets[r.payload].codec));
        for (const codec of offered) {
            offeredCodecs.add(codec);
            if (!t.representations.some(r => r.kind === 'full' && assets[r.payload].codec === codec))
                throw Error('Missing independent full');
        }
    }
    record(c.runtime, 'runtime');
    if (!Array.isArray(c.runtime.files) || !c.runtime.files.length)
        throw Error('Missing runtime files');
    const runtimePaths = new Set();
    for (const f of c.runtime.files) {
        identity(f);
        path(f.path);
        if (runtimePaths.has(f.path.toLowerCase()) || paths.has(f.path.toLowerCase()))
            throw Error('Runtime path collision');
        runtimePaths.add(f.path.toLowerCase());
        if (!Array.isArray(f.codecs) || !f.codecs.length ||
            new Set(f.codecs).size !== f.codecs.length || f.codecs.some(v => !codecs.has(v)))
            throw Error('Invalid runtime codecs');
        if (f.role === 'zstd')
            identity(f.decoded);
        if (!['always', 'delta', 'prediction', 'zstd'].includes(f.role))
            throw Error('Invalid runtime role');
    }
    if (typeof c.runtime.entry !== 'string' || !runtimePaths.has(c.runtime.entry.toLowerCase()))
        throw Error('Missing runtime entry');
    if (c.runtime.entries !== undefined) {
        record(c.runtime.entries, 'codec entries');
        for (const [codec, entry] of Object.entries(c.runtime.entries)) {
            if (!codecs.has(codec))
                throw Error('Invalid entry codec');
            path(entry);
            if (!c.runtime.files.some(f => f.path === entry && f.codecs.includes(codec))) {
                throw Error('Missing codec entry');
            }
        }
    }
    // Selection must never return an entry omitted by codec/role pruning. Exact
    // spelling matters on case-sensitive filesystems and hosted URLs.
    path(c.runtime.entry);
    if (!c.runtime.files.some(f => f.path === c.runtime.entry))
        throw Error('Missing exact runtime entry');
    for (const codec of offeredCodecs) {
        const entry = c.runtime.entries?.[codec] ?? c.runtime.entry;
        if (!c.runtime.files.some(f => f.path === entry && f.role === 'always' && f.codecs.includes(codec)))
            throw Error(`Missing unconditional ${codec} runtime entry`);
    }
    if (offeredCodecs.has('zstd')) {
        const decoders = c.runtime.files.filter(f => f.role === 'zstd');
        if (decoders.length !== 1 || !decoders[0].codecs.includes('zstd') ||
            decoders[0].decoded.bytes < 8 || decoders[0].decoded.bytes > 4 * 1024 * 1024)
            throw Error('Missing or ambiguous Zstandard decoder');
    }
    return c;
}
function resources(c, chosen, codec) {
    const ids = new Set();
    let delta = false, prediction = false;
    for (const rep of Object.values(chosen)) {
        ids.add(rep.payload);
        if (rep.kind === 'delta')
            delta = true;
        if (rep.prediction !== undefined) {
            prediction = true;
            ids.add(rep.prediction);
        }
    }
    const runtime = c.runtime.files.filter(f => f.codecs.includes(codec) &&
        (f.role === 'always' || f.role === 'delta' && delta || f.role === 'prediction' && prediction || f.role === 'zstd' && codec === 'zstd'));
    const files = [...ids].sort().map(k => c.assets[k]);
    const all = [...files, ...runtime].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
    return { ids, runtime, files: all, bytes: all.reduce((n, a) => n + a.bytes, 0) };
}
/** Deterministic bounded search; up to 12 targets is exhaustive over full roots.
 * Larger sets use safe all-full plus single-root candidates, not an optimality claim.
 * The consumer selects exact IDs and a codec: absence is always an error.
 */
export function selectWasmAssets(catalog, { targets: requested, codec, fullTargets = [] }) {
    const c = validateCatalog(catalog);
    if (!codecs.has(codec))
        throw Error('Unsupported requested codec');
    if (!Array.isArray(requested) || requested.length > MAX_TARGETS || new Set(requested).size !== requested.length)
        throw Error('Invalid requested targets');
    if (requested.some(k => typeof k !== 'string' || !idPattern.test(k)))
        throw Error('Invalid requested target identifier');
    const names = [...requested].sort();
    if (names.some(k => !own(c.targets, k)))
        throw Error('Unavailable target');
    if (!Array.isArray(fullTargets) || new Set(fullTargets).size !== fullTargets.length || fullTargets.some(k => !names.includes(k)))
        throw Error('Invalid full targets');
    if (!names.length)
        return immutable({ formatVersion: 1, decoderApiVersion: 1, codec, targets: {}, assets: {}, runtime: { entry: null, files: [] }, files: [], bytes: 0, rawBytes: 0 });
    const full = {}, alternatives = {};
    for (const name of names) {
        const reps = c.targets[name].representations.filter(r => c.assets[r.payload].codec === codec);
        const cost = r => c.assets[r.payload].bytes + (r.prediction === undefined ? 0 : c.assets[r.prediction].bytes);
        reps.sort((a, b) => cost(a) - cost(b) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
        full[name] = reps.find(r => r.kind === 'full');
        if (!full[name])
            throw Error(`No ${codec} full for ${name}`);
        alternatives[name] = reps;
    }
    let best;
    function consider(roots) {
        if (fullTargets.some(k => !roots.has(k)))
            return;
        const chosen = {};
        for (const name of names) {
            chosen[name] = roots.has(name) ? full[name] : alternatives[name].find(r => r.kind === 'delta' && roots.has(r.base));
            if (!chosen[name])
                return;
        }
        const r = resources(c, chosen, codec);
        // Assets used by representations and decoder files are counted once. For a
        // fixed root set the per-target choice is greedy; no global optimality claim.
        const signature = names.map(k => chosen[k].id).join('\n');
        if (!best || r.bytes < best.bytes || r.bytes === best.bytes && signature < best.signature)
            best = { ...r, chosen, signature };
    }
    if (names.length <= 12) {
        for (let mask = 1; mask < 2 ** names.length; mask++)
            consider(new Set(names.filter((_, i) => mask & 2 ** i)));
    }
    else {
        consider(new Set(names));
        for (const root of names)
            consider(new Set([root, ...fullTargets]));
    }
    if (!best)
        throw Error('No valid plan');
    const plan = {
        formatVersion: 1, decoderApiVersion: 1, codec,
        targets: Object.fromEntries(names.map(name => [name, { identity: c.targets[name].identity, raw: c.targets[name].raw, representation: best.chosen[name] }])),
        assets: Object.fromEntries([...best.ids].sort().map(k => [k, c.assets[k]])),
        runtime: { entry: c.runtime.entries?.[codec] ?? c.runtime.entry, files: best.runtime }, files: best.files,
        bytes: best.bytes, rawBytes: names.reduce((n, k) => n + c.targets[k].raw.bytes, 0),
    };
    return validatePlan(plan);
}
/** Validate a pruned, executable plan without reintroducing the catalog's other targets. */
export function validatePlan(input) {
    const plan = record(input, 'plan');
    if (!codecs.has(plan.codec))
        throw Error('Invalid plan codec');
    const targets = record(plan.targets, 'targets');
    if (Object.keys(targets).length === 0) {
        if (plan.formatVersion !== 1 || plan.decoderApiVersion !== 1 || Object.keys(record(plan.assets, 'assets')).length || plan.runtime?.entry !== null || plan.runtime?.files?.length !== 0 || plan.files?.length !== 0 || plan.bytes !== 0 || plan.rawBytes !== 0)
            throw Error('Nonempty disabled runtime');
        return immutable(JSON.parse(JSON.stringify(plan)));
    }
    // Reuse catalog validation with synthetic independent-full metadata only for
    // delta targets. These aliases are validation-local, never output or loaded.
    const shadow = { formatVersion: plan.formatVersion, decoderApiVersion: plan.decoderApiVersion,
        assets: { ...plan.assets }, runtime: plan.runtime, targets: {} };
    for (const [k, t] of Object.entries(targets)) {
        record(t, 'planned target');
        const rep = record(t.representation, 'planned representation');
        if (!own(plan.assets, rep.payload) || plan.assets[rep.payload].codec !== plan.codec)
            throw Error('Plan codec mismatch');
        shadow.targets[k] = { identity: t.identity, raw: t.raw, representations: [rep] };
        if (rep.kind === 'delta') {
            if (!own(targets, rep.base) || targets[rep.base].representation.kind !== 'full')
                throw Error('Unselected base or delta chain');
            const sid = `synthetic-${k}`;
            if (own(shadow.assets, sid))
                throw Error('Reserved asset identifier');
            shadow.assets[sid] = { path: `validation-only/${sid}`, codec: plan.codec, bytes: 0, sha256: '0'.repeat(64), decoded: { bytes: t.raw.bytes, sha256: t.raw.sha256 } };
            shadow.targets[k].representations.push({ id: sid, kind: 'full', payload: sid });
        }
    }
    validateCatalog(shadow);
    const refs = resources({ assets: plan.assets, runtime: plan.runtime }, Object.fromEntries(Object.entries(targets).map(([k, t]) => [k, t.representation])), plan.codec);
    if (Object.keys(plan.assets).some(k => !refs.ids.has(k)) || refs.ids.size !== Object.keys(plan.assets).length)
        throw Error('Unused or missing asset in plan');
    if (!Array.isArray(plan.files) || JSON.stringify(refs.files) !== JSON.stringify(plan.files) || refs.bytes !== plan.bytes ||
        refs.runtime.length !== plan.runtime.files.length)
        throw Error('Plan file closure mismatch');
    if (Object.values(targets).reduce((n, t) => n + t.raw.bytes, 0) !== plan.rawBytes)
        throw Error('Raw total mismatch');
    return immutable(JSON.parse(JSON.stringify(plan)));
}
