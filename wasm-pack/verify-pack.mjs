/** Build-time independent verification of EVERY alternative, not only the winner. */
import { readFile, lstat } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { gunzipSync, brotliDecompressSync } from 'node:zlib';
const [directory, inputRoot] = process.argv.slice(2);
if (!directory || !inputRoot)
    throw Error('Usage: verify-pack.mjs PACK RAW_ROOT');
const root = path.resolve(directory);
const { validateCatalog, selectWasmAssets } = await import(pathToFileURL(path.join(root, 'runtime/catalog.mjs')));
const { verify } = await import(pathToFileURL(path.join(root, 'runtime/loader-core.mjs')));
const c = validateCatalog(JSON.parse(await readFile(path.join(root, 'catalog.json'), 'utf8')));
async function readSafe(relative) {
    let p = root;
    for (const part of relative.split('/')) {
        p = path.join(p, part);
        if ((await lstat(p)).isSymbolicLink())
            throw Error('Linked pack');
    }
    return new Uint8Array(await readFile(p));
}
for (const f of [...Object.values(c.assets), ...c.runtime.files])
    await verify(await readSafe(f.path), f);
const decompress = (b, codec, n) => {
    const opts = { maxOutputLength: Math.max(1, n) };
    if (codec === 'gzip')
        return new Uint8Array(gunzipSync(b, opts));
    if (codec === 'brotli')
        return new Uint8Array(brotliDecompressSync(b, opts));
    throw Error('Use artifact-owned Zstandard decoder');
};
let checked = 0;
for (const [id, t] of Object.entries(c.targets)) {
    const original = new Uint8Array(await readFile(path.resolve(inputRoot, t.raw.path)));
    await verify(original, t.raw);
    for (const rep of t.representations) {
        const codec = c.assets[rep.payload].codec;
        // A temporary catalog with this candidate forced; all full choices retained
        // only to satisfy validation. Setting full cost high does not change bytes.
        const copy = structuredClone(c);
        const ids = rep.kind === 'full' ? [id] : [id, rep.base];
        copy.targets[id].representations = [...t.representations.filter(r => r.kind === 'full'), rep].filter((v, i, a) => a.findIndex(x => x.id === v.id) === i);
        // Construct the exact plan through the selector with a candidate-only target
        // then override its representation and recompute closure via a helper.
        const fullPlan = selectWasmAssets(copy, { targets: ids, codec, fullTargets: ids });
        const plan = JSON.parse(JSON.stringify(fullPlan));
        plan.targets[id].representation = rep;
        const used = new Set(Object.values(plan.targets).flatMap(x => [x.representation.payload, ...(x.representation.prediction ? [x.representation.prediction] : [])]));
        plan.assets = Object.fromEntries([...used].sort().map(k => [k, c.assets[k]]));
        plan.runtime.files = c.runtime.files.filter(f => f.codecs.includes(codec));
        plan.files = [...Object.values(plan.assets), ...plan.runtime.files].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
        plan.bytes = plan.files.reduce((n, f) => n + f.bytes, 0);
        const { createWasmLoader } = await import(pathToFileURL(path.join(root, plan.runtime.entry)));
        const loader = createWasmLoader({ plan, readAsset: a => readSafe(a.path), decompress });
        const result = await loader.load(id);
        if (!Buffer.from(result).equals(Buffer.from(original)))
            throw Error(`Roundtrip differs: ${id}/${rep.id}`);
        checked++;
    }
}
console.log(JSON.stringify({ verifiedRepresentations: checked, targets: Object.keys(c.targets).length }));
