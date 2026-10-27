// Source-bound real-Dawn T1 alias comparison. Each lane owns an independent union arena.
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
const argv = process.argv.slice(2);
const option = (key, fallback) => {const i = argv.indexOf(key); return i < 0 ? fallback : argv[i + 1];};
const lab = resolve(option('--work', '.'));
const {create, globals} = await import(pathToFileURL(resolve(option('--webgpu-module', ''))).href);
Object.assign(globalThis, globals);
const output = resolve(option('--output', join(lab, 'merged-correctness.json')));
const workgroups = option('--workgroups', '256,512,1024').split(',').map(Number);
const limit = Number(option('--limit', Infinity));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const check = (ok, message) => {if (!ok) throw new Error(message);};
const words = bytes => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const tmp = new ArrayBuffer(4), f = new Float32Array(tmp), u = new Uint32Array(tmp);
const bits = x => {f[0] = x; return u[0];};
const number = x => {u[0] = x; return f[0];};
const hex = x => '0x' + x.toString(16).padStart(8, '0');
const classify = x => (x & 0x7fffffff) > 0x7f800000 ? 'nan' : (x & 0x7fffffff) === 0x7f800000 ? x >>> 31 ? '-infinity' : '+infinity' : (x & 0x7fffffff) === 0 ? x >>> 31 ? '-zero' : '+zero' : (x & 0x7fffffff) < 0x00800000 ? 'subnormal' : 'finite';
const integrated = JSON.parse(readFileSync(join(lab, 'manifest.json')));
const pins = integrated.source;
const original = {generatedShaderSha256: integrated.baselineShaderSha256};
const report = {status: 'started', scope: 'T1 only, actual extracted fused factory + exact production baseline WGSL; independent alias arenas; actual selected backend, no actual graph or model throughput claim.',
    runnerSha256: hash(readFileSync(fileURLToPath(import.meta.url))), pins, factoryShaderSha256: integrated.shaderSha256, adapter: {}, devices: [], cases: [], errors: []};
check(integrated.status === 'prepared', 'Source-bound preparation must finish before GPU');
const backend = option('--backend', undefined);
const owner = create(backend ? ['backend=' + backend] : []); globalThis[Symbol.for('router-tail-merged-lab.owner')] = owner;
const live = new Set(); let device;

function patterns(pattern, used, round) {
    const floats = x => x.map(bits);
    switch (pattern) {
        case 'positive': return undefined;
        case 'sum-tree': return [0x3f14bccd, 0x3ed66b0c, 0x394a5e79, 0x374e75b7, 0x36676f58];
        case 'direct-div': return [0x3f6ed347, bits(Math.fround(number(0x3f7fffff) - number(0x3f6ed347))), 0];
        case 'cancellation': return round ? floats([2 ** 23, -(2 ** 23), 0.25, -0.5, 0.125, 0.0625, -0.03125]) : floats([2 ** 24, 1, -(2 ** 24), -1, 0.0001, -0.00005, 0.125]);
        case 'below-clamp': return floats([2 ** -18, 2 ** -19, 2 ** -20]);
        case 'clamp-threshold': return [round ? 0x387fffff : 0x38800001, 0, 0, 0];
        case 'clamp-range': return floats([0.125, 0.25, 0.5, 0.75, 1]);
        case 'signed-zero': return round ? [0x80000000, 0x80000000, 0, 0x80000000, 0] : [0, 0x80000000, 0, 0x80000000, 0];
        case 'subnormal': return round ? [1, 2, 0x80000001, 0x007fffff, 0x807fffff] : [1, 0x007fffff, 0x00800000, 0x80000001, 0];
        case 'qnan': return [0x7fc12345, bits(0.25), bits(0.5), 0, bits(0.125)];
        case 'snan': return [bits(0.5), 0x7f812345, bits(0.25), 0, bits(0.125)];
        case '+inf': return [0x7f800000, bits(0.25), bits(0.5), 0, bits(0.125)];
        case '-inf': return [0xff800000, bits(0.25), bits(0.5), 0, bits(0.125)];
        case 'mixed-inf': return [0x7f800000, 0xff800000, bits(0.5), 0, bits(0.125)];
        case 'overflow': return [0x7f7fffff, 0x7f7fffff, bits(0.25), 0, bits(0.125)];
        case 'unselected-nonfinite': return floats(Array.from({length: used}, (_, i) => (i + 1) / 32));
        default: throw new Error('Unknown pattern');
    }
}
function fixtures(wg) {
    const geometry = [[1, 1], [5, 3], [16, 4], [16, 16], [512, 10], [64, 31], [64, 32], [64, 33], [wg + 1, wg - 1], [wg, wg]];
    const modes = ['same', 'inside', 'before', 'aligned-prefix'];
    const finite = geometry.map(([experts, used], i) => ({name: `e${experts}-u${used}`, experts, used, pattern: 'positive', layout: modes[i % 4], edge: false, clampInplace: !!(i % 2), duplicates: i === 4, prefix: i % 2 ? 256 : 0, sharedIds: i % 2 === 0}));
    for (const layout of modes) finite.push({name: 'layout-' + layout, experts: 512, used: 10, pattern: 'positive', layout, edge: false, clampInplace: true, duplicates: false, prefix: 256, sharedIds: true});
    for (const sharedIds of [false, true]) finite.push({name: 'disjoint-' + (sharedIds ? 'same-buffer-ids' : 'separate-ids'), experts: 512, used: 10, pattern: 'positive', layout: 'disjoint', edge: false, clampInplace: true, duplicates: false, prefix: 256, sharedIds});
    const special = [['sum-tree', 5, false], ['direct-div', 3, false], ['cancellation', 7, false], ['below-clamp', 3, false], ['clamp-threshold', 4, false], ['clamp-range', 5, false],
        ['signed-zero', 5, true], ['subnormal', 5, true], ['qnan', 5, true], ['snan', 5, true], ['+inf', 5, true], ['-inf', 5, true], ['mixed-inf', 5, true], ['overflow', 5, true], ['unselected-nonfinite', 5, false]];
    return [...finite, ...special.map(([pattern, used, edge], i) => ({name: pattern, experts: 17, used, pattern, edge, layout: modes[i % 4], clampInplace: !!(i % 2), duplicates: false, prefix: 256, sharedIds: i % 2 === 0}))].filter(x => x.name.includes(option('--filter', ''))).slice(0, limit);
}

try {
    let adapter = await owner.requestAdapter(); check(adapter, 'No adapter');
    report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize'].map(k => [k, adapter.info[k]]));
    report.adapter.features = [...adapter.features];
    for (const wg of workgroups) {
        if (report.devices.length) {adapter = await owner.requestAdapter(); check(adapter, 'No fresh adapter');}
        for (const k of ['vendor', 'architecture', 'device', 'description']) check(adapter.info[k] === report.adapter[k], 'Adapter changed');
        check([256, 512, 1024].includes(wg) && wg <= adapter.limits.maxComputeInvocationsPerWorkgroup, 'WG unsupported');
        device = await adapter.requestDevice({requiredFeatures: ['shader-f16'], requiredLimits: {maxComputeInvocationsPerWorkgroup: wg, maxComputeWorkgroupSizeX: wg}});
        const alignment = device.limits.minStorageBufferOffsetAlignment;
        check(alignment === 256 && device.limits.maxComputeInvocationsPerWorkgroup === wg, 'Factory/fixture limits mismatch');
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        device.lost.then(info => {if (info.reason !== 'destroyed') report.errors.push('Device lost: ' + info.message);});
        const deviceRow = {wg, features: [...device.features], alignment, pipelines: []}; report.devices.push(deviceRow);
        function upload(bytes, usage, label) {const buffer = device.createBuffer({size: bytes.length, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label}); live.add(buffer); device.queue.writeBuffer(buffer, 0, bytes); return buffer;}
        const pipelines = {};
        for (const name of ['get_rows', 'sum_rows', 'clamp', 'clamp_inplace', 'div', 'router_tail', 'merged']) {
            const path = name === 'merged' ? join(lab, 'generated', String(wg), 'router_tail_merged.wgsl') : join(lab, 'generated', String(wg), name + '.wgsl');
            const code = readFileSync(path), expected = name === 'merged' ? integrated.shaderSha256[String(wg)]["router_tail_merged.wgsl"] : name === 'router_tail' ? integrated.shaderSha256[String(wg)]['router_tail.wgsl'] : original.generatedShaderSha256[String(wg)][name + '.wgsl'];
            check(hash(code) === expected, 'Factory shader hash mismatch');
            device.pushErrorScope('validation');
            const module = device.createShaderModule({code: code.toString(), label: name}), info = await module.getCompilationInfo();
            check(!info.messages.some(m => m.type === 'error'), 'WGSL compilation error: ' + JSON.stringify(info.messages));
            pipelines[name] = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}, label: name});
            const error = await device.popErrorScope(); check(!error, 'Pipeline validation error: ' + error?.message);
            deviceRow.pipelines.push({name, sha256: expected, messages: info.messages});
        }
        function binding(buffer, offset, bytes) {const aligned = Math.floor(offset / alignment) * alignment; return {buffer, offset: aligned, size: offset - aligned + bytes, elements: (offset - aligned) / 4};}
        function group(pipeline, resources) {return device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({binding, resource: {buffer: resource.buffer, offset: resource.offset, size: resource.size}}))});}
        function uniform(data, label) {const bytes = Buffer.from(new Uint32Array(data).buffer), buffer = upload(bytes, GPUBufferUsage.UNIFORM, label); return {bytes, buffer, offset: 0, size: bytes.length};}
        async function readAll(buffers) {
            const encoder = device.createCommandEncoder(), copies = [];
            for (const [name, buffer] of Object.entries(buffers)) {
                const target = device.createBuffer({size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}); live.add(target);
                encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size); copies.push([name, target]);
            }
            device.queue.submit([encoder.finish()]); const results = {};
            await Promise.all(copies.map(async ([name, buffer]) => {await buffer.mapAsync(GPUMapMode.READ); results[name] = Buffer.from(buffer.getMappedRange().slice(0)); buffer.unmap(); buffer.destroy(); live.delete(buffer);}));
            return results;
        }
        for (const spec of fixtures(wg)) {
            const existing = new Set(live), src = spec.prefix + 12;
            const disjoint = spec.layout === 'disjoint';
            const dst = disjoint ? Math.ceil((src + spec.experts * 4 + 19 * 4) / alignment) * alignment + 12 : spec.layout === 'same' ? src : spec.layout === 'inside' ? src + 4 : spec.layout === 'before' ? spec.prefix + 4 : src + spec.experts * 4 + 4;
            const end = Math.max(src + spec.experts * 4, dst + spec.used * 4), unionBegin = Math.min(Math.floor(src / alignment), Math.floor(dst / alignment)) * alignment;
            const unionEnd = Math.max(src + spec.experts * 4, dst + spec.used * 4);
            const arenaIdsOffset = Math.ceil((unionEnd + 19 * 4) / alignment) * alignment + 20;
            const baseInitial = Buffer.alloc(spec.sharedIds ? arenaIdsOffset + (spec.experts + 19) * 4 : end + 19 * 4); words(baseInitial).fill(0x7ea5c3f1);
            const idsOffset = spec.prefix + 20, idsInitial = Buffer.alloc(idsOffset + (spec.experts + 19) * 4); words(idsInitial).fill(0x7fffffff);
            const gOffset = 12, sOffset = 20, cOffset = 28;
            const sentinel = n => {const b = Buffer.alloc(n); words(b).fill(0x7ea5c3f1); return b;};
            const gInitial = sentinel(gOffset + (spec.used + 19) * 4), sInitial = sentinel(sOffset + 20 * 4), cInitial = sentinel(cOffset + 20 * 4);
            const initialBuffers = {baseline: baseInitial, candidate: baseInitial, gathered: gInitial, sum: sInitial, clamped: cInitial};
            if (!spec.sharedIds) initialBuffers.ids = idsInitial;
            const buffers = Object.fromEntries(Object.entries(initialBuffers).map(([k, bytes]) => [k, upload(bytes, GPUBufferUsage.STORAGE, k)]));
            const sourceBinding = binding(buffers.baseline, src, spec.experts * 4), destinationBinding = binding(buffers.baseline, dst, spec.used * 4);
            const idsBinding = binding(spec.sharedIds ? buffers.baseline : buffers.ids, spec.sharedIds ? arenaIdsOffset : idsOffset, spec.used * 4);
            const candidateIdsBinding = binding(spec.sharedIds ? buffers.candidate : buffers.ids, spec.sharedIds ? arenaIdsOffset : idsOffset, spec.used * 4);
            const gBinding = binding(buffers.gathered, gOffset, spec.used * 4), sBinding = binding(buffers.sum, sOffset, 4), cBinding = binding(buffers.clamped, cOffset, 4);
            const params = {
                get_rows: uniform([sourceBinding.elements, idsBinding.elements, gBinding.elements, 1, spec.experts, spec.experts, 1, spec.experts, spec.experts, 1, spec.used, spec.used, 1, spec.used, 1, 1, 1, 1], 'get-rows-params'),
                sum_rows: uniform([gBinding.elements, sBinding.elements, spec.used, spec.used, spec.used, spec.used, 1, 1], 'sum-params'),
                clamp: uniform([1, sBinding.elements, spec.clampInplace ? sBinding.elements : cBinding.elements, 1, 1, 1, 1, 1, 1, 1, bits(2 ** -14), bits(Infinity)], 'clamp-params'),
                div: uniform([spec.used, gBinding.elements, spec.clampInplace ? sBinding.elements : cBinding.elements, destinationBinding.elements, 1, spec.used, spec.used, spec.used, 1, 1, 1, 1, spec.used, 1, 1, 1, 1, 1, 1], 'div-params'),
                merged: uniform([disjoint ? sourceBinding.elements : (src - unionBegin) / 4, candidateIdsBinding.elements, disjoint ? destinationBinding.elements : (dst - unionBegin) / 4, spec.experts, spec.experts, spec.used, spec.used, 1, bits(2 ** -14), bits(Infinity)], 'merged-params'),
            };
            const clampName = spec.clampInplace ? 'clamp_inplace' : 'clamp';
            const groups = {
                get_rows: group(pipelines.get_rows, [sourceBinding, idsBinding, gBinding, params.get_rows]),
                sum_rows: group(pipelines.sum_rows, [gBinding, sBinding, params.sum_rows]),
                clamp: group(pipelines[clampName], spec.clampInplace ? [sBinding, params.clamp] : [sBinding, cBinding, params.clamp]),
                div: group(pipelines.div, [gBinding, spec.clampInplace ? sBinding : cBinding, destinationBinding, params.div]),
                merged: disjoint ? group(pipelines.router_tail, [binding(buffers.candidate, src, spec.experts * 4), candidateIdsBinding, binding(buffers.candidate, dst, spec.used * 4), params.merged]) : group(pipelines.merged, [{buffer: buffers.candidate, offset: unionBegin, size: unionEnd - unionBegin}, candidateIdsBinding, params.merged]),
            };
            const row = {wg, ...spec, sourceByteOffset: src, outputByteOffset: dst, unionBinding: {offset: unionBegin, size: unionEnd - unionBegin}, rounds: []}; report.cases.push(row);
            for (let round = 0; round < 2; round++) {
                const arena = Buffer.from(baseInitial), ids = Buffer.from(idsInitial), a = words(arena), d = words(ids), selected = patterns(spec.pattern, spec.used, round);
                for (let expert = 0; expert < spec.experts; expert++) a[src / 4 + expert] = spec.pattern === 'unselected-nonfinite' ? [0x7fc12345, 0x7f800000, 0xff800000][expert % 3] : bits(((expert * 73 + round * 19) % 997 + 1) / 4096);
                for (let slot = 0; slot < spec.used; slot++) {
                    const expert = ((spec.duplicates ? slot % Math.max(1, Math.floor(spec.used / 2)) : slot) + round * 3) % spec.experts;
                    d[idsOffset / 4 + slot] = expert; if (selected) a[src / 4 + expert] = selected[slot];
                }
                if (spec.sharedIds) ids.copy(arena, arenaIdsOffset, idsOffset);
                for (const name of ['baseline', 'candidate']) device.queue.writeBuffer(buffers[name], 0, arena);
                for (const [name, b] of Object.entries({ids, gathered: gInitial, sum: sInitial, clamped: cInitial})) if (buffers[name]) device.queue.writeBuffer(buffers[name], 0, b);
                const low = spec.pattern === 'clamp-range' ? (round ? -0.25 : 0.25) : 2 ** -14, high = spec.pattern === 'clamp-range' ? (round ? 0.75 : 0.5) : Infinity;
                words(params.clamp.bytes).set([bits(low), bits(high)], 10); words(params.merged.bytes).set([bits(low), bits(high)], 8);
                for (const entry of Object.values(params)) device.queue.writeBuffer(entry.buffer, 0, entry.bytes);
                device.pushErrorScope('validation');
                const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
                for (const [key, pipeline, grid] of [['get_rows', 'get_rows', Math.ceil(spec.used / wg)], ['sum_rows', 'sum_rows', 1], ['clamp', clampName, 1], ['div', 'div', Math.ceil(spec.used / wg)], ['merged', disjoint ? 'router_tail' : 'merged', 1]]) {
                    pass.setPipeline(pipelines[pipeline]); pass.setBindGroup(0, groups[key]); pass.dispatchWorkgroups(grid, 1, 1);
                }
                pass.end(); device.queue.submit([encoder.finish()]);
                const result = await readAll({...buffers, ...Object.fromEntries(Object.entries(params).map(([k, p]) => ['params-' + k, p.buffer]))});
                const error = await device.popErrorScope(); check(!error, 'Case validation: ' + error?.message);
                const baseline = words(result.baseline).slice(dst / 4, dst / 4 + spec.used), candidate = words(result.candidate).slice(dst / 4, dst / 4 + spec.used);
                const differences = {bitDifferences: 0, classDifferences: 0, signedZeroDifferences: 0, nanPayloadDifferences: 0, examples: []}, classes = {baseline: {}, candidate: {}};
                for (let slot = 0; slot < spec.used; slot++) {
                    const bc = classify(baseline[slot]), cc = classify(candidate[slot]); classes.baseline[bc] = (classes.baseline[bc] ?? 0) + 1; classes.candidate[cc] = (classes.candidate[cc] ?? 0) + 1;
                    if (baseline[slot] !== candidate[slot]) {differences.bitDifferences++; if (bc !== cc) differences.classDifferences++; if (bc === 'nan' && cc === 'nan') differences.nanPayloadDifferences++; if (bc.endsWith('zero') && cc.endsWith('zero')) differences.signedZeroDifferences++; if (differences.examples.length < 12) differences.examples.push({slot, baseline: hex(baseline[slot]), candidate: hex(candidate[slot]), baselineClass: bc, candidateClass: cc});}
                }
                for (const name of ['baseline', 'candidate']) {
                    const out = words(result[name]), before = words(arena);
                    for (let i = 0; i < out.length; i++) if (i < dst / 4 || i >= dst / 4 + spec.used) check(out[i] === before[i], 'Union input/guard mutation outside permitted output: ' + name);
                }
                if (spec.sharedIds) for (const name of ['baseline', 'candidate']) check(result[name].subarray(arenaIdsOffset).equals(ids.subarray(idsOffset)), 'IDS input mutation in shared arena');
                else check(result.ids.equals(ids), 'IDS input mutation');
                for (const [name, entry] of Object.entries(params)) check(result['params-' + name].equals(entry.bytes), 'Uniform mutation');
                for (const [name, bytes, offset, count] of [['gathered', gInitial, gOffset, spec.used], ['sum', sInitial, sOffset, 1], ['clamped', cInitial, cOffset, spec.clampInplace ? 0 : 1]]) {
                    const out = words(result[name]), before = words(bytes); for (let i = 0; i < out.length; i++) if (i < offset / 4 || i >= offset / 4 + count) check(out[i] === before[i], 'Intermediate guard mutation');
                }
                row.rounds.push({round, ...differences, classes, checkedOutputWords: spec.used, allUnionBytesOutsideOutputAndIdsAndUniformsUnchanged: true,
                    initialArenaSha256: hash(arena), initialIdsSha256: hash(ids), baselineArenaSha256: hash(result.baseline), candidateArenaSha256: hash(result.candidate), firstOutputWords: [...baseline.slice(0, 12)].map(hex)});
                check(spec.edge || !differences.bitDifferences, 'Finite alias bit difference');
            }
            row.status = row.rounds.some(r => r.bitDifferences) ? 'edge-difference-observed' : 'bit-identical';
            console.log(JSON.stringify({wg, case: spec.name, layout: spec.layout, status: row.status}));
            await device.queue.onSubmittedWorkDone(); for (const buffer of [...live]) if (!existing.has(buffer)) {buffer.destroy(); live.delete(buffer);}
        }
        await device.queue.onSubmittedWorkDone(); device.destroy(); device = undefined;
    }
    check(!report.errors.length, 'Uncaptured GPU validation error');
    report.summary = {cases: report.cases.length, rounds: 0, checkedOutputWords: 0, finiteBitDifferences: 0, edgeBitDifferences: 0, edgeClassDifferences: 0};
    for (const c of report.cases) for (const r of c.rounds) {report.summary.rounds++; report.summary.checkedOutputWords += r.checkedOutputWords; report.summary[c.edge ? 'edgeBitDifferences' : 'finiteBitDifferences'] += r.bitDifferences; if (c.edge) report.summary.edgeClassDifferences += r.classDifferences;}
    report.status = report.summary.edgeBitDifferences ? 'finite-passed-edge-differences-observed' : 'passed';
} catch (error) {report.status = 'failed'; report.failure = {message: error.message, stack: error.stack};}
finally {for (const buffer of live) buffer.destroy(); device?.destroy(); writeFileSync(output, JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify({status: report.status, summary: report.summary, failure: report.failure}));}
process.exitCode = report.status === 'failed' ? 1 : 0;
