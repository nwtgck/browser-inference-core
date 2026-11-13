// Standalone numerical check using actual factory-generated upstream WGSL.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fixture } from './fixtures.mjs';
const [directory, outputPath] = process.argv.slice(2);
if (!directory || !outputPath || process.argv.length !== 4) throw new Error('Usage: check_gpu.mjs SHADERS REPORT.json');
const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE);
Object.assign(globalThis, globals);
const ownerKey = Symbol('Q4 standalone Dawn owner');
const report = { schema: 1, scope: 'Actual prepared Q4_K factory vs unchanged upstream WG256, and actual upstream CPU dequantizer; standalone, no real-model or target-device speed claim', cases: [], benchmarks: [], errors: [], ok: false };
const buffers = new Set();
let device;
try {
  globalThis[ownerKey] = create([]);
  const adapter = await globalThis[ownerKey].requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize'].map(key => [key, adapter.info[key]]));
  report.features = [...adapter.features];
  const sg = adapter.info.subgroupMaxSize;
  if (![4, 8, 16, 32, 64, 128].includes(sg) || ![4, 8, 16, 32, 64, 128].includes(adapter.info.subgroupMinSize) || adapter.info.subgroupMinSize > sg) throw new Error('Unsupported subgroup size range');
  for (const feature of ['subgroups', 'shader-f16']) if (!adapter.features.has(feature)) throw new Error('Missing feature: ' + feature);
  const timed = process.env.LCB_Q4_BENCHMARKS === '1';
  if (timed && !adapter.features.has('timestamp-query')) throw new Error('Benchmark requires timestamp-query');
  device = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'shader-f16', ...(timed ? ['timestamp-query'] : [])] });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  const pipelines = new Map();
  for (const kind of ['ordinary', 'indexed']) for (const size of [32, 64, 128, 256]) {
    if (size < sg) continue;
    const code = await readFile(join(directory, `${kind}-${size}.wgsl`), 'utf8');
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    pipelines.set(kind + size, await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
  }
  const cases = JSON.parse(await readFile(join(directory, 'cases.json'), 'utf8'));
  if (!Array.isArray(cases) || !cases.length) throw new Error('No actual-factory cases');
  function allocate(size, usage) {
    const buffer = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
    buffers.add(buffer);
    return buffer;
  }
  function upload(data, usage = GPUBufferUsage.STORAGE) {
    const buffer = allocate(data.byteLength, usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
  async function read(buffer) {
    const stage = allocate(buffer.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, stage, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(stage.getMappedRange().slice(0));
    stage.unmap(); stage.destroy(); buffers.delete(stage);
    return words;
  }
  function binding(pipeline, inputs) {
    return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: inputs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  function dispatch(pipeline, group, groups, split, query = null, resolve = null) {
    // Same 2D flattening as compute_2d_workgroups, including a trailing WG.
    const height = Math.ceil(groups / split), width = Math.ceil(groups / height);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(query ? { timestampWrites: {
      querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
    } } : {});
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(width, height); pass.end();
    if (query) encoder.resolveQuerySet(query, 0, 2, resolve, 0);
    device.queue.submit([encoder.finish()]);
  }
  function compareOutputs(data, results) {
    let mismatch = 0, guardErrors = 0, nonfinite = 0, cpuErrors = 0, nonzero = 0, maxAbsoluteError = 0, maxErrorToAbsoluteSum = 0;
    const floats = results.map(words => new Float32Array(words.buffer));
    for (let i = 0; i < data.initial.length; i++) {
      if (i >= data.offD && i < data.offD + data.outputWords) {
        mismatch += Number(results[0][i] !== results[1][i]);
        nonzero += Number((results[0][i] & 0x7fffffff) !== 0);
        const j = (i - data.offD) * 2, expected = data.cpu[j], sumAbs = data.cpu[j + 1];
        for (const output of floats) {
          const error = Math.abs(output[i] - expected), scaled = error / Math.max(1, sumAbs);
          maxAbsoluteError = Math.max(maxAbsoluteError, error); maxErrorToAbsoluteSum = Math.max(maxErrorToAbsoluteSum, scaled);
          nonfinite += Number(!Number.isFinite(output[i]));
          // Bound against an independent dequantized dot, allowing ordinary F32
          // accumulation/reduction rounding without demanding CPU/GPU bit parity.
          cpuErrors += Number(error > 64 * 2 ** -23 * Math.max(1, sumAbs));
        }
      } else for (const result of results) guardErrors += Number(result[i] !== data.initial[i]);
    }
    return { mismatch, guardErrors, nonfinite, cpuErrors, nonzero, maxAbsoluteError, maxErrorToAbsoluteSum };
  }
  async function checkInputs(pairs) {
    let inputErrors = 0;
    for (const [original, buffer] of pairs) {
      const actual = await read(buffer), expected = new Uint32Array(original.buffer);
      for (let i = 0; i < expected.length; i++) inputErrors += Number(actual[i] !== expected[i]);
    }
    return inputErrors;
  }
  for (const [caseIndex, spec] of cases.entries()) {
    if (!['ordinary', 'indexed'].includes(spec.kind) || !Number.isInteger(spec.k) || spec.k <= 0 || spec.k % 256 || !Number.isInteger(spec.m) || spec.m <= 0 || ![0, 1, 2].includes(spec.mode) || typeof spec.sharedInput !== 'boolean' || typeof spec.broadcast !== 'boolean' || ![32, 64, 128, 256].includes(spec.workgroups?.[sg])) throw new Error('Invalid actual-factory case');
    const chosen = spec.workgroups[sg], data = fixture(spec);
    const cpuBytes = await readFile(join(process.env.LCB_Q4_CPU_REFERENCE_DIR, `${caseIndex}.f64`));
    if (cpuBytes.length !== data.outputWords * 16) throw new Error('CPU reference output count mismatch');
    data.cpu = new Float64Array(cpuBytes.buffer.slice(cpuBytes.byteOffset, cpuBytes.byteOffset + cpuBytes.byteLength));
    const a = upload(data.weights), x = upload(data.input), ids = upload(data.ids), u = upload(data.params, GPUBufferUsage.UNIFORM);
    const results = [];
    for (const size of [256, chosen]) {
      const dst = upload(data.initial), pipeline = pipelines.get(spec.kind + size);
      dispatch(pipeline, binding(pipeline, spec.kind === 'indexed' ? [a, x, ids, dst, u] : [a, x, dst, u]), data.groups, spec.mode ? 17 : 65535);
      results.push(await read(dst));
    }
    const metrics = compareOutputs(data, results);
    const inputErrors = await checkInputs([[data.weights, a], [data.input, x], [data.ids, ids], [data.params, u]]);
    report.cases.push({ ...spec, chosen, outputWords: data.outputWords, ...metrics, inputErrors });
    await device.queue.onSubmittedWorkDone(); for (const buffer of buffers) buffer.destroy(); buffers.clear();
    if (metrics.mismatch || metrics.guardErrors || inputErrors || metrics.nonfinite || metrics.cpuErrors || !metrics.nonzero || report.errors.length) throw new Error('Numerical/boundary failure: ' + JSON.stringify(report.cases.at(-1)));
  }
  if (timed) {
    const median = values => { const sorted = [...values].sort((a, b) => a - b), n = sorted.length; return (sorted[Math.floor(n / 2)] + sorted[Math.ceil(n / 2) - 1]) / 2; };
    for (const [kind, k, m] of [['ordinary', 2048, 2048], ['ordinary', 2048, 10752], ['ordinary', 2048, 6144], ['ordinary', 10752, 2048], ['indexed', 512, 2048], ['indexed', 2048, 512]]) {
      const spec = cases.find(row => row.kind === kind && row.k === k && row.m === m && !row.sharedInput && !row.broadcast);
      if (!spec) throw new Error('Benchmark lacks actual-factory case');
      const data = fixture(spec), chosen = spec.workgroups[sg];
      const cpuBytes = await readFile(join(process.env.LCB_Q4_CPU_REFERENCE_DIR, `${cases.indexOf(spec)}.f64`));
      if (cpuBytes.length !== data.outputWords * 16) throw new Error('Benchmark CPU reference count mismatch');
      data.cpu = new Float64Array(cpuBytes.buffer.slice(cpuBytes.byteOffset, cpuBytes.byteOffset + cpuBytes.byteLength));
      const a = upload(data.weights), x = upload(data.input), ids = upload(data.ids), u = upload(data.params, GPUBufferUsage.UNIFORM), dst = upload(data.initial);
      const entries = [256, chosen].map(size => { const pipeline = pipelines.get(kind + size); return { pipeline, group: binding(pipeline, kind === 'indexed' ? [a, x, ids, dst, u] : [a, x, dst, u]) }; });
      const query = device.createQuerySet({ type: 'timestamp', count: 2 }), resolve = allocate(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const pairs = [], baseline = [], selected = [];
      for (let rep = 0; rep < 20; rep++) {
        const pair = {};
        for (const index of rep % 2 ? [1, 0] : [0, 1]) {
          const entry = entries[index]; dispatch(entry.pipeline, entry.group, data.groups, 65535, query, resolve);
          const words = await read(resolve), stamps = new BigUint64Array(words.buffer), interval = stamps[1] - stamps[0];
          if (interval < 0n || interval > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Invalid GPU timestamp interval');
          pair[index ? 'selectedMs' : 'baselineMs'] = Number(interval) / 1e6;
        }
        if (rep >= 4) { pairs.push(pair); baseline.push(pair.baselineMs); selected.push(pair.selectedMs); }
      }
      query.destroy();
      const wallPairs = [], wallBaseline = [], wallSelected = [], wallBatchDispatches = 20;
      if (process.env.LCB_Q4_WALL_BENCHMARKS === '1') {
        for (let rep = 0; rep < 12; rep++) {
          const pair = {};
          for (const index of rep % 2 ? [1, 0] : [0, 1]) {
            const entry = entries[index];
            const begin = performance.now();
            for (let batch = 0; batch < wallBatchDispatches; batch++) dispatch(entry.pipeline, entry.group, data.groups, 65535);
            await device.queue.onSubmittedWorkDone();
            pair[index ? 'selectedMs' : 'baselineMs'] = (performance.now() - begin) / wallBatchDispatches;
          }
          if (rep >= 4) { wallPairs.push(pair); wallBaseline.push(pair.baselineMs); wallSelected.push(pair.selectedMs); }
        }
      }
      // Check actual output, input preservation and CPU reference again after
      // all timed dispatches, including the unprofiled queue batches.
      const afterResults = [];
      for (const entry of entries) { dispatch(entry.pipeline, entry.group, data.groups, 65535); afterResults.push(await read(dst)); }
      const afterTiming = compareOutputs(data, afterResults);
      afterTiming.inputErrors = await checkInputs([[data.weights, a], [data.input, x], [data.ids, ids], [data.params, u]]);
      report.benchmarks.push({ kind, k, m, chosen, fallbackControl: chosen === 256, slots: kind === 'indexed' ? 8 : 1, warmupPairs: 4, measuredPairs: 16, ordering: 'AB/BA alternating; same buffers; GPU pass timestamps', pairs, baselineMedianMs: median(baseline), selectedMedianMs: median(selected), medianRatio: median(selected) > 0 ? median(baseline) / median(selected) : null, zeroDurationPasses: [...baseline, ...selected].filter(value => value === 0).length,
        wall: wallPairs.length ? { method: '20 separate command submissions per queue-completion batch, no timestamp writes; elapsed milliseconds per dispatch', warmupPairs: 4, measuredPairs: 8, wallBatchDispatches, pairs: wallPairs, baselineMedianMs: median(wallBaseline), selectedMedianMs: median(wallSelected), medianRatio: median(wallSelected) > 0 ? median(wallBaseline) / median(wallSelected) : null } : null, afterTiming });
      if (afterTiming.mismatch || afterTiming.guardErrors || afterTiming.inputErrors || afterTiming.nonfinite || afterTiming.cpuErrors || !afterTiming.nonzero || report.errors.length) throw new Error('Post-timing numerical/boundary failure: ' + JSON.stringify(report.benchmarks.at(-1)));
      await device.queue.onSubmittedWorkDone(); for (const buffer of buffers) buffer.destroy(); buffers.clear();
    }
  }
  report.ok = report.cases.length === cases.length && report.cases.length > 0 && report.errors.length === 0;
} catch (error) {
  report.errors.push(error.stack ?? String(error));
} finally {
  for (const buffer of buffers) buffer.destroy(); device?.destroy();
  await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ ok: report.ok, cases: report.cases.length, errors: report.errors }));
// Keep the Dawn owner reachable through all queue and final report awaits.
process.exit(report.ok ? 0 : 1);
