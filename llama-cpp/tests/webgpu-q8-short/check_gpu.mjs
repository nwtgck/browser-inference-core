// Standalone test only. Actual factory-generated WGSL; no hand-copied kernel.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
const [shaderDirectory, outputPath] = process.argv.slice(2);
if (!shaderDirectory || !outputPath || process.argv.length !== 4) {
  throw new Error('Usage: check_gpu.mjs SHADER_DIRECTORY REPORT.json');
}
const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE);
Object.assign(globalThis, globals);
// Dawn's async runner depends on this owner, including during await/mapAsync.
const ownerKey = Symbol('q8 standalone Dawn owner');
const report = {
  schema: 1,
  scope: 'Standalone actual-factory Q8_0 WGSL comparison; not browser/model performance',
  cases: [], benchmarks: [], errors: [], ok: false,
};
const buffers = new Set();
let device;
try {
  globalThis[ownerKey] = create([]);
  const forcedGc = process.env.LCB_Q8_FORCE_GC === '1';
  if (forcedGc && typeof global.gc !== 'function') throw new Error('Forced GC requested without --expose-gc');
  const adapter = await globalThis[ownerKey].requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description',
    'subgroupMinSize', 'subgroupMaxSize'].map(key => [key, adapter.info[key]]));
  report.features = [...adapter.features];
  const sg = adapter.info.subgroupMaxSize;
  if (!Number.isInteger(sg) || sg < 4 || sg > 128 || (sg & (sg - 1)) !== 0) {
    throw new Error('Invalid or unsupported maximum subgroup size');
  }
  for (const feature of ['subgroups', 'shader-f16']) {
    if (!adapter.features.has(feature)) throw new Error('Missing required feature: ' + feature);
  }
  const timed = process.env.LCB_Q8_BENCHMARKS === '1';
  if (timed && !adapter.features.has('timestamp-query')) throw new Error('Benchmark requires timestamp-query');
  device = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'shader-f16',
    ...(timed ? ['timestamp-query'] : [])] });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  const pipelines = new Map();
  for (const size of [32, 64, 128, 256]) {
    if (size < sg) continue;
    const code = await readFile(join(shaderDirectory, `q8-${size}.wgsl`), 'utf8');
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    pipelines.set(size, await device.createComputePipelineAsync({
      layout: 'auto', compute: { module, entryPoint: 'main' },
    }));
  }
  function select(k, m) {
    return m >= 1024 && k <= 1024 ? Math.max(32, sg, 2 ** Math.ceil(Math.log2(k / 8))) : 256;
  }
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
    const staging = allocate(buffer.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    buffers.delete(staging);
    return result;
  }
  async function release() {
    await device.queue.onSubmittedWorkDone();
    for (const buffer of buffers) buffer.destroy();
    buffers.clear();
    if (forcedGc) global.gc();
  }
  function fixture(k, m, mode, batched) {
    const blocks = k / 32;
    const a2 = batched ? 2 : 1, a3 = batched ? 2 : 1;
    const broadcast2 = batched ? 2 : 1, broadcast3 = batched ? 2 : 1;
    const x2 = a2 * broadcast2, x3 = a3 * broadcast3;
    const offsetA = mode ? 1 : 0, offsetX = mode ? 3 : 0, offsetD = mode ? 7 : 0;
    const stride01 = blocks + (mode ? 1 : 0);
    const stride02 = m * stride01 + (mode ? 3 : 0);
    const stride03 = a2 * stride02 + (mode ? 5 : 0);
    const stride11 = k + (mode ? 5 : 0);
    const stride12 = stride11 + (mode ? 3 : 0);
    const stride13 = x2 * stride12 + (mode ? 7 : 0);
    const weights = new Uint8Array(Math.ceil(((offsetA + a3 * stride03) * 34 + 16) / 4) * 4);
    weights.fill(0x42);
    const view = new DataView(weights.buffer);
    const scales = mode === 2 ? [0x0000, 0x3800, 0xbc00, 0x3000, 0x4400] : [0x3800, 0x3c00, 0x4000, 0x4400];
    for (let i3 = 0; i3 < a3; i3++) for (let i2 = 0; i2 < a2; i2++) {
      for (let row = 0; row < m; row++) for (let block = 0; block < blocks; block++) {
        const start = (offsetA + i3 * stride03 + i2 * stride02 + row * stride01 + block) * 34;
        view.setUint16(start, scales[(row + block + i2 + i3) % scales.length], true);
        for (let j = 0; j < 32; j++) view.setInt8(start + 2 + j,
          ((j * 31 + block * 73 + row * 47 + i2 * 13 + i3 * 19 + 1) % 256) - 128);
      }
    }
    const input = new Float32Array(offsetX + x3 * stride13 + 16);
    input.fill(99.5);
    for (let i3 = 0; i3 < x3; i3++) for (let i2 = 0; i2 < x2; i2++) for (let i = 0; i < k; i++) {
      input[offsetX + i3 * stride13 + i2 * stride12 + i] = mode === 2
        ? Math.fround(Math.sin(i * 0.43 + i2 + i3) * 2 ** ((i % 17) - 8))
        : (((i * 53 + i2 * 17 + i3 * 41) % 1009) - 504) / 1024;
    }
    const outputWords = m * x2 * x3;
    const initial = new Uint32Array(offsetD + outputWords + 16);
    initial.fill(0x7eabcdef);
    const params = new Uint32Array([offsetA, offsetX, offsetD, m, 1, k, stride01, stride11,
      stride02, stride12, stride03, stride13, a2, a3, broadcast2, broadcast3]);
    return { weights, input, initial, params, offsetD, outputWords, groups: Math.ceil(m / 4) * x2 * x3 };
  }
  function binding(pipeline, inputs) {
    return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: inputs.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }
  function dispatch(pipeline, group, groups, split = 0, query = null, resolve = null) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(query ? { timestampWrites: {
      querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
    } } : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    const width = split ? Math.min(groups, split) : groups;
    pass.dispatchWorkgroups(width, Math.ceil(groups / width));
    pass.end();
    if (query) encoder.resolveQuerySet(query, 0, 2, resolve, 0);
    device.queue.submit([encoder.finish()]);
  }
  const cases = [];
  for (let k = 32; k <= 1024; k += 32) for (const mode of [0, 1, 2]) {
    cases.push({ k, m: mode === 0 ? 1024 : 1025, mode, batched: false });
  }
  for (const k of [32, 256, 320, 512, 640, 1024]) {
    cases.push({ k, m: 1025, mode: 2, batched: true });
  }
  for (const [k, m] of [[320, 1023], [1056, 1025], [2560, 1025], [320, 10240], [640, 2560]]) {
    cases.push({ k, m, mode: 2, batched: false });
  }
  for (const spec of cases) {
    const data = fixture(spec.k, spec.m, spec.mode, spec.batched);
    const a = upload(data.weights), x = upload(data.input), u = upload(data.params, GPUBufferUsage.UNIFORM);
    const chosen = select(spec.k, spec.m);
    const results = [];
    for (const size of [256, chosen]) {
      const dst = upload(data.initial), pipeline = pipelines.get(size);
      dispatch(pipeline, binding(pipeline, [a, x, dst, u]), data.groups, spec.mode === 1 || spec.batched ? 17 : 0);
      results.push(await read(dst));
    }
    let mismatch = 0, guardErrors = 0, inputErrors = 0, nonfinite = 0, nonzero = 0;
    for (let i = 0; i < data.initial.length; i++) {
      if (i >= data.offsetD && i < data.offsetD + data.outputWords) {
        mismatch += Number(results[0][i] !== results[1][i]);
        nonfinite += Number((results[0][i] & 0x7f800000) === 0x7f800000);
        nonzero += Number((results[0][i] & 0x7fffffff) !== 0);
      } else for (const result of results) guardErrors += Number(result[i] !== data.initial[i]);
    }
    for (const [original, buffer] of [[data.weights, a], [data.input, x], [data.params, u]]) {
      const result = await read(buffer), words = new Uint32Array(original.buffer);
      for (let i = 0; i < words.length; i++) inputErrors += Number(words[i] !== result[i]);
    }
    report.cases.push({ ...spec, chosen, outputWords: data.outputWords,
      mismatch, guardErrors, inputErrors, nonfinite, nonzero });
    await release();
    if (mismatch || guardErrors || inputErrors || nonfinite || !nonzero || report.errors.length) {
      throw new Error('Numerical/boundary failure: ' + JSON.stringify(report.cases.at(-1)));
    }
  }
  if (timed) {
    const median = array => {
      const sorted = [...array].sort((a, b) => a - b), middle = sorted.length / 2;
      return (sorted[Math.floor(middle)] + sorted[Math.ceil(middle) - 1]) / 2;
    };
    for (const [k, m] of [[320, 10240], [640, 2560], [256, 10240], [1024, 2560], [2560, 10240]]) {
      const data = fixture(k, m, 2, false), a = upload(data.weights), x = upload(data.input);
      const u = upload(data.params, GPUBufferUsage.UNIFORM), dst = upload(data.initial);
      const chosen = select(k, m);
      const entries = [256, chosen].map(size => ({ size, pipeline: pipelines.get(size),
        group: binding(pipelines.get(size), [a, x, dst, u]) }));
      const query = device.createQuerySet({ type: 'timestamp', count: 2 });
      const resolve = allocate(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const oldTimes = [], newTimes = [], pairs = [];
      for (let rep = 0; rep < 16; rep++) {
        const pair = {};
        for (const index of rep % 2 ? [1, 0] : [0, 1]) {
          const entry = entries[index];
          dispatch(entry.pipeline, entry.group, data.groups, 0, query, resolve);
          const words = await read(resolve), stamps = new BigUint64Array(words.buffer);
          const delta = stamps[1] - stamps[0];
          if (delta < 0n || delta > BigInt(Number.MAX_SAFE_INTEGER)) {
            throw new Error('Invalid benchmark timestamp interval');
          }
          const elapsed = Number(delta) / 1e6;
          pair[index ? 'selectedMs' : 'baselineMs'] = elapsed;
        }
        if (rep >= 4) {
          oldTimes.push(pair.baselineMs); newTimes.push(pair.selectedMs); pairs.push(pair);
        }
      }
      query.destroy();
      report.benchmarks.push({ k, m, chosen, warmupPairs: 4, measuredPairs: 12,
        ordering: 'AB/BA alternating; same buffers; GPU pass timestamps only', pairs,
        baselineMedianMs: median(oldTimes), selectedMedianMs: median(newTimes),
        zeroDurationPasses: [...oldTimes, ...newTimes].filter(value => value === 0).length,
        medianRatio: median(newTimes) > 0 ? median(oldTimes) / median(newTimes) : null,
      });
      await release();
    }
  }
  report.ok = report.cases.length === cases.length && report.cases.length > 0 && report.errors.length === 0;
  report.forcedGc = forcedGc;
} catch (error) {
  report.errors.push(error.stack ?? String(error));
} finally {
  for (const buffer of buffers) buffer.destroy();
  device?.destroy();
  await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ ok: report.ok, cases: report.cases.length, errors: report.errors }));
// Keep the owner strongly reachable through the final asynchronous write above.
process.exit(report.ok ? 0 : 1);
