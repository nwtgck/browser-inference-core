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
const ownerKey = Symbol('expert standalone Dawn owner');
let activeType = 'Q5_1';
const report = {
  schema: 1,
  scope: 'Actual prepared expert-matvec factory against unchanged upstream factory; standalone, not real-model performance',
  cases: [], benchmarks: [], errors: [], ok: false,
};
const buffers = new Set();
let device;
try {
  globalThis[ownerKey] = create([]);
  const forcedGc = process.env.LCB_EXPERT_FORCE_GC === '1';
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
  const timed = process.env.LCB_EXPERT_BENCHMARKS === '1';
  if (timed && !adapter.features.has('timestamp-query')) throw new Error('Benchmark requires timestamp-query');
  device = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'shader-f16',
    ...(timed ? ['timestamp-query'] : [])] });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  const pipelines = new Map();
  for (const type of ['Q5_1', 'IQ4_NL']) for (const size of [128, 256]) {
    if (size < sg) continue;
    const code = await readFile(join(shaderDirectory, `${type}-${size}.wgsl`), 'utf8');
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    pipelines.set(type + size, await device.createComputePipelineAsync({
      layout: 'auto', compute: { module, entryPoint: 'main' },
    }));
  }
  const cases = JSON.parse(await readFile(join(shaderDirectory, 'cases.json'), 'utf8'));
  if (!Array.isArray(cases) || !cases.length) throw new Error('No factory-generated cases');
  for (const spec of cases) {
    if (!['Q5_1','IQ4_NL'].includes(spec.type) || !Number.isInteger(spec.k) || spec.k <= 0 || spec.k % 32 ||
        !Number.isInteger(spec.m) || spec.m <= 0 || ![128,256].includes(spec.chosen) ||
        ![11,512].includes(spec.storedExperts) || ![0,1,2].includes(spec.mode) ||
        typeof spec.sharedInput !== 'boolean') throw new Error('Invalid factory case');
  }
  function select(k, m, type) {
    const row = cases.find(row => row.k === k && row.m === m && row.type === type);
    if (!row) throw new Error('Benchmark shape lacks actual-factory selection');
    return row.chosen;
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
  function fixture(k, m, mode, sharedInput, storedExperts = 11) {
    const experts=storedExperts,slots=10,blocks=k/32,bytes=activeType==='Q5_1'?24:18;
    const offsetA=mode?1:0,offsetX=mode?3:0,offsetI=mode?2:0,offsetD=mode?7:0;
    const stride01=blocks+(mode?1:0),stride02=m*stride01+(mode?3:0),bNe1=sharedInput?1:slots,stride11=k+(mode?5:0);
    const weights=new Uint8Array(Math.ceil(((offsetA+experts*stride02)*bytes+16)/4)*4); weights.fill(0x42);
    const view=new DataView(weights.buffer),scales=[0x3800,0xbc00,0,0x3000,0x4000];
    for(let e=0;e<experts;e++)for(let r=0;r<m;r++)for(let b=0;b<blocks;b++) {
      const base=(offsetA+e*stride02+r*stride01+b)*bytes;
      for(let j=2;j<bytes;j++)weights[base+j]=(e*71+r*37+b*19+j*13)%256;
      view.setUint16(base,scales[(e+r+b)%scales.length],true);
      if(bytes===24)view.setUint16(base+2,scales[(e+r+b+1)%scales.length],true);
    }
    const input=new Float32Array(offsetX+bNe1*stride11+16);input.fill(99.5);
    for(let e=0;e<bNe1;e++)for(let i=0;i<k;i++)input[offsetX+e*stride11+i]=mode===2?Math.fround(Math.sin(i*0.43+e)*2**((i%17)-8)):(((i*53+e*17)%1009)-504)/1024;
    const ids=new Uint32Array(offsetI+slots+4);ids.fill(9);
    for(let i=0;i<slots;i++)ids[offsetI+i]=(i*7+3)%experts;
    ids[offsetI]=experts-1;
    ids[offsetI+slots-1]=ids[offsetI]; // duplicate selected expert must not alias output slots
    const outputWords=m*slots,initial=new Uint32Array(offsetD+outputWords+16);initial.fill(0x7eabcdef);
    const params=new Uint32Array([offsetA,offsetX,offsetI,offsetD,k,m,512,slots,bNe1,stride01,stride11,stride02,bNe1*stride11,0,0,0]);
    return {weights,input,ids,initial,params,offsetD,outputWords,groups:Math.ceil(m/4)*slots};
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
  for (const spec of cases) {
    activeType=spec.type;
    const data = fixture(spec.k, spec.m, spec.mode, spec.sharedInput, spec.storedExperts);
    const a = upload(data.weights), x = upload(data.input), ids = upload(data.ids), u = upload(data.params, GPUBufferUsage.UNIFORM);
    const chosen = spec.chosen;
    const results = [];
    for (const size of [256, chosen]) {
      const dst = upload(data.initial), pipeline = pipelines.get(activeType + size);
      dispatch(pipeline, binding(pipeline, [a, x, ids, dst, u]), data.groups, spec.mode === 1 || spec.sharedInput ? 17 : 0);
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
    for (const [original, buffer] of [[data.weights, a], [data.input, x], [data.ids, ids], [data.params, u]]) {
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
    for (const type of ['Q5_1','IQ4_NL']) for (const [k, m] of [[640, 2560], [1056, 256]]) {
      activeType=type;
      const data = fixture(k, m, 2, false), a = upload(data.weights), x = upload(data.input);
      const ids=upload(data.ids), u = upload(data.params, GPUBufferUsage.UNIFORM), dst = upload(data.initial);
      const chosen = select(k, m, type);
      const entries = [256, chosen].map(size => ({ size, pipeline: pipelines.get(activeType + size),
        group: binding(pipelines.get(activeType + size), [a, x, ids, dst, u]) }));
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
      report.benchmarks.push({ type, k, m, chosen, warmupPairs: 4, measuredPairs: 12,
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
