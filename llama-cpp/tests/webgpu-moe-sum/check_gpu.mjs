// Production-factory shader vs real upstream binary ADD chain.
// Standalone GPU check, not compiled-runtime/model validation.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
const [referenceDir, reportPath] = process.argv.slice(2);
if (!referenceDir || !reportPath || process.argv.length !== 4 || !process.env.LCB_TEST_WEBGPU_MODULE) {
  throw new Error('Usage: LCB_TEST_WEBGPU_MODULE=... node check_gpu.mjs REFERENCE_DIR REPORT.json');
}
const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE);
Object.assign(globalThis, globals);
const owner = Symbol('experimental MoE Dawn lifetime');
const report = { version: 1, scope: 'standalone shader parity; not compiled runtime/model validation', cases: [], benchmarks: [], errors: [], ok: false };
let device;
const live = new Set();
const sentinel = 0x4f123456;
function words(n) { const a = new Uint32Array(n); a.fill(sentinel); return a; }
function allocate(bytes, usage) { const b = device.createBuffer({ size: bytes, usage }); live.add(b); return b; }
async function pipeline(code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(x => x.type === 'error');
  if (errors.length) throw new Error(errors.map(x => x.message).join('\n'));
  return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
}
function group(pipe, bindings, parameters, allocated) {
  const p = allocate(Math.ceil(parameters.length * 4 / 16) * 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  allocated.add(p); device.queue.writeBuffer(p, 0, new Uint32Array(parameters));
  return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [...bindings.map((x, binding) => ({ binding, resource: x })), { binding: bindings.length, resource: { buffer: p } }] });
}
try {
  if (typeof global.gc !== 'function') throw new Error('Run with --expose-gc; lifetime stress is required');
  globalThis[owner] = create(['backend=vulkan']);
  const adapter = await globalThis[owner].requestAdapter();
  if (!adapter || !adapter.features.has('shader-f16')) throw new Error('Required shader-f16 adapter unavailable');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info[k]]));
  const benchmark = process.env.MOE_SUM_BENCHMARKS === '1';
  if (benchmark && !adapter.features.has('timestamp-query')) throw new Error('timestamp-query unavailable');
  device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', ...(benchmark ? ['timestamp-query'] : [])] });
  device.addEventListener('uncapturederror', e => report.errors.push(e.error.message));
  const first = await pipeline(await readFile(join(referenceDir, 'SRC_OVERLAP.wgsl'), 'utf8'), 'upstream-first-ADD');
  const next = await pipeline(await readFile(join(referenceDir, 'INPLACE.wgsl'), 'utf8'), 'upstream-following-ADD');
  const variants = new Map();
  for (const experts of [3, 4, 8, 10, 16, 32]) {
    let code = await readFile(join(referenceDir, `moe-${experts}.wgsl`), 'utf8');
    if (process.env.MOE_SUM_NEGATIVE === '1') code = code.replace('= sum;', '= sum + vec4<f32>(1.0);');
    variants.set(experts, await pipeline(code, `production-${experts}`));
  }
  for (const width of [4, 12, 128, 512, 1024, 2560, 3072, 4096]) for (const experts of [3, 4, 8, 10, 16, 32]) for (const tokens of [1, 3, 31]) for (const shared of [false, true]) {
    const allocated = new Set();
    try {
      device.pushErrorScope('validation');
      const prefix = shared ? 12 : 4, outPrefix = shared ? 20 : 4;
      const count = width * tokens, rootCount = count * experts;
      const rootWords = Math.ceil((prefix + rootCount + 16) / 64) * 64;
      const outWords = Math.ceil((outPrefix + count + 16) / 64) * 64;
      const input = words(rootWords), output = words(outWords);
      const f = new Float32Array(input.buffer);
      // Include cancellation, signed zero, large and tiny normal finite values.
      const values = [0, -0, 1, -1, 2 ** 24, -(2 ** 24), 0.25, -0.125, 2 ** -100, -(2 ** -100), 17.75, -63.5];
      for (let i = 0; i < rootCount; ++i) f[prefix + i] = values[(i * 7 + Math.floor(i / width) * 3) % values.length];
      const cpu = new Float32Array(count);
      for (let token = 0; token < tokens; ++token) for (let col = 0; col < width; ++col) {
        let sum = f[prefix + token * width * experts + col];
        for (let e = 1; e < experts; ++e) sum = Math.fround(sum + f[prefix + (token * experts + e) * width + col]);
        cpu[token * width + col] = sum;
      }
      const cpuBits = new Uint32Array(cpu.buffer);
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      let rootBuffer, referenceBuffer, candidateBuffer, rootOffset = 0, referenceOffset = 0, candidateOffset = 0;
      if (shared) {
        rootBuffer = referenceBuffer = candidateBuffer = allocate((rootWords + 2 * outWords) * 4, storage);
        allocated.add(rootBuffer); referenceOffset = rootWords * 4; candidateOffset = (rootWords + outWords) * 4;
      } else {
        rootBuffer = allocate(rootWords * 4, storage); referenceBuffer = allocate(outWords * 4, storage); candidateBuffer = allocate(outWords * 4, storage);
        for (const b of [rootBuffer, referenceBuffer, candidateBuffer]) allocated.add(b);
      }
      device.queue.writeBuffer(rootBuffer, rootOffset, input); device.queue.writeBuffer(referenceBuffer, referenceOffset, output); device.queue.writeBuffer(candidateBuffer, candidateOffset, output);
      const src = { buffer: rootBuffer, offset: rootOffset, size: rootWords * 4 };
      const ref = { buffer: referenceBuffer, offset: referenceOffset, size: outWords * 4 };
      const dst = { buffer: candidateBuffer, offset: candidateOffset, size: outWords * 4 };
      const encoder = device.createCommandEncoder();
      const staged = [];
      for (let e = 1; e < experts; ++e) {
        const isFirst = e === 1, pipe = isFirst ? first : next;
        const stride0 = isFirst ? width * experts : width;
        const params = [count, isFirst ? prefix : outPrefix, prefix + e * width, outPrefix,
          1, stride0, stride0 * tokens, stride0 * tokens,
          1, width * experts, width * experts * tokens, width * experts * tokens,
          width, tokens, 1, width, tokens, 1, 1];
        const bind = group(pipe, isFirst ? [src, ref] : [ref, src], params, allocated);
        staged.push({ pipe, bind, x: Math.ceil(count / 256), y: 1 });
        const pass = encoder.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(count / 256)); pass.end();
      }
      const selected = variants.get(experts);
      const bind = group(selected, [src, dst], [prefix / 4, outPrefix / 4, width / 4, tokens], allocated);
      const pass = encoder.beginComputePass(); pass.setPipeline(selected); pass.setBindGroup(0, bind);
      const groups = Math.ceil(count / 4 / 128), x = Math.min(groups, 3), y = Math.ceil(groups / x);
      pass.dispatchWorkgroups(x, y); pass.end();
      const readback = allocate((rootWords + 2 * outWords) * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST); allocated.add(readback);
      encoder.copyBufferToBuffer(rootBuffer, rootOffset, readback, 0, rootWords * 4);
      encoder.copyBufferToBuffer(referenceBuffer, referenceOffset, readback, rootWords * 4, outWords * 4);
      encoder.copyBufferToBuffer(candidateBuffer, candidateOffset, readback, (rootWords + outWords) * 4, outWords * 4);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const got = new Uint32Array(readback.getMappedRange()); let mismatch = 0, cpuMismatch = 0, guard = 0, changedInput = 0;
      for (let i = 0; i < rootWords; ++i) if (got[i] !== input[i]) ++changedInput;
      for (let i = 0; i < outWords; ++i) {
        const a = got[rootWords + i], b = got[rootWords + outWords + i];
        if (i >= outPrefix && i < outPrefix + count) { if (a !== b) ++mismatch; if (a !== cpuBits[i - outPrefix]) ++cpuMismatch; }
        else { if (a !== sentinel) ++guard; if (b !== sentinel) ++guard; }
      }
      readback.unmap();
      const validation = await device.popErrorScope();
      const row = { width, experts, tokens, shared, outputElements: count, mismatch, cpuMismatch, guard, changedInput, validation: validation?.message ?? null };
      report.cases.push(row);
      if (mismatch || cpuMismatch || guard || changedInput || validation) throw new Error(`Case failed: ${JSON.stringify(row)}`);
      if (benchmark && width === 2560 && experts === 10 && !shared) {
        // Same-pass staged baseline: no artificial extra pass boundary per ADD.
        const fused = [{ pipe: selected, bind, x, y }];
        const pairs = [];
        async function measure(operations) {
          const queries = device.createQuerySet({ type: 'timestamp', count: 2 });
          const resolved = allocate(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
          const read = allocate(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          try {
            const enc = device.createCommandEncoder();
            const compute = enc.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
            for (const op of operations) { compute.setPipeline(op.pipe); compute.setBindGroup(0, op.bind); compute.dispatchWorkgroups(op.x, op.y); }
            compute.end(); enc.resolveQuerySet(queries, 0, 2, resolved, 0); enc.copyBufferToBuffer(resolved, 0, read, 0, 16);
            device.queue.submit([enc.finish()]); await read.mapAsync(GPUMapMode.READ);
            const times = new BigUint64Array(read.getMappedRange()); const ms = Number(times[1] - times[0]) / 1e6;
            read.unmap(); return ms;
          } finally { queries.destroy(); resolved.destroy(); read.destroy(); live.delete(resolved); live.delete(read); }
        }
        for (let repeat = 0; repeat < 16; ++repeat) {
          const order = repeat % 2 ? [['fused', fused], ['staged', staged]] : [['staged', staged], ['fused', fused]];
          const pair = { repeat, warmup: repeat < 4, order: order.map(x => x[0]) };
          for (const [name, ops] of order) pair[name + 'Ms'] = await measure(ops);
          pairs.push(pair);
        }
        report.benchmarks.push({ width, experts, tokens, scope: 'software GPU; both alternatives use one compute pass; standalone only', pairs });
      }
      if (typeof global.gc === 'function') global.gc();
    } finally { for (const b of allocated) { b.destroy(); live.delete(b); } }
  }
  report.ok = report.cases.length === 288 && report.errors.length === 0;
} catch (error) { report.errors.push(error.stack ?? String(error)); }
finally {
  for (const b of live) b.destroy();
  device?.destroy();
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  delete globalThis[owner];
}
console.log(JSON.stringify({ ok: report.ok, cases: report.cases.length, errors: report.errors }));
process.exit(report.ok ? 0 : 1);
