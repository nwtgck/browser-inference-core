// Actual factory output and upstream CPU references; no browser/model/timing claim.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { checkSubgroupGeometry, subgroupGeometryCode } from '../webgpu-q8-subgroup/subgroup-geometry.mjs';

const [shaderDirectory, fixtureDirectory, reportPath, subgroupText] = process.argv.slice(2);
if (process.argv.length !== 6 || !shaderDirectory || !fixtureDirectory || !reportPath
    || ![16, 32, 64].includes(Number(subgroupText))) {
  throw new Error('Usage: check_gpu.mjs SHADER_DIRECTORY FIXTURE_DIRECTORY REPORT.json FIXED_SUBGROUP');
}
const subgroup = Number(subgroupText);
const hash = data => createHash('sha256').update(data).digest('hex');
const require = (condition, message) => { if (!condition) throw new Error(message); };
const ownerKey = Symbol('IQ3_S actual factory WebGPU owner');
const report = {
  schema: 1,
  scope: 'Actual prepared IQ3_S indexed factory/dispatch decisions and upstream CPU dequantization plus Float64 dot; synthetic inputs only, no model quality or target-device speed claim',
  promotionPolicy: 'The conservative error bound is an arithmetic sanity gate. Baseline/Float64 error observations require independent review.',
  reference: 'Every output, Float64 [dot,sumabs] from unchanged pinned CPU dequantize_row_iq3_s and actual uploaded Float32 input; normal half scales only',
  status: 'started', cases: [], errors: [],
};
const buffers = new Set();
let device;
try {
  const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE);
  Object.assign(globalThis, globals);
  globalThis[ownerKey] = create([]);
  const adapter = await globalThis[ownerKey].requestAdapter();
  require(adapter, 'No WebGPU adapter');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize']
    .map(key => [key, adapter.info[key]]));
  require(adapter.info.subgroupMinSize === subgroup && adapter.info.subgroupMaxSize === subgroup, 'Unexpected fixed subgroup limits');
  device = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'shader-f16'] });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  device.lost.then(info => { if (info.reason !== 'destroyed') report.errors.push('Device lost: ' + info.message); });
  function allocate(bytes, usage) {
    const buffer = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage });
    buffers.add(buffer); return buffer;
  }
  function upload(data, usage) {
    require(data.byteLength % 4 === 0, 'Upload padding must be a multiple of four');
    const buffer = allocate(data.byteLength, usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(buffer, 0, data); return buffer;
  }
  async function read(buffer) {
    const staging = allocate(buffer.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = Buffer.from(staging.getMappedRange().slice(0));
      staging.unmap(); return data;
    } finally { staging.destroy(); buffers.delete(staging); }
  }
  async function pipeline(code, label) {
    const module = device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    require(!info.messages.some(message => message.type === 'error'), JSON.stringify(info.messages));
    return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  function bind(pipeline, inputs) {
    return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: inputs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  // Test-only observation: fixed limits permit fewer active lanes per subgroup.
  const probePipeline = await pipeline(subgroupGeometryCode, 'subgroup geometry probe');
  const probe = upload(new Uint32Array(128 * 5), GPUBufferUsage.STORAGE);
  const probeEncoder = device.createCommandEncoder();
  const probePass = probeEncoder.beginComputePass();
  probePass.setPipeline(probePipeline); probePass.setBindGroup(0, bind(probePipeline, [probe]));
  probePass.dispatchWorkgroups(1); probePass.end(); device.queue.submit([probeEncoder.finish()]);
  const bytes = await read(probe);
  const geometry = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  report.subgroupProbe = { shaderSha256: hash(subgroupGeometryCode),
    ...checkSubgroupGeometry({ geometry, advertisedWidth: subgroup }) };
  probe.destroy(); buffers.delete(probe);

  const identity = JSON.parse(await readFile(join(dirname(resolve(shaderDirectory)), 'source.json'), 'utf8'));
  const decisions = JSON.parse(await readFile(join(shaderDirectory, 'case-decisions.json'), 'utf8'));
  const fixtures = JSON.parse(await readFile(join(fixtureDirectory, 'manifest.json'), 'utf8'));
  require(Array.isArray(fixtures) && fixtures.length > 0 && decisions.length === fixtures.length
    && new Set(decisions.map(row => row.name)).size === fixtures.length, 'Invalid fixture decision coverage');
  report.source = identity;
  require(hash(await readFile(join(fixtureDirectory, 'manifest.json'))) === identity.fixtureManifestSha256, 'Fixture manifest identity mismatch');
  const pipelines = new Map();
  for (const file of new Set(['iq3s-baseline.wgsl', ...decisions.map(row => row.selectedFile)])) {
    const code = await readFile(join(shaderDirectory, file));
    require(hash(code) === identity.shaderSha256[file], 'Factory WGSL hash mismatch: ' + file);
    pipelines.set(file, await pipeline(code.toString(), file));
  }
  const caseOutputs = new Map();
  for (const spec of fixtures) {
    const decision = decisions.find(row => row.name === spec.name);
    require(decision && spec.params.length === 13, 'Missing actual factory decision');
    const data = {};
    for (const key of ['weights', 'input', 'ids', 'reference']) {
      data[key] = await readFile(join(fixtureDirectory, spec[key + 'File']));
      require(hash(data[key]) === spec.hashes[key], 'Fixture hash mismatch: ' + key);
    }
    const truth = new Float64Array(data.reference.buffer.slice(data.reference.byteOffset, data.reference.byteOffset + data.reference.byteLength));
    const outputs = spec.m * spec.slots, offset = spec.params[3];
    require(truth.length === outputs * 2, 'Reference output count mismatch');
    const skippedSlots = new Set(Array.from({ length: spec.slots }, (_, slot) => slot)
      .filter(slot => data.ids.readUInt32LE((spec.params[2] + slot) * 4) === 0xffffffff));
    require(JSON.stringify([...skippedSlots]) === JSON.stringify(spec.skippedSlots), 'Skipped-slot fixture metadata mismatch');
    if (skippedSlots.size > 0) {
      require(decision.selectedFile === `iq3s-independent-sg${subgroup}.wgsl`
        && decision.wgSize === 128 && decision.outputsPerWg === 4 * (128 / subgroup),
      'Skipped-slot case did not select the independent-subgroup implementation: ' + spec.name);
    }
    const initial = new Uint32Array(offset + outputs + 16); initial.fill(0x7eabcdef);
    const params = new Uint32Array(16); params.set(spec.params);
    const a = upload(data.weights, GPUBufferUsage.STORAGE), x = upload(data.input, GPUBufferUsage.STORAGE);
    const ids = upload(data.ids, GPUBufferUsage.STORAGE), uniform = upload(params, GPUBufferUsage.UNIFORM);
    const row = { name: spec.name, k: spec.k, m: spec.m, slots: spec.slots, params: spec.params,
      baselineCase: spec.baselineCase, skippedSlots: [...skippedSlots],
      actualDecision: decision, referenceOutputs: outputs, variants: [] };
    report.cases.push(row);
    let baseline;
    for (const variant of [{ name: 'upstream-reduction', file: 'iq3s-baseline.wgsl', outputsPerWg: 4 },
      { name: 'actual-selection', file: decision.selectedFile, outputsPerWg: decision.outputsPerWg }]) {
      const dst = upload(initial, GPUBufferUsage.STORAGE), entry = pipelines.get(variant.file);
      const count = Math.ceil(spec.m / variant.outputsPerWg) * spec.slots;
      const width = Math.min(count, spec.params[0] ? 17 : device.limits.maxComputeWorkgroupsPerDimension);
      const height = Math.ceil(count / width);
      require(height <= device.limits.maxComputeWorkgroupsPerDimension, 'Fixture grid exceeds device limit');
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(entry); pass.setBindGroup(0, bind(entry, [a, x, ids, dst, uniform]));
      pass.dispatchWorkgroups(width, height); pass.end(); device.queue.submit([encoder.finish()]);
      const bytes = await read(dst);
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      if (!baseline) baseline = Uint32Array.from(words);
      let guardErrors = 0, nonfinite = 0, boundFailures = 0, bitDifferences = 0;
      let skippedZeroErrors = 0, unchangedSlotErrors = 0;
      const paired = spec.baselineCase ? caseOutputs.get(spec.baselineCase + ':' + variant.name) : undefined;
      require(!spec.baselineCase || paired?.length === outputs, 'Missing paired baseline output');
      let maxAbsError = 0, maxRelativeError = 0, maxNormalizedError = 0, squared = 0, relativeSamples = 0;
      const operations = 2 * spec.k + 256 + 32, gamma = operations * 2 ** -23 / (1 - operations * 2 ** -23);
      for (let i = 0; i < words.length; i++) {
        if (i < offset || i >= offset + outputs) { guardErrors += Number(words[i] !== initial[i]); continue; }
        const j = i - offset, actual = floats[i], reference = truth[j * 2], sumAbs = truth[j * 2 + 1];
        if (skippedSlots.has(Math.floor(j / spec.m))) {
          skippedZeroErrors += Number(words[i] !== 0);
        } else if (paired) {
          unchangedSlotErrors += Number(words[i] !== paired[j]);
        }
        nonfinite += Number(!Number.isFinite(actual));
        const error = Math.abs(actual - reference);
        boundFailures += Number(error > gamma * sumAbs);
        bitDifferences += Number(words[i] !== baseline[i]);
        maxAbsError = Math.max(maxAbsError, error); maxNormalizedError = Math.max(maxNormalizedError, error / (sumAbs + 1e-300));
        squared += error * error;
        if (Math.abs(reference) >= Math.max(1e-6, sumAbs * 1e-6)) {
          relativeSamples++; maxRelativeError = Math.max(maxRelativeError, error / Math.abs(reference));
        }
      }
      const mutations = {};
      for (const [key, buffer, expected] of [['weights', a, data.weights], ['input', x, data.input],
        ['ids', ids, data.ids], ['params', uniform, Buffer.from(params.buffer)]]) {
        mutations[key] = Number(!(await read(buffer)).equals(expected));
      }
      row.variants.push({ name: variant.name, file: variant.file, checkedOutputs: outputs, dispatchGrid: [width, height],
        guardErrors, nonfinite, boundFailures, bitDifferences, skippedZeroErrors, unchangedSlotErrors,
        maxAbsError, maxRelativeError, relativeSamples,
        maxNormalizedError, rmse: Math.sqrt(squared / outputs), gamma, mutations });
      require(!guardErrors && !nonfinite && !boundFailures && Object.values(mutations).every(value => !value), 'Numerical/boundary failure: ' + spec.name);
      require(!skippedZeroErrors && !unchangedSlotErrors, 'Skipped-slot or paired-output failure: ' + spec.name);
      caseOutputs.set(spec.name + ':' + variant.name, Uint32Array.from(words.subarray(offset, offset + outputs)));
      dst.destroy(); buffers.delete(dst);
    }
    await device.queue.onSubmittedWorkDone();
    for (const buffer of buffers) buffer.destroy(); buffers.clear();
  }
  const specializedSkipped = report.cases.filter(row => row.skippedSlots.length > 0
    && row.actualDecision.selectedFile !== 'iq3s-baseline.wgsl');
  require(specializedSkipped.some(row => row.skippedSlots.length === row.slots), 'Missing actual-selection all-skipped coverage');
  require(specializedSkipped.some(row => row.params[8] > 1 && row.skippedSlots.length < row.slots), 'Missing actual-selection per-slot-input skipped coverage');
  require(!report.errors.length, 'WebGPU validation errors');
  report.status = 'passed';
} catch (error) {
  report.status = 'failed'; report.failure = { message: error.message, stack: error.stack };
} finally {
  for (const buffer of buffers) buffer.destroy();
  device?.destroy();
  await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ status: report.status, cases: report.cases.length, failure: report.failure, errors: report.errors }));
process.exit(report.status === 'passed' ? 0 : 1);
