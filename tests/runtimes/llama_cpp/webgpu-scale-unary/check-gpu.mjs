import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [shaderDirectory, output, webgpuModule, order] = process.argv.slice(2);
const hash = value => createHash('sha256').update(value).digest('hex');
const report = {
  ok: false, scope: 'Actual factory-generated shader versus exact upstream staged shader and independent F32 CPU reference; software adapter does not establish browser/model/target throughput',
  node: process.version, executable: process.execPath, arguments: process.execArgv, pipelineOrder: order,
  tolerance: { absolute: 2e-6, relative: 2e-5, scope: 'Predeclared finite-output limits, checked independently against staged GPU and F32 CPU arithmetic. Bit differences are reported separately.' },
  nonfinitePolicy: 'No sanitize/clamp/new algebra is introduced. NaN/Inf CPU classification and staged/fused classifications are reported separately; only observed staged/fused classification parity is checked. This does not promise IEEE NaN payloads or universal WGSL nonfinite behavior.',
  cases: [], errors: [], shaderSha256: {}, compilationMessages: [],
};
let device, owner;
const buffers = new Set();
const programs = new Map();
const invariant = (condition, message) => { if (!condition) throw new Error(message); };
const align = (value, alignment) => Math.ceil(value / alignment) * alignment;
const makeBuffer = descriptor => { const buffer = device.createBuffer(descriptor); buffers.add(buffer); return buffer; };
function uniform(words) {
  const buffer = makeBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, words);
  return buffer;
}
function scaleParams(n, source, destination, scale, bias) {
  const words = new Uint32Array([source, destination, n, n, n, n, n, n, n, n, 1, 1, 0, 0]);
  const floats = new Float32Array(words.buffer); floats[12] = scale; floats[13] = bias;
  return words;
}
function unaryParams(n, source, destination) { return new Uint32Array([n, source, destination, 1, n, n, n, n, 1, 1]); }
function fusedParams(n, source, destination, coefficients) {
  const words = new Uint32Array([n, source, destination, 0, 0, 0, 0, 0]);
  new Float32Array(words.buffer).set(coefficients, 4); return words;
}
function classify(value) {
  if (Number.isNaN(value)) return 'nan';
  if (value === Infinity) return '+inf';
  if (value === -Infinity) return '-inf';
  if (Object.is(value, -0)) return '-zero';
  if (value === 0) return '+zero';
  return 'finite';
}
function reference(input, key, coefficients) {
  const [scale, bias, finalScale, finalBias] = coefficients.map(Math.fround);
  return Float32Array.from(input, value => {
    const scaled = Math.fround(Math.fround(value * scale) + bias);
    const denominator = Math.fround(1 + Math.fround(Math.exp(-scaled)));
    const activated = Math.fround((key & 1 ? 1 : scaled) / denominator);
    return key & 2 ? Math.fround(Math.fround(activated * finalScale) + finalBias) : activated;
  });
}
function measure(actual, expected) {
  let maximumAbsolute = 0, maximumRelative = 0, mismatches = 0;
  for (let i = 0; i < actual.length; ++i) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) { if (classify(actual[i]) !== classify(expected[i])) ++mismatches; continue; }
    const absolute = Math.abs(actual[i] - expected[i]);
    maximumAbsolute = Math.max(maximumAbsolute, absolute);
    maximumRelative = Math.max(maximumRelative, absolute / Math.max(Math.abs(expected[i]), 1e-30));
    if (absolute > report.tolerance.absolute + report.tolerance.relative * Math.abs(expected[i])) ++mismatches;
  }
  return { maximumAbsolute, maximumRelative, mismatches };
}
async function execute(test, staged) {
  const initialBuffers = new Set(buffers);
  try {
    const { input, key, coefficients, prefix, mode } = test;
    const n = input.length;
    const alignment = device.limits.minStorageBufferOffsetAlignment;
    const sourceOffset = prefix;
    const destinationOffset = mode === 'inplace' ? sourceOffset : mode === 'same-buffer-disjoint' ? align(sourceOffset + n * 4 + alignment, alignment) + prefix % alignment : prefix + 16;
    const sourceSize = align(sourceOffset + n * 4 + 32, 4);
    const destinationSize = align(destinationOffset + n * 4 + 32, 4);
    const combinedSize = Math.max(sourceSize, destinationSize);
    const initialSource = new Uint8Array(mode === 'same-buffer-disjoint' || mode === 'inplace' ? combinedSize : sourceSize).fill(0x39);
    new Float32Array(initialSource.buffer, sourceOffset, n).set(input);
    const source = makeBuffer({ size: initialSource.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const destination = mode === 'separate' ? makeBuffer({ size: destinationSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }) : source;
    const initialDestination = mode === 'separate' ? new Uint8Array(destinationSize).fill(0xa7) : initialSource;
    device.queue.writeBuffer(source, 0, initialSource);
    if (destination !== source) device.queue.writeBuffer(destination, 0, initialDestination);
    const sourceAligned = Math.floor(sourceOffset / alignment) * alignment;
    const destinationAligned = Math.floor(destinationOffset / alignment) * alignment;
    const sourceRange = { buffer: source, offset: sourceAligned, size: sourceOffset - sourceAligned + n * 4 };
    const destinationRange = { buffer: destination, offset: destinationAligned, size: destinationOffset - destinationAligned + n * 4 };
    const sourceIndex = (sourceOffset - sourceAligned) / 4;
    const destinationIndex = (destinationOffset - destinationAligned) / 4;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const groups = Math.ceil(n / 128);
    const x = groups > 2 ? 2 : groups, y = Math.ceil(groups / x);
    function dispatch(programName, entries, words) {
      const pipeline = programs.get(programName);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...entries.map((resource, binding) => ({ binding, resource })), { binding: entries.length, resource: { buffer: uniform(words) } }] });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(x, y);
    }
    if (!staged) {
      dispatch('fused-' + key, key & 4 ? [sourceRange] : [sourceRange, destinationRange], fusedParams(n, sourceIndex, destinationIndex, coefficients));
    } else if (key & 4) {
      dispatch('scale-inplace', [sourceRange], scaleParams(n, sourceIndex, sourceIndex, coefficients[0], coefficients[1]));
      dispatch((key & 1 ? 'SIGMOID' : 'SILU') + '-inplace', [sourceRange], unaryParams(n, sourceIndex, sourceIndex));
      if (key & 2) dispatch('scale-inplace', [sourceRange], scaleParams(n, sourceIndex, sourceIndex, coefficients[2], coefficients[3]));
    } else {
      const temporaryA = makeBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
      const firstRange = { buffer: temporaryA, offset: 0, size: n * 4 };
      dispatch('scale', [sourceRange, firstRange], scaleParams(n, sourceIndex, 0, coefficients[0], coefficients[1]));
      const temporaryB = key & 2 ? makeBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE }) : undefined;
      const secondRange = temporaryB ? { buffer: temporaryB, offset: 0, size: n * 4 } : destinationRange;
      dispatch(key & 1 ? 'SIGMOID' : 'SILU', [firstRange, secondRange], unaryParams(n, 0, temporaryB ? 0 : destinationIndex));
      if (temporaryB) dispatch('scale', [secondRange, destinationRange], scaleParams(n, 0, destinationIndex, coefficients[2], coefficients[3]));
    }
    pass.end();
    const extra = destination === source ? 0 : initialSource.length;
    const readback = makeBuffer({ size: initialDestination.length + extra, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(destination, 0, readback, 0, initialDestination.length);
    if (extra) encoder.copyBufferToBuffer(source, 0, readback, initialDestination.length, initialSource.length);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(readback.getMappedRange()).slice();
    readback.unmap();
    for (let i = 0; i < initialDestination.length; ++i) {
      if (i < destinationOffset || i >= destinationOffset + n * 4) invariant(raw[i] === initialDestination[i], 'Destination/source guard changed');
    }
    if (extra) invariant(raw.subarray(initialDestination.length).every((value, i) => value === initialSource[i]), 'Out-of-place source changed');
    return { values: new Float32Array(raw.buffer, destinationOffset, n).slice(), bits: new Uint32Array(raw.buffer, destinationOffset, n).slice() };
  } finally {
    for (const buffer of buffers) if (!initialBuffers.has(buffer)) { buffer.destroy(); buffers.delete(buffer); }
  }
}
try {
  const { create, globals } = await import(pathToFileURL(webgpuModule).href);
  Object.assign(globalThis, globals);
  owner = create(['backend=vulkan', 'enable-dawn-features=allow_unsafe_apis']);
  globalThis[Symbol.for('scale-unary-test.gpu-owner')] = owner;
  const adapter = await owner.requestAdapter();
  invariant(adapter, 'No WebGPU adapter');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
  report.features = [...adapter.features].sort();
  device = await adapter.requestDevice({ requiredLimits: { minStorageBufferOffsetAlignment: adapter.limits.minStorageBufferOffsetAlignment } });
  report.storageBufferAlignment = device.limits.minStorageBufferOffsetAlignment;
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  device.pushErrorScope('validation');
  const names = [...Array(8)].map((_, key) => 'fused-' + key).concat(['scale', 'scale-inplace', 'SILU', 'SIGMOID', 'SILU-inplace', 'SIGMOID-inplace']);
  if (order === 'reverse') names.reverse();
  for (const name of names) {
    const code = readFileSync(join(shaderDirectory, name + '.wgsl'), 'utf8');
    report.shaderSha256[name] = hash(code);
    const module = device.createShaderModule({ code });
    const information = await module.getCompilationInfo();
    report.compilationMessages.push(...information.messages.map(message => ({ name, type: message.type, message: message.message })));
    invariant(!information.messages.some(message => message.type === 'error'), 'WGSL compilation failed: ' + name);
    programs.set(name, await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
  }
  let bitDifferences = 0;
  const prefixes = [0, 4, device.limits.minStorageBufferOffsetAlignment - 4, device.limits.minStorageBufferOffsetAlignment + 4];
  for (let key = 0; key < 8; ++key) for (const n of [1, 4, 7, 127, 128, 129, 257, 1025])
  for (let prefixIndex = 0; prefixIndex < prefixes.length; ++prefixIndex) for (const mode of (key & 4 ? ['inplace'] : ['separate', 'same-buffer-disjoint'])) {
    const coefficients = prefixIndex % 2 ? [0.3, -0.07, 1.7, 0.03] : [0.25, 0.125, 2, -0.125];
    const input = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.618 + n) * 12);
    const test = { input, key, coefficients, prefix: prefixes[prefixIndex], mode };
    const staged = await execute(test, true), fused = await execute(test, false), cpu = reference(input, key, coefficients);
    const stagedError = measure(fused.values, staged.values), cpuError = measure(fused.values, cpu);
    const bits = fused.bits.reduce((count, value, i) => count + Number(value !== staged.bits[i]), 0);
    bitDifferences += bits;
    report.cases.push({ key, n, prefix: test.prefix, mode, coefficients, finite: true, stagedError, cpuError, bitDifferences: bits, guards: true, sourcePreservedWhenOutOfPlace: true });
    invariant(stagedError.mismatches === 0 && cpuError.mismatches === 0, 'Finite arithmetic mismatch: ' + JSON.stringify(report.cases.at(-1)));
  }
  for (let key = 0; key < 8; ++key) {
    const input = new Float32Array([0, -0, Infinity, -Infinity, NaN, 3.402823466e38, -3.402823466e38, 1.175494351e-38, -1.175494351e-38, 80, -80, 1]);
    const test = { input, key, coefficients: [1, 0, 2, 0], prefix: 4, mode: key & 4 ? 'inplace' : 'separate' };
    const staged = await execute(test, true), fused = await execute(test, false), cpu = reference(input, key, test.coefficients);
    const stagedClasses = [...staged.values].map(classify), fusedClasses = [...fused.values].map(classify);
    const classificationDifferences = fusedClasses.reduce((count, value, i) => count + Number(value !== stagedClasses[i]), 0);
    report.cases.push({ key, n: input.length, finite: false, inputClasses: [...input].map(classify), stagedClasses, fusedClasses, cpuClasses: [...cpu].map(classify), stagedBits: [...staged.bits], fusedBits: [...fused.bits], classificationDifferences, guards: true });
    invariant(classificationDifferences === 0, 'Observed nonfinite staged/fused classification mismatch');
  }
  await device.queue.onSubmittedWorkDone();
  const validation = await device.popErrorScope();
  if (validation) report.errors.push(validation.message);
  invariant(report.errors.length === 0, 'WebGPU validation errors');
  report.finiteBitDifferences = bitDifferences;
  report.ok = true;
} catch (error) {
  report.error = { message: error?.message ?? String(error), stack: error?.stack };
} finally {
  for (const buffer of buffers) buffer.destroy();
  device?.destroy();
  report.ownerPresentThroughTeardown = owner !== undefined && globalThis[Symbol.for('scale-unary-test.gpu-owner')] === owner;
  writeFileSync(output, JSON.stringify(report, undefined, 2) + '\n');
  console.log(JSON.stringify({ ok: report.ok, cases: report.cases.length, finiteBitDifferences: report.finiteBitDifferences, output, error: report.error }));
}
process.exit(report.ok ? 0 : 1);
