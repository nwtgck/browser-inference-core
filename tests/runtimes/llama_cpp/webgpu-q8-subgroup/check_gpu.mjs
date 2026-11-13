// Source-bound numerical checker. Consume prepared factory WGSL, never a copied kernel.
// Usage: node check_gpu.mjs SHADER_DIRECTORY REPORT.json MANIFEST.json
// Set LCB_TEST_WEBGPU_MODULE to the local node-webgpu module's file: URL.
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { checkSubgroupGeometry, subgroupGeometryCode } from './subgroup-geometry.mjs';

const [shaderDirectory, reportPath, manifestPath] = process.argv.slice(2);
if (!shaderDirectory || !reportPath || !manifestPath || process.argv.length !== 5) {
  throw new Error('Usage: check_gpu.mjs SHADER_DIRECTORY REPORT.json MANIFEST.json');
}

const epsilon32 = 2 ** -23;
const tiny = 1e-300;
const ownerKey = Symbol('Q8 numerical lab Dawn owner');
const buffers = new Set();
const queries = new Set();
const report = {
  schema: 1,
  scope: 'Prepared-factory standalone Q8_0 matvec numerical check; not browser, model, or Apple GPU evidence',
  promotionPolicy: 'The arithmetic bound is a sanity gate only. Review each variant against the Float64 reference and baseline independently; these local timings do not justify target-device promotion.',
  reference: {
    implementation: 'Independent JavaScript Float64 dequantization and dot, using the uploaded Float32 input values',
    coverage: 'Every output, no row sampling',
    bound: 'gamma_n * sum(abs(weight * input)) + halfSubnormalFtzAllowance; n=2*K+maxVariantWgSize+32; gamma_n=n*2^-23/(1-n*2^-23)',
    boundInterpretation: 'Conservative float32 arithmetic sanity bound; it does not establish numerical equivalence, model quality, or promotion suitability',
    halfSubnormalFtzAllowance: 'Sum of absolute terms with binary16 subnormal scales; permits allowed binary16 flush-to-zero in external payloads',
    relativeError: '|GPU-reference|/|reference| only when |reference| >= max(1e-6, sumAbs*1e-6)',
    normalizedError: '|GPU-reference|/(sumAbs+1e-300)',
  },
  cases: [], benchmarks: [], errors: [], ok: false,
};
let device;

function positiveInteger({ value, name, maximum }) {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new Error(`${name} must be an integer in [1, ${maximum}]`);
  }
  return value;
}

function halfToNumber({ bits }) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const mantissa = bits & 1023;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (exponent === 0 ? mantissa * 2 ** -24 : (1024 + mantissa) * 2 ** (exponent - 25));
}

function median({ values }) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function allocate({ size, usage }) {
  const buffer = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
  buffers.add(buffer);
  return buffer;
}

function upload({ data, usage }) {
  const buffer = allocate({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function readWords({ buffer }) {
  const staging = allocate({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return result;
  } finally {
    staging.destroy();
    buffers.delete(staging);
  }
}

async function releaseBuffers() {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of buffers) buffer.destroy();
  buffers.clear();
  if (process.env.LCB_Q8_FORCE_GC === '1') global.gc();
}

function bind({ pipeline, sources }) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: sources.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

function encode({ entry, groups, split, dispatches, query, queryResolve }) {
  const started = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(query ? { timestampWrites: {
    querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
  } } : {});
  pass.setPipeline(entry.pipeline);
  pass.setBindGroup(0, entry.group);
  const width = Math.min(groups, split || device.limits.maxComputeWorkgroupsPerDimension);
  const height = Math.ceil(groups / width);
  if (height > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Dispatch grid too large');
  for (let i = 0; i < dispatches; i++) pass.dispatchWorkgroups(width, height);
  pass.end();
  if (query) encoder.resolveQuerySet(query, 0, 2, queryResolve, 0);
  const command = encoder.finish();
  return { command, cpuEncodeMs: performance.now() - started, width, height };
}

async function fixture({ spec, manifestDirectory, maxWgSize }) {
  const { k, m, mode, batched } = spec;
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
  // Normal binary16 values with nonzero mantissas, mixed exponents/signs, and zero.
  // Subnormals are excluded here because binary16 FTZ is permitted on some devices.
  const scales = [0x0000, 0x13ad, 0xa65b, 0x2d71, 0xb5b3, 0x3835, 0xbc27, 0x4159, 0x4913];
  let external;
  if (spec.weightsPath !== undefined) {
    external = await readFile(resolve(manifestDirectory, spec.weightsPath));
    if (external.byteLength !== m * blocks * 34) {
      throw new Error(`${spec.name}: weightsPath must contain exactly M*(K/32)*34 contiguous Q8_0 bytes`);
    }
  }
  let halfSubnormalScales = 0;
  for (let i3 = 0; i3 < a3; i3++) for (let i2 = 0; i2 < a2; i2++) {
    for (let row = 0; row < m; row++) for (let block = 0; block < blocks; block++) {
      const start = (offsetA + i3 * stride03 + i2 * stride02 + row * stride01 + block) * 34;
      if (external) {
        const source = (row * blocks + block) * 34;
        weights.set(external.subarray(source, source + 34), start);
      } else {
        view.setUint16(start, scales[(row * 3 + block * 5 + i2 + i3 * 7) % scales.length], true);
        for (let j = 0; j < 32; j++) view.setInt8(start + 2 + j,
          ((j * 31 + block * 73 + row * 47 + i2 * 13 + i3 * 19 + 1) % 256) - 128);
      }
      const scaleBits = view.getUint16(start, true);
      if (!Number.isFinite(halfToNumber({ bits: scaleBits }))) throw new Error(`${spec.name}: nonfinite binary16 scale`);
      halfSubnormalScales += Number((scaleBits & 0x7c00) === 0 && (scaleBits & 0x03ff) !== 0);
    }
  }
  const input = new Float32Array(offsetX + x3 * stride13 + 16);
  input.fill(99.5);
  for (let i3 = 0; i3 < x3; i3++) for (let i2 = 0; i2 < x2; i2++) for (let i = 0; i < k; i++) {
    // Adjacent opposite signs exercise cancellation; nonbinary values and exponents exercise rounding.
    input[offsetX + i3 * stride13 + i2 * stride12 + i] = mode === 2
      ? Math.fround((i % 2 ? -1 : 1) * (0.375 + Math.sin(Math.floor(i / 2) * 0.43 + i2 * 0.7 + i3 * 1.1)) * 2 ** ((Math.floor(i / 2) % 17) - 8))
      : Math.fround((((i * 53 + i2 * 17 + i3 * 41) % 1009) - 504) / 997);
  }
  const outputWords = m * x2 * x3;
  const initial = new Uint32Array(offsetD + outputWords + 16);
  initial.fill(0x7eabcdef);
  const params = new Uint32Array([offsetA, offsetX, offsetD, m, 1, k, stride01, stride11,
    stride02, stride12, stride03, stride13, a2, a3, broadcast2, broadcast3]);
  const reference = new Float64Array(outputWords), sumAbs = new Float64Array(outputWords);
  const ftzAllowance = new Float64Array(outputWords);
  const referenceStarted = performance.now();
  for (let i3 = 0; i3 < x3; i3++) for (let i2 = 0; i2 < x2; i2++) {
    const aBase = offsetA + Math.floor(i3 / broadcast3) * stride03 + Math.floor(i2 / broadcast2) * stride02;
    const xBase = offsetX + i3 * stride13 + i2 * stride12;
    for (let row = 0; row < m; row++) {
      let total = 0, absolute = 0, ftz = 0;
      for (let block = 0; block < blocks; block++) {
        const start = (aBase + row * stride01 + block) * 34;
        const bits = view.getUint16(start, true);
        const scale = halfToNumber({ bits });
        const subnormal = (bits & 0x7c00) === 0 && (bits & 0x03ff) !== 0;
        for (let j = 0; j < 32; j++) {
          const term = scale * view.getInt8(start + 2 + j) * input[xBase + block * 32 + j];
          total += term;
          absolute += Math.abs(term);
          if (subnormal) ftz += Math.abs(term);
        }
      }
      const index = (i3 * x2 + i2) * m + row;
      reference[index] = total; sumAbs[index] = absolute; ftzAllowance[index] = ftz;
    }
  }
  const operations = 2 * k + maxWgSize + 32;
  if (operations * epsilon32 >= 1) throw new Error('Reference error bound is invalid for this K');
  return { weights, input, initial, params, offsetD, outputWords, reference, sumAbs, ftzAllowance,
    batches: x2 * x3, gamma: operations * epsilon32 / (1 - operations * epsilon32), operations,
    halfSubnormalScales, referenceMs: performance.now() - referenceStarted,
    layout: { offsetA, offsetX, offsetD, stride01, stride02, stride03, stride11, stride12, stride13,
      a2, a3, broadcast2, broadcast3 },
  };
}

function compare({ words, data, baseline }) {
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  let guardErrors = 0, nonfinite = 0, boundFailures = 0, nonzeroOutputs = 0;
  let maxAbsError = 0, maxRelativeError = 0, relativeSamples = 0, maxNormalizedError = 0;
  let squaredError = 0, maxBoundFraction = 0, epsilonNormalizedExceeded = 0;
  let exactBaselineDifferences = 0, maxBaselineAbsDifference = 0, maxBaselineUlps = 0;
  let maxBaselineNormalizedDifference = 0, betterThanBaseline = 0, worseThanBaseline = 0;
  const worst = [];
  for (let i = 0; i < words.length; i++) {
    if (i < data.offsetD || i >= data.offsetD + data.outputWords) {
      guardErrors += Number(words[i] !== data.initial[i]);
      continue;
    }
    const index = i - data.offsetD, actual = floats[i], truth = data.reference[index];
    if (!Number.isFinite(actual)) { nonfinite++; continue; }
    nonzeroOutputs += Number(actual !== 0);
    const absError = Math.abs(actual - truth);
    const bound = data.gamma * data.sumAbs[index] + data.ftzAllowance[index];
    const normalized = absError / (data.sumAbs[index] + tiny);
    const boundFraction = absError / (bound + tiny);
    maxAbsError = Math.max(maxAbsError, absError);
    maxNormalizedError = Math.max(maxNormalizedError, normalized);
    maxBoundFraction = Math.max(maxBoundFraction, boundFraction);
    epsilonNormalizedExceeded += Number(normalized > epsilon32);
    squaredError += absError * absError;
    if (Math.abs(truth) >= Math.max(1e-6, data.sumAbs[index] * 1e-6)) {
      relativeSamples++; maxRelativeError = Math.max(maxRelativeError, absError / Math.abs(truth));
    }
    if (absError > bound) {
      boundFailures++;
      if (worst.length < 5) worst.push({ index, actual, reference: truth, absError, sumAbs: data.sumAbs[index], bound });
    }
    if (baseline) {
      const baselineFloat = new Float32Array(baseline.buffer, baseline.byteOffset, baseline.length)[i];
      exactBaselineDifferences += Number(words[i] !== baseline[i]);
      const difference = Math.abs(actual - baselineFloat);
      maxBaselineAbsDifference = Math.max(maxBaselineAbsDifference, difference);
      maxBaselineNormalizedDifference = Math.max(maxBaselineNormalizedDifference, difference / (data.sumAbs[index] + tiny));
      const ordered = word => word & 0x80000000 ? 0x80000000 - (word & 0x7fffffff) : 0x80000000 + word;
      maxBaselineUlps = Math.max(maxBaselineUlps, Math.abs(ordered(words[i]) - ordered(baseline[i])));
      const baselineError = Math.abs(baselineFloat - truth);
      betterThanBaseline += Number(absError < baselineError);
      worseThanBaseline += Number(absError > baselineError);
    }
  }
  return { checkedOutputs: data.outputWords, guardErrors, nonfinite, nonzeroOutputs, boundFailures,
    maxAbsError, maxRelativeError, relativeSamples, maxNormalizedError,
    rmse: Math.sqrt(squaredError / data.outputWords), maxBoundFraction, epsilonNormalizedExceeded,
    ...(baseline ? { baselineComparison: { exactBaselineDifferences, maxAbsDifference: maxBaselineAbsDifference,
      maxNormalizedDifference: maxBaselineNormalizedDifference, maxUlps: maxBaselineUlps,
      betterThanBaseline, worseThanBaseline,
      interpretation: 'Observations only; bit differences and changes of reduction order do not set the acceptance bound' } } : {}),
    ...(worst.length ? { boundFailureExamples: worst } : {}),
  };
}

async function unchanged({ data, a, x, uniform }) {
  const result = { weights: 0, input: 0, params: 0 };
  for (const [key, original, buffer] of [['weights', data.weights, a], ['input', data.input, x], ['params', data.params, uniform]]) {
    const actual = await readWords({ buffer });
    const expected = new Uint32Array(original.buffer, original.byteOffset, original.byteLength / 4);
    for (let i = 0; i < expected.length; i++) result[key] += Number(expected[i] !== actual[i]);
  }
  return { ...result, total: result.weights + result.input + result.params };
}

function failed({ comparison, mutations }) {
  return comparison.nonfinite || comparison.boundFailures || comparison.guardErrors || mutations.total;
}

try {
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (!Array.isArray(manifest.variants) || !manifest.variants.length) throw new Error('Manifest variants must be a nonempty array');
  const variants = manifest.variants.map(value => {
    if (typeof value.name !== 'string' || !value.name || typeof value.file !== 'string' || !value.file) throw new Error('Variant name/file required');
    positiveInteger({ value: value.wgSize, name: 'wgSize', maximum: 1024 });
    positiveInteger({ value: value.outputsPerWg, name: 'outputsPerWg', maximum: 1024 });
    return value;
  });
  if (new Set(variants.map(value => value.name)).size !== variants.length) throw new Error('Variant names must be unique');
  const baselineName = manifest.baseline ?? variants[0].name;
  if (!variants.some(value => value.name === baselineName)) throw new Error('Unknown baseline variant');
  const orderedVariants = [...variants].sort((a, b) => Number(b.name === baselineName) - Number(a.name === baselineName));
  const maxWgSize = Math.max(...variants.map(value => value.wgSize));
  const defaultCases = [
    { name: 'real-ffn-up', k: 2560, m: 10240, mode: 2, batched: false },
    { name: 'real-attn-qkv', k: 2560, m: 6144, mode: 2, batched: false },
    { name: 'real-attn-out', k: 6144, m: 2560, mode: 2, batched: false },
    { name: 'real-low-rank', k: 10240, m: 320, mode: 2, batched: false },
    { name: 'tail-small', k: 32, m: 35, mode: 0, batched: false },
    { name: 'tail-offset-strides', k: 1056, m: 35, mode: 1, batched: false },
    { name: 'tail-batch-broadcast', k: 320, m: 35, mode: 2, batched: true },
  ];
  const cases = (manifest.cases ?? defaultCases).map((value, index) => {
    positiveInteger({ value: value.k, name: 'K', maximum: 1048576 });
    positiveInteger({ value: value.m, name: 'M', maximum: 1048576 });
    if (value.k % 32) throw new Error('K must be divisible by 32 for Q8_0');
    const mode = value.mode ?? 2;
    if (![0, 1, 2].includes(mode)) throw new Error('Case mode must be 0, 1 or 2');
    if (value.batched !== undefined && typeof value.batched !== 'boolean') throw new Error('batched must be boolean');
    if (value.weightsPath !== undefined && typeof value.weightsPath !== 'string') throw new Error('weightsPath must be a string');
    if (value.split !== undefined) positiveInteger({ value: value.split, name: 'split', maximum: 65535 });
    return { ...value, name: value.name ?? `case-${index}`, mode, batched: value.batched ?? false };
  });
  if (!cases.length || new Set(cases.map(value => value.name)).size !== cases.length) throw new Error('Cases must be nonempty and named uniquely');
  const benchmark = manifest.benchmarks ?? false;
  if (benchmark !== false && (typeof benchmark !== 'object' || Array.isArray(benchmark) || benchmark === null)) throw new Error('benchmarks must be false or an object');
  const forcedGc = process.env.LCB_Q8_FORCE_GC === '1';
  if (forcedGc && typeof global.gc !== 'function') throw new Error('Forced GC requested without --expose-gc');
  if (!process.env.LCB_TEST_WEBGPU_MODULE) throw new Error('LCB_TEST_WEBGPU_MODULE is required');
  const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE);
  Object.assign(globalThis, globals);
  globalThis[ownerKey] = create([]);
  const adapter = await globalThis[ownerKey].requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize']
    .map(key => [key, adapter.info[key]]));
  report.features = [...adapter.features];
  for (const variant of variants) {
    const fixed = variant.experiment?.fixedSubgroupSizeRequired;
    if (fixed !== undefined && (adapter.info.subgroupMinSize !== fixed || adapter.info.subgroupMaxSize !== fixed)) {
      throw new Error(`${variant.name} requires advertised fixed subgroup limits of ${fixed}`);
    }
  }
  report.forcedGc = forcedGc;
  report.variants = variants;
  report.baseline = baselineName;
  const requiredFeatures = new Set(variants.flatMap(value => value.requiredFeatures ?? ['subgroups', 'shader-f16']));
  if (benchmark) requiredFeatures.add('timestamp-query');
  for (const feature of requiredFeatures) if (!adapter.features.has(feature)) throw new Error(`Missing required feature: ${feature}`);
  device = await adapter.requestDevice({ requiredFeatures: [...requiredFeatures] });
  device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
  device.lost.then(info => { if (info.reason !== 'destroyed') report.errors.push(`Device lost: ${info.reason}: ${info.message}`); });
  // Fixed limits describe a maximum width, so legal partial subgroups are accepted.
  const fixedWidths = new Set(variants.map(variant => variant.experiment?.fixedSubgroupSizeRequired)
    .filter(value => value !== undefined));
  if (fixedWidths.size) {
    if (fixedWidths.size !== 1) throw new Error('Inconsistent fixed-width specializations');
    const module = device.createShaderModule({ code: subgroupGeometryCode, label: 'subgroup geometry probe' });
    const info = await module.getCompilationInfo();
    if (info.messages.some(message => message.type === 'error')) throw new Error(JSON.stringify(info.messages));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const buffer = upload({ data: new Uint32Array(128 * 5), usage: GPUBufferUsage.STORAGE });
    const entry = { pipeline, group: bind({ pipeline, sources: [buffer] }) };
    const encoded = encode({ entry, groups: 1, dispatches: 1 });
    device.queue.submit([encoded.command]);
    report.subgroupProbe = { shaderSha256: createHash('sha256').update(subgroupGeometryCode).digest('hex'),
      ...checkSubgroupGeometry({ geometry: await readWords({ buffer }), advertisedWidth: [...fixedWidths][0] }) };
    await releaseBuffers();
  }
  const pipelines = new Map();
  for (const variant of variants) {
    const code = await readFile(resolve(shaderDirectory, variant.file), 'utf8');
    const module = device.createShaderModule({ label: variant.name, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(`${variant.name}: ${errors.map(message => message.message).join('\n')}`);
    pipelines.set(variant.name, await device.createComputePipelineAsync({
      label: variant.name, layout: 'auto', compute: { module, entryPoint: 'main' },
    }));
  }
  const manifestDirectory = dirname(resolve(manifestPath));
  const numericalStarted = performance.now();
  for (const spec of cases) {
    const data = await fixture({ spec, manifestDirectory, maxWgSize });
    const a = upload({ data: data.weights, usage: GPUBufferUsage.STORAGE });
    const x = upload({ data: data.input, usage: GPUBufferUsage.STORAGE });
    const uniform = upload({ data: data.params, usage: GPUBufferUsage.UNIFORM });
    const caseReport = { ...spec, weightsPath: spec.weightsPath === undefined ? undefined : spec.weightsPath,
      weightsSource: spec.weightsPath === undefined ? 'deterministic synthetic Q8_0' : 'external contiguous Q8_0 matrix, replicated into A batches',
      referenceOutputs: data.outputWords, referenceMs: data.referenceMs, halfSubnormalScales: data.halfSubnormalScales,
      gamma: data.gamma, boundOperations: data.operations, layout: data.layout, variants: [] };
    report.cases.push(caseReport);
    let baseline;
    for (const variant of orderedVariants) {
      const dst = upload({ data: data.initial, usage: GPUBufferUsage.STORAGE });
      const pipeline = pipelines.get(variant.name);
      const entry = { pipeline, group: bind({ pipeline, sources: [a, x, dst, uniform] }) };
      const groups = Math.ceil(spec.m / variant.outputsPerWg) * data.batches;
      const encoded = encode({ entry, groups, split: spec.split ?? (spec.mode === 1 || spec.batched ? 17 : 0),
        dispatches: 1, query: undefined, queryResolve: undefined });
      device.queue.submit([encoded.command]);
      const words = await readWords({ buffer: dst });
      const comparison = compare({ words, data, baseline });
      const mutations = await unchanged({ data, a, x, uniform });
      caseReport.variants.push({ name: variant.name, groups, dispatchGrid: [encoded.width, encoded.height], ...comparison, mutations });
      if (variant.name === baselineName) baseline = words;
      dst.destroy(); buffers.delete(dst);
      if (failed({ comparison, mutations }) || report.errors.length) throw new Error(`Numerical/boundary failure in ${spec.name}/${variant.name}`);
    }
    await releaseBuffers();
  }
  report.numericalWallMs = performance.now() - numericalStarted;

  if (benchmark) {
    const warmupPairs = benchmark.warmupPairs ?? 3;
    const measuredPairs = benchmark.measuredPairs ?? 9;
    const dispatches = benchmark.dispatchesPerPass ?? 1;
    positiveInteger({ value: warmupPairs + 1, name: 'warmupPairs+1', maximum: 1001 });
    positiveInteger({ value: measuredPairs, name: 'measuredPairs', maximum: 1000 });
    positiveInteger({ value: dispatches, name: 'dispatchesPerPass', maximum: 10000 });
    if (benchmark.caseNames !== undefined && (!Array.isArray(benchmark.caseNames)
        || benchmark.caseNames.some(name => !cases.some(value => value.name === name)))) throw new Error('Unknown benchmark case name');
    const benchmarkCases = benchmark.caseNames === undefined ? cases.filter(value => !value.batched && value.k >= 2560)
      : cases.filter(value => benchmark.caseNames.includes(value.name));
    let seed = (benchmark.seed ?? 0x51a7c0de) >>> 0;
    const nextBit = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed & 1; };
    const benchmarkStarted = performance.now();
    for (const spec of benchmarkCases) {
      const data = await fixture({ spec, manifestDirectory, maxWgSize });
      const a = upload({ data: data.weights, usage: GPUBufferUsage.STORAGE });
      const x = upload({ data: data.input, usage: GPUBufferUsage.STORAGE });
      const uniform = upload({ data: data.params, usage: GPUBufferUsage.UNIFORM });
      const query = device.createQuerySet({ type: 'timestamp', count: 2 }); queries.add(query);
      const queryResolve = allocate({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      for (const candidate of variants.filter(value => value.name !== baselineName)) {
        const pairVariants = [variants.find(value => value.name === baselineName), candidate];
        const entries = pairVariants.map(variant => {
          const dst = upload({ data: data.initial, usage: GPUBufferUsage.STORAGE });
          const pipeline = pipelines.get(variant.name);
          return { ...variant, dst, pipeline, group: bind({ pipeline, sources: [a, x, dst, uniform] }),
            groups: Math.ceil(spec.m / variant.outputsPerWg) * data.batches };
        });
        const pairs = [];
        for (let rep = 0; rep < warmupPairs + measuredPairs; rep++) {
          const order = nextBit() ? [1, 0] : [0, 1];
          const pair = { order: order.map(index => entries[index].name), timings: {} };
          for (const index of order) {
            const entry = entries[index];
            const encoded = encode({ entry, groups: entry.groups, split: 0, dispatches, query, queryResolve });
            const submitted = performance.now();
            device.queue.submit([encoded.command]);
            await device.queue.onSubmittedWorkDone();
            const submitWaitMs = performance.now() - submitted;
            const words = await readWords({ buffer: queryResolve });
            const stamps = new BigUint64Array(words.buffer);
            const delta = stamps[1] - stamps[0];
            if (delta < 0n || delta > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Invalid benchmark timestamp interval');
            pair.timings[entry.name] = { gpuPassMs: Number(delta) / 1e6,
              gpuMsPerDispatch: Number(delta) / 1e6 / dispatches, cpuEncodeMs: encoded.cpuEncodeMs, submitWaitMs };
          }
          if (rep >= warmupPairs) pairs.push(pair);
        }
        const postChecks = [];
        let baselineWords;
        for (const entry of entries) {
          const words = await readWords({ buffer: entry.dst });
          const comparison = compare({ words, data, baseline: baselineWords });
          const mutations = await unchanged({ data, a, x, uniform });
          postChecks.push({ name: entry.name, ...comparison, mutations });
          if (entry.name === baselineName) baselineWords = words;
          if (failed({ comparison, mutations }) || report.errors.length) throw new Error(`Benchmark postcheck failure: ${spec.name}/${entry.name}`);
        }
        const summary = Object.fromEntries(entries.map(entry => [entry.name, Object.fromEntries(
          ['gpuPassMs', 'gpuMsPerDispatch', 'cpuEncodeMs', 'submitWaitMs'].map(metric => [metric,
            median({ values: pairs.map(pair => pair.timings[entry.name][metric]) })]))]));
        report.benchmarks.push({ case: spec.name, k: spec.k, m: spec.m, baseline: baselineName, candidate: candidate.name,
          warmupPairs, measuredPairs, dispatchesPerPass: dispatches,
          ordering: 'Seeded AB/BA order per pair; same uploaded sources, separate per-variant destinations',
          timingScope: 'GPU compute-pass timestamps include dispatches only; CPU encode and submit-to-completion wall times are separate; timestamp readback is outside both',
          summary, medianGpuRatio: summary[candidate.name].gpuMsPerDispatch > 0
            ? summary[baselineName].gpuMsPerDispatch / summary[candidate.name].gpuMsPerDispatch : null,
          zeroDurationPasses: pairs.flatMap(pair => Object.values(pair.timings)).filter(value => value.gpuPassMs === 0).length,
          pairs, postChecks });
        for (const entry of entries) { entry.dst.destroy(); buffers.delete(entry.dst); }
      }
      query.destroy(); queries.delete(query);
      await releaseBuffers();
    }
    report.benchmarkWallMs = performance.now() - benchmarkStarted;
  }
  report.ok = report.cases.length === cases.length && report.errors.length === 0;
} catch (error) {
  report.errors.push(error.stack ?? String(error));
} finally {
  for (const query of queries) query.destroy();
  for (const buffer of buffers) buffer.destroy();
  device?.destroy();
  await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ ok: report.ok, cases: report.cases.length, benchmarks: report.benchmarks.length, errors: report.errors }));
// Keep the Dawn owner strongly reachable across every await, including report output.
process.exit(report.ok ? 0 : 1);
