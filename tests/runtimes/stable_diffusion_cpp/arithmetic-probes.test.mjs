import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runArithmeticProbes } from './arithmetic-probes.mjs';

function fixture({ placementCount = 6, conversionTargets = ['f32', 'f32', 'f16', 'f16', 'f32', 'f16'],
  conversionBytes = undefined, failures = {}, pointerBytes = 4 } = {}) {
  const calls = [], freed = [], removed = [], strings = new Map(), callbacks = new Map();
  let next = 1, activeLog = 0;
  const utf8 = value => {
    const pointer = BigInt(next++);
    strings.set(pointer, value);
    return pointer;
  };
  const placement = backend => 'browser-placement-v1 nodes=3 ' +
    (backend === 'WebGPU' ? 'bf16=0 inspected=0 cpu_bf16=0 webgpu_bf16=0 other_bf16=0'
      : 'bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0');
  const emit = message => callbacks.get(activeLog)(0, Number(utf8(message)), 0);
  const module = {
    _sdc_test_graph_walk() { calls.push('graphWalk'); return failures.graphWalk ?? 1; },
    addFunction(callback, signature) {
      assert.equal(signature, 'vipp');
      const index = next++;
      callbacks.set(index, callback);
      return index;
    },
    removeFunction(index) { assert.ok(callbacks.delete(index)); removed.push(index); },
    async ccall(name, resultType, argumentTypes, values, options) {
      assert.equal(resultType, 'number');
      assert.deepEqual(argumentTypes, [pointerBytes === 8 ? 'bigint' : 'number']);
      assert.equal(typeof values[0], pointerBytes === 8 ? 'bigint' : 'number');
      assert.deepEqual(options, { async: true });
      const backend = strings.get(BigInt(values[0]));
      calls.push(`${backend}/${name}`);
      if (name === 'sdc_test_qwen_timestep') {
        emit(placement('CPU') + (backend === 'WebGPU'
          ? ' cpu_unsupported_bf16=1 webgpu_weights=1 host_weights=0 other_weights=0 webgpu_cpu_bf16=1 webgpu_cpu_bf16_use_bytes=2048'
          : ' cpu_unsupported_bf16=0 webgpu_weights=0 host_weights=1 other_weights=0 webgpu_cpu_bf16=0 webgpu_cpu_bf16_use_bytes=0'));
      }
      if (name === 'sdc_test_bf16_weights') {
        for (let index = 0; index < placementCount; index++) emit(placement(backend));
        if (backend === 'WebGPU') {
          for (const target of conversionTargets) {
            emit(`browser-weight-conversion-v1 target=${target} tensors=2 source_bytes=${conversionBytes ?? 2064} destination_bytes=${target === 'f32' ? 4128 : 2064} extra_bytes=${target === 'f32' ? 2064 : 0}`);
          }
        }
      }
      return failures[name] ?? 1;
    },
  };
  const core = {
    pointerBytes, utf8,
    readUtf8(pointer) { return strings.get(pointer); },
    free(pointer) { freed.push(pointer); },
    api: { async sd_set_log_callback(index) { activeLog = Number(index); } },
  };
  return { module, core, calls, freed, removed, activeLog: () => activeLog };
}

test('the Worker-serialized runner reaches every CPU probe with six BF16 cases', async () => {
  const serialized = Function(`return (${runArithmeticProbes.toString()});`)();
  for (const pointerBytes of [4, 8]) {
    const replay = fixture({ pointerBytes });
    const result = await serialized({ ...replay, backendNames: ['CPU'] });
    assert.deepEqual(replay.calls, ['graphWalk', 'CPU/sdc_test_qwen_timestep', 'CPU/sdc_test_bf16_weights',
      'CPU/sdc_test_conv3d_bias', 'CPU/sdc_test_webgpu_performance']);
    assert.equal(result.graphWalk, true);
    for (const probe of ['timestep', 'bf16Weights', 'conv3dBias', 'webgpuPerformance']) {
      assert.deepEqual(result[probe], [{ backend: 'CPU', passed: true }]);
    }
    assert.equal(replay.freed.length, 1);
    assert.equal(replay.removed.length, 1);
    assert.equal(replay.activeLog(), 0);
  }
});

test('the six-case WebGPU conversion order and unique-byte totals remain exact', async () => {
  const replay = fixture();
  const result = await runArithmeticProbes({ ...replay, backendNames: ['CPU', 'WebGPU'] });
  for (const probe of ['timestep', 'bf16Weights', 'conv3dBias', 'webgpuPerformance']) {
    assert.deepEqual(result[probe], [{ backend: 'CPU', passed: true }, { backend: 'WebGPU', passed: true }]);
  }
  assert.equal(replay.calls.at(-1), 'WebGPU/sdc_test_webgpu_performance');
});

test('four old placement summaries cannot pass the expanded BF16 probe', async () => {
  const replay = fixture({ placementCount: 4 });
  await assert.rejects(runArithmeticProbes({ ...replay, backendNames: ['CPU'] }), /expected 6, observed 4/);
  assert.equal(replay.calls.at(-1), 'CPU/sdc_test_bf16_weights');
  assert.equal(replay.freed.length, 1);
  assert.equal(replay.removed.length, 1);
  assert.equal(replay.activeLog(), 0);
});

test('reordered conversion targets and double-counted multipart bytes are rejected', async () => {
  for (const options of [
    { conversionTargets: ['f32', 'f16', 'f32', 'f16', 'f32', 'f16'] },
    { conversionBytes: 4128 },
  ]) {
    const replay = fixture(options);
    await assert.rejects(runArithmeticProbes({ ...replay, backendNames: ['CPU', 'WebGPU'] }),
      /unique-byte accounting or conversion order/);
    assert.equal(replay.calls.at(-1), 'WebGPU/sdc_test_bf16_weights');
  }
});

test('downstream convolution and performance failures remain mandatory failures', async () => {
  for (const [probe, message] of [
    ['sdc_test_conv3d_bias', /3D convolution bias CPU failed: -7/],
    ['sdc_test_webgpu_performance', /Bounded graph\/copy CPU failed: -7/],
  ]) {
    const replay = fixture({ failures: { [probe]: -7 } });
    await assert.rejects(runArithmeticProbes({ ...replay, backendNames: ['CPU'] }), message);
    assert.equal(replay.calls.at(-1), `CPU/${probe}`);
    assert.equal(replay.freed.length, 1);
    assert.equal(replay.removed.length, 1);
    assert.equal(replay.activeLog(), 0);
  }
});

test('a missing CPU baseline cannot produce a passing arithmetic result', async () => {
  for (const backendNames of [[], ['WebGPU'], ['CPU', 'CPU'], ['CPU', 'other']]) {
    const replay = fixture();
    await assert.rejects(runArithmeticProbes({ ...replay, backendNames }), TypeError);
    assert.deepEqual(replay.calls, []);
  }
});
