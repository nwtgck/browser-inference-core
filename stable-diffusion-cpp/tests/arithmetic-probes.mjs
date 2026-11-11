// Serialized into the browser Worker and also run against real Wasm in Node.
// Keep all dependencies inside this function or in its explicit arguments.
export async function runArithmeticProbes({ module, core, backendNames }) {
  if (JSON.stringify(backendNames) !== '["CPU"]' && JSON.stringify(backendNames) !== '["CPU","WebGPU"]') {
    throw new TypeError('Arithmetic backends must be CPU, optionally followed by WebGPU');
  }
  const timestep = [], bf16Weights = [], conv3dBias = [], webgpuPerformance = [];
  const graphWalk = module._sdc_test_graph_walk() === 1;
  if (!graphWalk) throw Error('Deep graph construction/compute propagation failed');

  // Match bf16-weights-probe.cpp: GGUF single/multipart for each target,
  // followed by single-file safetensors for each target. Both GGUF layouts
  // contain the same 1032 BF16 values and have identical unique-byte totals.
  const weightCases = [
    { format: 'gguf', multipart: false, target: 'f32' },
    { format: 'gguf', multipart: true, target: 'f32' },
    { format: 'gguf', multipart: false, target: 'f16' },
    { format: 'gguf', multipart: true, target: 'f16' },
    { format: 'safetensors', multipart: false, target: 'f32' },
    { format: 'safetensors', multipart: false, target: 'f16' },
  ];
  for (const name of backendNames) {
    const pointer = core.utf8(name);
    const placement = [], conversions = [];
    const placementLog = module.addFunction((_level, text) => {
      const message = core.readUtf8(BigInt(text));
      if (message.includes('browser-placement-v1 ')) placement.push(message);
      if (message.includes('browser-weight-conversion-v1 ')) conversions.push(message);
    }, 'vipp');
    try {
      await core.api.sd_set_log_callback(BigInt(placementLog), 0n);
      const invoke = probe => module.ccall(probe, 'number',
        [core.pointerBytes === 8 ? 'bigint' : 'number'],
        [core.pointerBytes === 8 ? pointer : Number(pointer)], { async: true });
      const code = await invoke('sdc_test_qwen_timestep');
      if (code !== 1) throw Error(`Synthetic Qwen timestep ${name} failed: ${code}`);
      if (placement.length !== 1 || !placement[0].includes('bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0') || placement[0].includes('probe')) {
        throw Error('Missing or unsafe native BF16 placement summary');
      }
      const expectedWeights = name === 'WebGPU'
        ? 'cpu_unsupported_bf16=1 webgpu_weights=1 host_weights=0 other_weights=0 webgpu_cpu_bf16=1 webgpu_cpu_bf16_use_bytes=2048'
        : 'cpu_unsupported_bf16=0 webgpu_weights=0 host_weights=1 other_weights=0 webgpu_cpu_bf16=0 webgpu_cpu_bf16_use_bytes=0';
      if (!placement[0].includes(expectedWeights)) throw Error('Original BF16 weight placement was lost across scheduler allocation');
      timestep.push({ backend: name, passed: true });
      placement.length = 0;
      const converted = await invoke('sdc_test_bf16_weights');
      if (converted !== 1) throw Error(`Synthetic BF16 weight loading ${name} failed: ${converted}`);
      if (placement.length !== weightCases.length || placement.some(line => !line.includes(name === 'WebGPU'
        ? 'bf16=0 inspected=0 cpu_bf16=0 webgpu_bf16=0 other_bf16=0'
        : 'bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0'))) {
        throw Error(`Unexpected converted-weight placement: expected ${weightCases.length}, observed ${placement.length}`);
      }
      const expectedConversions = name === 'WebGPU' ? weightCases : [];
      if (conversions.length !== expectedConversions.length || expectedConversions.some(({ target }, index) =>
        !conversions[index].includes(`target=${target} tensors=2 source_bytes=2064 destination_bytes=${target === 'f32' ? 4128 : 2064} extra_bytes=${target === 'f32' ? 2064 : 0}`))) {
        throw Error('BF16 unique-byte accounting or conversion order failed');
      }
      bf16Weights.push({ backend: name, passed: true });

      const convolution = await invoke('sdc_test_conv3d_bias');
      if (convolution !== 1) throw Error(`Synthetic 3D convolution bias ${name} failed: ${convolution}`);
      conv3dBias.push({ backend: name, passed: true });
      const performance = await invoke('sdc_test_webgpu_performance');
      if (performance !== 1) throw Error(`Bounded graph/copy ${name} failed: ${performance}`);
      webgpuPerformance.push({ backend: name, passed: true });
    } finally {
      await core.api.sd_set_log_callback(0n, 0n);
      module.removeFunction(placementLog);
      core.free(pointer);
    }
  }
  return { timestep, bf16Weights, graphWalk, conv3dBias, webgpuPerformance };
}
