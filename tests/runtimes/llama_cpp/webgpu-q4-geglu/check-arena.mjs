import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [shaderDirectory, modulePath] = process.argv.slice(2);
assert.ok(shaderDirectory && modulePath, 'Provide a native provider shader directory and a node-webgpu module path');

const dawn = await import(pathToFileURL(resolve(modulePath)).href);
const { GPUBufferUsage, GPUMapMode } = dawn.globals;
const gpu = dawn.create(['backend=vulkan']);
const adapter = await gpu.requestAdapter();
assert.ok(adapter);
const subgroup = adapter.info.subgroupMinSize;
assert.ok(subgroup === adapter.info.subgroupMaxSize && [16, 32].includes(subgroup), 'Use a fixed subgroup 16 or 32 adapter');
const device = await adapter.requestDevice({ requiredFeatures: ['subgroups', 'shader-f16'] });
const queue = device.queue;
const { UNIFORM, STORAGE, COPY_SRC, COPY_DST, MAP_READ } = GPUBufferUsage;
const k = 1536;
const m = 6144;
const blocksPerRow = k / 32;
const rowBytes = blocksPerRow * 18;

function weights(rowQuant) {
  const bytes = Buffer.alloc(rowBytes * m);
  for (let row = 0; row < m; ++row) {
    const q = rowQuant(row);
    for (let offset = row * rowBytes; offset < (row + 1) * rowBytes; offset += 18) {
      bytes.writeUInt16LE(0x3c00, offset); // f16 scale 1.
      bytes.fill((q << 4) | q, offset + 2, offset + 18);
    }
  }
  return bytes;
}

const gateQuant = row => 5 + row % 7;

const upQuant = row => 9 + row % 5;

const inputs = [
  weights(gateQuant),
  Buffer.from(new Float32Array(k).fill(1 / 2048).buffer),
  weights(upQuant),
  Buffer.alloc(4 * m, 0x5a),
];

const source = await readFile(resolve(shaderDirectory, `q4-geglu-sg${subgroup}.wgsl`), 'utf8');
const shader = device.createShaderModule({ code: source });
const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader } });

const align = value => Math.ceil(value / device.limits.minStorageBufferOffsetAlignment) *
    device.limits.minStorageBufferOffsetAlignment;

async function dispatch(layout) {
  const resources = [];
  const entries = [];
  const arenas = [];
  try {
    for (const bindings of layout) {
      let end = align(4);
      const ranges = bindings.map(binding => {
        const offset = align(end);
        const size = inputs[binding].byteLength;
        end = offset + size;
        return { binding, offset, size };
      });
      const size = align(end) + align(4);
      const expected = Buffer.alloc(size, 0xa7);
      const storage = device.createBuffer({ size, usage: STORAGE | COPY_SRC | COPY_DST });
      const readback = device.createBuffer({ size, usage: MAP_READ | COPY_DST });
      resources.push(storage, readback);
      for (const { binding, offset, size } of ranges) {
        inputs[binding].copy(expected, offset);
        entries.push({ binding, resource: { buffer: storage, offset, size } });
      }
      arenas.push({ storage, readback, expected, output: ranges.find(range => range.binding === 3) });
      queue.writeBuffer(storage, 0, expected);
    }
    const uniform = device.createBuffer({ size: 32, usage: UNIFORM | COPY_DST });
    resources.push(uniform);
    queue.writeBuffer(uniform, 0, new Uint32Array([0, 0, 0, 0, k, m, blocksPerRow, 0]));
    entries.push({ binding: 4, resource: { buffer: uniform } });

    device.pushErrorScope('validation');
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(m / (128 / subgroup * 4));
    pass.end();
    for (const { storage, readback } of arenas) {
      encoder.copyBufferToBuffer(storage, 0, readback, 0, storage.size);
    }
    queue.submit([encoder.finish()]);
    const validation = await device.popErrorScope();
    assert.equal(validation?.message, undefined);

    let output;
    for (const arena of arenas) {
      await arena.readback.mapAsync(GPUMapMode.READ);
      const contents = Buffer.from(arena.readback.getMappedRange());
      if (arena.output) {
        const { offset, size } = arena.output;
        output = Buffer.from(contents.subarray(offset, offset + size));
        output.copy(arena.expected, offset);
      }
      assert.ok(contents.equals(arena.expected), 'Weights, activation and padding must stay unchanged');
      arena.readback.unmap();
    }
    assert.ok(output);
    return output;
  } finally {
    for (const resource of resources) {
      resource.destroy();
    }
  }
}

try {
  const separate = await dispatch([[0], [1], [2], [3]]);
  for (let row = 0; row < m; ++row) {
    const gate = (gateQuant(row) - 8) * k / 2048;
    const up = (upQuant(row) - 8) * k / 2048;
    const expected = 0.5 * gate * (1 + Math.tanh(0.7978845608028654 * gate * (1 + 0.044715 * gate * gate))) * up;
    const actual = separate.readFloatLE(row * 4);
    assert.ok(Math.abs(actual - expected) < 3e-6, `row ${row}: ${actual} vs ${expected}`);
  }
  for (const layout of [[[0], [1, 3], [2]], [[0, 1, 2, 3]]]) {
    const shared = await dispatch(layout);
    assert.ok(shared.equals(separate), 'Shared and separate storage must produce identical rows');
  }
  console.log(`Q4_0 GEGLU shared arena, subgroup ${subgroup}: ${m} rows match in all three layouts`);
} finally {
  device.destroy();
}
