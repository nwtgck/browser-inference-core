import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { createSyntheticModel } from '#tools/validation/synthetic-model';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

test('synthetic GGUF is deterministic and contains a complete small untrained llama with valid tensor storage', async () => {
  await using temporary = await temporaryDirectory();
  const output = path.join(temporary.root, 'one.gguf');
  await createSyntheticModel({ output });
  const bytes = await readFile(output);
  await createSyntheticModel({ output: path.join(temporary.root, 'two.gguf') });
  assert.deepEqual(bytes, await readFile(path.join(temporary.root, 'two.gguf')));
  assert.equal(bytes.subarray(0, 4).toString(), 'GGUF');
  let position = 4;
  const uint32 = () => {
    const value = bytes.readUInt32LE(position);
    position += 4;
    return value;
  };
  const uint64 = () => {
    const value = Number(bytes.readBigUInt64LE(position));
    position += 8;
    return value;
  };
  const string = () => {
    const size = uint64();
    const value = bytes.subarray(position, position + size).toString();
    position += size;
    return value;
  };
  const value = ({ kind }: { kind: number }): unknown => {
    switch (kind) {
    case 4: case 5: return uint32();
    case 6: {
      const result = bytes.readFloatLE(position);
      position += 4;
      return result;
    }
    case 7: return bytes[position++];
    case 8: return string();
    case 9: {
      const element = uint32();
      const length = uint64();
      return Array.from({ length }, () => value({ kind: element }));
    }
    default: throw new Error(`Unsupported fixture metadata kind ${kind}`);
    }
  };
  assert.equal(uint32(), 3);
  const tensorCount = uint64();
  const metadataCount = uint64();
  assert.equal(tensorCount, 12);
  const metadata = new Map<string, unknown>();
  for (let index = 0; index < metadataCount; index++) {
    metadata.set(string(), value({ kind: uint32() }));
  }
  assert.equal(metadata.get('general.architecture'), 'llama');
  assert.equal(metadata.get('llama.block_count'), 1);
  assert.equal(metadata.get('llama.embedding_length'), 32);
  assert.equal(metadata.get('llama.attention.head_count'), 4);
  const tokens = metadata.get('tokenizer.ggml.tokens');
  assert.ok(Array.isArray(tokens));
  assert.equal(tokens.length, 259);
  assert.deepEqual(tokens.slice(0, 4), ['<unk>', '<s>', '</s>', '<0x00>']);
  assert.equal(tokens[258], '<0xFF>');
  const tensors: { name: string; dimensions: number[]; offset: number }[] = [];
  for (let index = 0; index < tensorCount; index++) {
    const name = string();
    const dimensions = Array.from({ length: uint32() }, () => uint64());
    assert.equal(uint32(), 0, 'Every tensor uses ordinary F32');
    const offset = uint64();
    assert.equal(offset % 32, 0);
    tensors.push({ name, dimensions, offset });
  }
  const dataStart = Math.ceil(position / 32) * 32;
  let end = dataStart;
  for (const tensor of tensors) {
    const count = tensor.dimensions.reduce((product, dimension) => product * dimension, 1);
    const start = dataStart + tensor.offset;
    assert.ok(start >= end);
    end = start + count * 4;
    assert.ok(end <= bytes.length);
    for (let offset = start; offset < end; offset += 4) {
      const weight = bytes.readFloatLE(offset);
      if (tensor.name.includes('norm.weight')) {
        assert.equal(weight, 1);
      } else {
        assert.ok(Number.isFinite(weight) && weight >= -0.1 && weight <= 0.1);
      }
    }
  }
  assert.equal(end, bytes.length);
  assert.deepEqual(tensors.find((tensor) => tensor.name === 'token_embd.weight')?.dimensions, [32, 259]);
  assert.deepEqual(tensors.find((tensor) => tensor.name === 'blk.0.ffn_down.weight')?.dimensions, [64, 32]);
});
