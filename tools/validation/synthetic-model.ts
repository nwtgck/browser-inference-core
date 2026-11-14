import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { directory } from '#tools/shared/files';

function ggufString({ value }: { value: string }): Buffer {
  const text = Buffer.from(value);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(text.length));
  return Buffer.concat([length, text]);
}

function uint32({ value }: { value: number }): Buffer {
  const data = Buffer.alloc(4);
  data.writeUInt32LE(value);
  return data;
}

function uint64({ value }: { value: number }): Buffer {
  const data = Buffer.alloc(8);
  data.writeBigUInt64LE(BigInt(value));
  return data;
}

function float32({ value }: { value: number }): Buffer {
  const data = Buffer.alloc(4);
  data.writeFloatLE(value);
  return data;
}

/** Small deterministic, untrained GGUF. No downloaded or proprietary weights. */
export async function createSyntheticModel({ output }: { output: string }): Promise<void> {
  const vocab = ['<unk>', '<s>', '</s>'];
  for (let value = 0; value < 256; value++) {
    vocab.push(`<0x${value.toString(16).padStart(2, '0').toUpperCase()}>`);
  }
  const metadata: { name: string; kind: number; data: Buffer }[] = [];
  for (const [name, value] of [['general.architecture', 'llama'], ['general.name', 'synthetic-boundary-test-untrained'], ['tokenizer.ggml.model', 'llama']]) {
    if (name === undefined || value === undefined) {
      throw new Error('Incomplete synthetic metadata');
    }
    metadata.push({ name, kind: 8, data: ggufString({ value }) });
  }
  for (const [name, value] of Object.entries({ 'llama.context_length': 128, 'llama.embedding_length': 32, 'llama.block_count': 1, 'llama.feed_forward_length': 64, 'llama.attention.head_count': 4, 'llama.attention.head_count_kv': 4, 'llama.rope.dimension_count': 8, 'tokenizer.ggml.bos_token_id': 1, 'tokenizer.ggml.eos_token_id': 2, 'tokenizer.ggml.unknown_token_id': 0 })) {
    metadata.push({ name, kind: 4, data: uint32({ value }) });
  }
  metadata.push({ name: 'llama.attention.layer_norm_rms_epsilon', kind: 6, data: float32({ value: 1e-5 }) });
  metadata.push({ name: 'tokenizer.ggml.add_bos_token', kind: 7, data: Buffer.from([1]) });
  metadata.push({ name: 'tokenizer.ggml.tokens', kind: 9, data: Buffer.concat([uint32({ value: 8 }), uint64({ value: vocab.length }), ...vocab.map((value) => ggufString({ value }))]) });
  metadata.push({ name: 'tokenizer.ggml.scores', kind: 9, data: Buffer.concat([uint32({ value: 6 }), uint64({ value: vocab.length }), Buffer.alloc(vocab.length * 4)]) });
  metadata.push({ name: 'tokenizer.ggml.token_type', kind: 9, data: Buffer.concat([uint32({ value: 5 }), uint64({ value: vocab.length }), ...vocab.map((_, index) => uint32({ value: index === 0 ? 2 : index < 3 ? 3 : 6 }))]) });
  const tensors: { name: string; dimensions: number[] }[] = [
    { name: 'token_embd.weight', dimensions: [32, vocab.length] }, { name: 'output_norm.weight', dimensions: [32] }, { name: 'output.weight', dimensions: [32, vocab.length] },
    { name: 'blk.0.attn_norm.weight', dimensions: [32] }, { name: 'blk.0.ffn_norm.weight', dimensions: [32] },
  ];
  for (const part of ['q', 'k', 'v', 'output']) {
    tensors.push({ name: `blk.0.attn_${part}.weight`, dimensions: [32, 32] });
  }
  for (const part of ['gate', 'up']) {
    tensors.push({ name: `blk.0.ffn_${part}.weight`, dimensions: [32, 64] });
  }
  tensors.push({ name: 'blk.0.ffn_down.weight', dimensions: [64, 32] });
  const descriptors: Buffer[] = [];
  const weights: Buffer[] = [];
  let offset = 0;
  let seed = 1847;
  for (const tensor of tensors) {
    const padding = (32 - offset % 32) % 32;
    weights.push(Buffer.alloc(padding));
    offset += padding;
    const count = tensor.dimensions.reduce((product, value) => product * value, 1);
    const data = Buffer.alloc(count * 4);
    for (let index = 0; index < count; index++) {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      data.writeFloatLE(tensor.name.includes('norm.weight') ? 1 : ((seed >>> 0) / 2 ** 32 * 0.2 - 0.1), index * 4);
    }
    descriptors.push(Buffer.concat([ggufString({ value: tensor.name }), uint32({ value: tensor.dimensions.length }), ...tensor.dimensions.map((value) => uint64({ value })), uint32({ value: 0 }), uint64({ value: offset })]));
    weights.push(data);
    offset += data.length;
  }
  const header = Buffer.concat([Buffer.from('GGUF'), uint32({ value: 3 }), uint64({ value: tensors.length }), uint64({ value: metadata.length }), ...metadata.map((entry) => Buffer.concat([ggufString({ value: entry.name }), uint32({ value: entry.kind }), entry.data])), ...descriptors]);
  await directory({ path: path.dirname(output), create: true });
  await writeFile(output, Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), ...weights]));
}
