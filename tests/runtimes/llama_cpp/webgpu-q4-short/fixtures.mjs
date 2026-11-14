// Shared deterministic fixtures for standalone GPU input and native reference.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
export function fixture(spec) {
  const { kind, k, m, mode, sharedInput, broadcast } = spec, indexed = kind === 'indexed';
  const blocks = k / 256, experts = 11, slots = indexed ? 8 : 1;
  const a2 = indexed ? experts : broadcast ? 2 : 1, a3 = broadcast ? 2 : 1;
  const x2 = broadcast ? 4 : 1, x3 = broadcast ? 6 : 1;
  const offA = mode ? 1 : 0, offX = mode ? 3 : 0, offI = mode ? 2 : 0, offD = mode ? 7 : 0;
  const stride01 = blocks + (mode ? 1 : 0), stride02 = m * stride01 + (mode ? 3 : 0), stride03 = a2 * stride02 + (mode ? 1 : 0);
  const bNe1 = indexed && !sharedInput ? slots : 1;
  const stride11 = k + (mode ? 5 : 0), stride12 = bNe1 * stride11 + (mode ? 3 : 0), stride13 = x2 * stride12 + (mode ? 5 : 0);
  const weights = new Uint8Array((offA + a3 * stride03 + 1) * 144); weights.fill(0x42);
  const view = new DataView(weights.buffer), scales = [0x2800, 0xac00, 0, 0x3000, 0x2400];
  for (let e3 = 0; e3 < a3; e3++) for (let e2 = 0; e2 < a2; e2++) for (let row = 0; row < m; row++) for (let block = 0; block < blocks; block++) {
    const base = (offA + e3 * stride03 + e2 * stride02 + row * stride01 + block) * 144;
    for (let j = 4; j < 144; j++) weights[base + j] = (e3 * 41 + e2 * 71 + row * 37 + block * 19 + j * 13) % 256;
    view.setUint16(base, scales[(e3 + e2 + row + block) % scales.length], true);
    view.setUint16(base + 2, scales[(e3 + 2 * e2 + row + block + 1) % scales.length], true);
  }
  const input = new Float32Array(offX + x3 * stride13 + 16); input.fill(99.5);
  for (let i3 = 0; i3 < x3; i3++) for (let i2 = 0; i2 < x2; i2++) for (let slot = 0; slot < bNe1; slot++) for (let i = 0; i < k; i++) {
    input[offX + i3 * stride13 + i2 * stride12 + slot * stride11 + i] = mode === 2 ? Math.fround(Math.sin(i * 0.43 + slot + i2 * 0.17 + i3 * 0.29) * 2 ** ((i % 17) - 8)) : (((i * 53 + slot * 17 + i2 * 7 + i3 * 11) % 1009) - 504) / 1024;
  }
  const ids = new Uint32Array(offI + slots + 4); ids.fill(9);
  for (let slot = 0; slot < slots; slot++) ids[offI + slot] = (slot * 7 + 3) % experts;
  ids[offI] = experts - 1; ids[offI + slots - 1] = ids[offI];
  const outputWords = m * slots * x2 * x3, initial = new Uint32Array(offD + outputWords + 16); initial.fill(0x7eabcdef);
  const params = new Uint32Array(indexed ? [offA, offX, offI, offD, k, m, experts, slots, bNe1, stride01, stride11, stride02, stride12] : [offA, offX, offD, m, 1, k, stride01, stride11, stride02, stride12, stride03, stride13, a2, a3, x2 / a2, x3 / a3]);
  return { weights, input, ids, initial, params, offD, outputWords, groups: Math.ceil(m / 4) * slots * x2 * x3 };
}

export function encodeFixture(spec, data) {
  const header = new Uint32Array([spec.kind === 'indexed' ? 1 : 0, data.weights.byteLength, data.input.length, data.ids.length, data.params.length]);
  return Buffer.concat([header, data.weights, data.input, data.ids, data.params].map(array => Buffer.from(array.buffer, array.byteOffset, array.byteLength)));
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [shaders, output] = process.argv.slice(2);
  if (!shaders || !output || process.argv.length !== 4) throw new Error('Usage: fixtures.mjs SHADERS OUTPUT');
  const cases = JSON.parse(await readFile(join(shaders, 'cases.json'), 'utf8'));
  await mkdir(output, { recursive: true });
  for (let i = 0; i < cases.length; i++) await writeFile(join(output, `${i}.bin`), encodeFixture(cases[i], fixture(cases[i])));
}
