/** A virtual, structurally valid large GGUF. Payload holes are never allocated.
 * Every padding tensor is 1 GiB; only the final scalar is actually read. */
export function makeFixture(gib, version = 3) {
  const metadata = [];
  const u32 = n => { for (let i = 0; i < 4; ++i) metadata.push((n >>> (8 * i)) & 255); };
  const u64 = n => { let value = BigInt(n); for (let i = 0; i < 8; ++i) { metadata.push(Number(value & 255n)); value >>= 8n; } };
  const text = value => { const bytes = new TextEncoder().encode(value); u64(bytes.length); metadata.push(...bytes); };
  u32(0x46554747); u32(version); u64(gib + 1); u64(2);
  text('tokenizer.ggml.tokens'); u32(9); u32(8); u64(2); text('alpha'); text('beta');
  text('test.array'); u32(9); u32(4); u64(2); u32(7); u32(9);
  for (let i = 0; i <= gib; ++i) {
    text(i === gib ? 'weight' : 'padding' + i); u32(1);
    u64(i === gib ? 1 : 268435456); u32(0); u64(BigInt(i) << 30n);
  }
  const header = Uint8Array.from(metadata);
  const offset = Math.ceil(header.length / 32) * 32 + gib * 1073741824;
  const marker = new Uint8Array([0, 0, 128, 63]);
  const requests = [];
  const maxChunkBytes = 4096;
  // This fixture has <1 KiB of metadata and one scalar. Budget actual bytes,
  // not calls: stdio can split/repeat buffered reads differently across builds.
  const maxTotalReadBytes = 256 * 1024;
  const readAheadBytes = 64 * 1024;
  let phase = 'filesystem', totalReadBytes = 0, largestRead = 0, violation;
  const summary = () => ({ gib, version, phase, fileBytes: offset + 4, tensorOffset: offset,
    metadataBytes: header.length, calls: requests.length, totalReadBytes, largestRead,
    maxChunkBytes, maxTotalReadBytes, violation, first: requests.slice(0, 4), last: requests.slice(-4) });
  const fail = reason => {
    violation = reason;
    throw Error('GGUF range-read guard: ' + JSON.stringify(summary()));
  };
  return {
    offset, requests, maxChunkBytes, summary,
    setPhase(value) { phase = value; },
    source: {
      size: offset + 4,
      read(destination, position) {
        const length = destination.length;
        if (!Number.isSafeInteger(position) || position < 0 || length < 1 || position > offset + 4 - length)
          fail('request outside the virtual file');
        if (length > maxChunkBytes) fail('uncapped source read');
        if (totalReadBytes + length > maxTotalReadBytes) fail('total byte budget exceeded');
        // Permit stdio read-ahead around metadata and the selected tensor, never
        // a sequential traversal of the multi-GiB padding tensors.
        if (!(position + length <= header.length + readAheadBytes || position >= offset - readAheadBytes))
          fail('read entered a padding tensor');
        requests.push({ position, length, phase });
        totalReadBytes += length;
        largestRead = Math.max(largestRead, length);
        const count = Math.min(destination.length, offset + 4 - position);
        destination.fill(0, 0, count);
        for (const [bytes, start] of [[header, 0], [marker, offset]]) {
          const from = Math.max(position, start), to = Math.min(position + count, start + bytes.length);
          if (from < to) destination.set(bytes.subarray(from - start, to - start), from - position);
        }
        return count;
      },
    },
  };
}
