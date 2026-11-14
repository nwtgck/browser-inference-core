import type { Readable } from 'node:stream';

/** Bounded binary reads avoid buffering an entire Git/Wasm payload in memory. */
export class BinaryReader {
  private readonly iterator: AsyncIterator<unknown>;
  private buffer: Buffer = Buffer.alloc(0);

  constructor({ stream }: { stream: Readable }) {
    this.iterator = stream[Symbol.asyncIterator]();
  }

  async take({ count }: { count: number }): Promise<Buffer> {
    if (!Number.isSafeInteger(count) || count <= 0) {
      throw new Error('Expected a positive binary read size');
    }
    while (!this.buffer.length) {
      const next = await this.iterator.next();
      if (next.done || !Buffer.isBuffer(next.value)) {
        throw new Error('Truncated binary stream');
      }
      this.buffer = next.value;
    }
    const result = this.buffer.subarray(0, count);
    this.buffer = this.buffer.subarray(result.length);
    return result;
  }

  async line({ limit }: { limit: number | undefined }): Promise<string> {
    const chunks: Buffer[] = [];
    for (let length = 0; length < (limit ?? 256); length++) {
      const byte = await this.take({ count: 1 });
      if (byte[0] === 10) {
        return Buffer.concat(chunks).toString('ascii');
      }
      chunks.push(byte);
    }
    throw new Error('Binary stream header exceeds its limit');
  }

  /** Drive the stream through EOF after a complete framed response. */
  async expectEnd(): Promise<void> {
    if (this.buffer.length) {
      throw new Error('Unexpected trailing binary stream bytes');
    }
    for (;;) {
      const next = await this.iterator.next();
      if (next.done) {
        return;
      }
      if (!Buffer.isBuffer(next.value)) {
        throw new Error('Expected a binary stream');
      }
      if (next.value.length) {
        throw new Error('Unexpected trailing binary stream bytes');
      }
    }
  }
}
