import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import test from 'node:test';
import { BinaryReader } from '#tools/shared/streams';

test('binary reads preserve byte boundaries and reject truncated/overlong headers', async () => {
  const reader = new BinaryReader({ stream: Readable.from([Buffer.from('abc\n'), Buffer.from([0, 255, 10])]) });
  assert.equal(await reader.line({ limit: undefined }), 'abc');
  assert.deepEqual(await reader.take({ count: 2 }), Buffer.from([0, 255]));
  assert.deepEqual(await reader.take({ count: 10 }), Buffer.from([10]));
  await assert.rejects(reader.take({ count: 1 }), /Truncated/u);
  await assert.rejects(new BinaryReader({ stream: Readable.from([Buffer.from('abcd')]) }).line({ limit: 3 }), /limit/u);
});

test('expectEnd drives the final binary chunk through stream EOF, including empty streams', async () => {
  for (const chunks of [[], [Buffer.from('final')], [Buffer.alloc(0), Buffer.from('final'), Buffer.alloc(0)]]) {
    await using stream = Readable.from(chunks);
    const reader = new BinaryReader({ stream });
    if (chunks.some((chunk) => chunk.length)) {
      assert.deepEqual(await reader.take({ count: 5 }), Buffer.from('final'));
    }
    await reader.expectEnd();
    assert.equal(stream.readableEnded, true);
    await reader.expectEnd();
    await assert.rejects(reader.take({ count: 1 }), /Truncated/u);
  }
});

test('expectEnd rejects buffered or subsequent trailing bytes', async () => {
  for (const chunks of [[Buffer.from('okextra')], [Buffer.from('ok'), Buffer.from('extra')]]) {
    await using stream = Readable.from(chunks);
    const reader = new BinaryReader({ stream });
    assert.deepEqual(await reader.take({ count: 2 }), Buffer.from('ok'));
    await assert.rejects(reader.expectEnd(), /trailing/u);
  }
});

test('expectEnd preserves stream failures and rejects nonbinary output', async () => {
  const failure = new Error('stream read failed');
  const broken = Readable.from((async function* () {
    yield Buffer.from('ok');
    throw failure;
  })());
  // Async disposal would rethrow the stream error already asserted below.
  try {
    const brokenReader = new BinaryReader({ stream: broken });
    assert.deepEqual(await brokenReader.take({ count: 2 }), Buffer.from('ok'));
    await assert.rejects(brokenReader.expectEnd(), (error) => error === failure);
  } finally {
    broken.destroy();
  }
  await using nonbinary = Readable.from(['text']);
  await assert.rejects(new BinaryReader({ stream: nonbinary }).expectEnd(), /binary stream/u);
});
