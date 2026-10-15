import test from 'node:test';
import assert from 'node:assert/strict';
import { mountReadOnlyFile } from '../examples/runtime/read-only-file.mjs';

function fixture(source, options) {
  const parent = { contents: {} }; let node; let last;
  class ErrnoError extends Error { constructor(code) { super(code); this.errno = code; } }
  const FS = {
    ErrnoError, createNode(_parent, name, mode) { node = { id: 1, name, mode }; return node; },
    mkdirTree() {}, analyzePath: () => ({ exists: false }), lookupPath: () => ({ node: parent }),
    destroyNode() {}, unlink(name) { last = name; delete parent.contents[node.name]; },
  };
  const core = { module: { FS }, constant: name => name };
  const mounted = mountReadOnlyFile(core, '/models/a.gguf', source, options);
  return { node, mounted, get removed() { return last; } };
}
test('small chunks satisfy a larger native read', () => {
  const calls = [];
  const { node } = fixture({ size: 20, read(view, offset) { calls.push([offset, view.length]); view.fill(7); return view.length; } }, { maxChunkBytes: 3 });
  const bytes = new Uint8Array(12);
  assert.equal(node.stream_ops.read({}, bytes, 2, 8, 4), 8);
  assert.deepEqual(calls, [[4,3],[7,3],[10,2]]);
  assert.deepEqual([...bytes], [0,0,7,7,7,7,7,7,7,7,0,0]);
});
test('logical positions beyond 2 GiB and 4 GiB are not truncated', () => {
  const seen = [];
  const { node } = fixture({ size: 5 * 2 ** 30 + 17, read(view, offset) { seen.push(offset); view.fill(offset % 251); return view.length; } });
  for (const offset of [2 ** 31 + 9, 2 ** 32 + 9, 5 * 2 ** 30]) {
    assert.equal(node.stream_ops.llseek({ position: 0 }, offset, 0), offset);
    const bytes = new Uint8Array(4);
    assert.equal(node.stream_ops.read({}, bytes, 0, 4, offset), 4);
    assert.equal(bytes[0], offset % 251);
  }
  assert.deepEqual(seen, [2 ** 31 + 9, 2 ** 32 + 9, 5 * 2 ** 30]);
});
test('short reads are accumulated, and EOF stays short', () => {
  const { node } = fixture({ size: 5, read(view, offset) { view[0] = offset; return 1; } });
  const bytes = new Uint8Array(10);
  assert.equal(node.stream_ops.read({}, bytes, 0, 10, 0), 5);
  assert.deepEqual([...bytes.subarray(0,5)], [0,1,2,3,4]);
  assert.equal(node.stream_ops.read({}, bytes, 0, 10, 5), 0);
});
test('Promise, negative/oversized count and thrown read failures become I/O errors', () => {
  for (const read of [() => Promise.resolve(1), () => -1, () => 99, () => { throw new Error('read failed'); }]) {
    const { node } = fixture({ size: 4, read });
    assert.throws(() => node.stream_ops.read({}, new Uint8Array(4), 0, 4, 0), e => e.errno === 'EIO');
  }
});
test('read-only access, seek validation and mapping refusal are explicit', () => {
  const { node } = fixture({ size: 4, read: () => 0 });
  assert.throws(() => node.stream_ops.open({ flags: 1 }), /EROFS/);
  assert.throws(() => node.stream_ops.llseek({}, -1, 0), /EINVAL/);
  assert.throws(() => node.stream_ops.llseek({}, 0, 9), /EINVAL/);
  assert.throws(() => node.stream_ops.mmap(), /EINVAL/);
});
test('a source cannot be removed while a native handle is open', () => {
  const f = fixture({ size: 4, read: () => 0 });
  f.node.stream_ops.open({ flags: 0 });
  assert.throws(() => f.mounted.remove(), /Close all/);
  f.node.stream_ops.close(); f.mounted.remove();
  assert.equal(f.removed, '/models/a.gguf');
  f.mounted.remove();
});

test('dupStream retains its own native open count even when the loader closes first', () => {
  const f = fixture({ size: 4, read: view => { view.fill(8); return view.length; } });
  f.node.stream_ops.open({ flags: 0 });
  f.node.stream_ops.dup();
  f.node.stream_ops.close();
  assert.throws(() => f.mounted.remove(), /Close all/);
  const bytes = new Uint8Array(4);
  assert.equal(f.node.stream_ops.read({}, bytes, 0, 4, 0), 4);
  assert.deepEqual([...bytes], [8,8,8,8]);
  f.node.stream_ops.close(); f.mounted.remove();
  assert.throws(() => f.node.stream_ops.dup(), /ENOENT/);
});
