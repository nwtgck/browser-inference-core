import test from 'node:test';
import assert from 'node:assert/strict';
import { mountReadOnlyFile } from '../../../runtimes/speech-cpp/host/read-only-file.mjs';

function mountFixture({ source, maxChunkBytes }) {
  const parent = { contents: {} };
  let node;
  let removed = false;
  class ErrnoError extends Error {
    constructor(code) {
      super(code);
      this.errno = code;
    }
  }
  const core = {
    constant: name => name,
    module: {
      FS: {
        ErrnoError,
        createNode(_parent, name, mode) {
          node = { id: 1, name, mode };
          return node;
        },
        mkdirTree() {},
        analyzePath: () => ({ exists: false }),
        lookupPath: () => ({ node: parent }),
        destroyNode() {},
        unlink() {
          removed = true;
          delete parent.contents[node.name];
        },
      },
    },
  };
  const mounted = mountReadOnlyFile(core, '/models/speech.gguf', source, { maxChunkBytes });
  return {
    node,
    mounted,
    get removed() {
      return removed;
    },
  };
}

test('large model reads preserve exact file offsets and bound each source access', () => {
  const start = 2 ** 32 + 7;
  const calls = [];
  const { node, mounted } = mountFixture({
    source: {
      size: start + 9,
      read(view, offset) {
        calls.push([offset, view.length]);
        // Force successful short reads, as allowed by the random-access source.
        view[0] = offset - start;
        return 1;
      },
    },
    maxChunkBytes: 3,
  });
  assert.equal(node.stream_ops.llseek({ position: 0 }, start, 0), start);
  assert.equal(node.node_ops.getattr().size, start + 9);
  const output = new Uint8Array(7);
  assert.equal(node.stream_ops.read({}, output, 1, 5, start), 5);
  assert.deepEqual([...output], [0, 0, 1, 2, 3, 4, 0]);
  assert.deepEqual(calls, [[start, 3], [start + 1, 3], [start + 2, 3], [start + 3, 2], [start + 4, 1]]);
  assert.equal(node.stream_ops.read({}, output, 0, 1, start + 9), 0);
  mounted.remove();
});

test('native file ownership prevents unmount until the last read handle closes', () => {
  const fixture = mountFixture({ source: { size: 4, read: () => 0 }, maxChunkBytes: 3 });
  fixture.node.stream_ops.open({ flags: 0 });
  assert.throws(() => fixture.mounted.remove(), /Close all native file handles/);
  assert.throws(() => fixture.node.stream_ops.write(), /EROFS/);
  assert.throws(() => fixture.node.stream_ops.mmap(), /EINVAL/);
  fixture.node.stream_ops.close();
  fixture.mounted.remove();
  assert.equal(fixture.removed, true);
  assert.throws(() => fixture.node.stream_ops.open({ flags: 0 }), /ENOENT/);
});

test('asynchronous or invalid source reads fail instead of accepting corrupt bytes', () => {
  for (const read of [() => Promise.resolve(1), () => -1, () => 8, () => {
    throw new Error('read failed');
  }]) {
    const { node, mounted } = mountFixture({ source: { size: 4, read }, maxChunkBytes: 3 });
    assert.throws(() => node.stream_ops.read({}, new Uint8Array(4), 0, 4, 0), error => error.errno === 'EIO');
    mounted.remove();
  }
});
