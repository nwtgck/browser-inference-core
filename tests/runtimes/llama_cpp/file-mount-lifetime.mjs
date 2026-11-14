/** Run against a real Emscripten FS. Does not allocate a 50 GB buffer or load model weights. */
export function checkFileMountLifetime(core, mountReadOnlyFile) {
  const size = 50_000_000_000;
  const mounted = mountReadOnlyFile(core, '/probe/large.gguf', {
    size,
    read(view, offset) {
      for (let i = 0; i < view.length; i++) view[i] = (offset + i) % 251;
      return view.length;
    },
  });
  const fs = core.module.FS;
  const live = new Set();
  const retain = stream => { live.add(stream); return stream; };
  const close = stream => { fs.close(stream); live.delete(stream); };
  const positions = [2 ** 31 - 3, 2 ** 32 - 3, 20 * 2 ** 30, size - 7];
  try {
    const original = retain(fs.open(mounted.path, 'r'));
    const duplicate = retain(fs.dupStream(original));
    const surviving = retain(fs.dupStream(duplicate));
    close(original); close(duplicate);
    let blocked = false;
    try { mounted.remove(); } catch { blocked = true; }
    if (!blocked) throw new Error('A duplicate-of-duplicate did not retain the mount');
    for (const offset of positions) {
      const bytes = new Uint8Array(11).fill(255);
      const count = fs.read(surviving, bytes, 0, bytes.length, offset);
      if (count !== Math.min(bytes.length, size - offset)) throw new Error('Incorrect large-file short read');
      for (let i = 0; i < count; i++) {
        if (bytes[i] !== (offset + i) % 251) throw new Error('A large-file offset was truncated');
      }
      if (bytes.subarray(count).some(value => value !== 255)) throw new Error('Read wrote beyond EOF');
    }
    if (fs.read(surviving, new Uint8Array(1), 0, 1, size) !== 0) throw new Error('Read past EOF');
    close(surviving);
  } finally {
    for (const stream of live) fs.close(stream);
    mounted.remove();
  }
  return { duplicateDepth: 2, logicalFileBytes: size, largeOffsetReads: positions.length, passed: true };
}
