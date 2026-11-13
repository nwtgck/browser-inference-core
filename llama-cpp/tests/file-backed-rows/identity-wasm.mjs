// Actual Wasm libc stat/fstat/dup against the production read-only adapter.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { mountReadOnlyFile } from '../../examples/runtime/read-only-file.mjs';

const { default: create } = await import(pathToFileURL(process.argv[2]).href);
const lines = [];
const module = await create({ print: text => lines.push(text) });
const core = { module, constant: name => module.ccall('lcb_identity_errno', 'number', ['string'], [name]) };
const reads = [];
const source = name => ({ size: 256, read(view, offset) {
  reads.push({ name, offset, bytes: view.length });
  for (let i = 0; i < view.length; i++) view[i] = offset + i;
  return view.length;
} });
const selected = mountReadOnlyFile(core, '/models/selected', source('selected'));
const other = mountReadOnlyFile(core, '/other/selected', source('other'));
try {
  const status = module.ccall('lcb_identity_check', 'number', ['string', 'string', 'string'],
    [selected.path, other.path, '/models/missing-original-ple']);
  assert.ok(status >= 20);
  const observation = JSON.parse(lines.find(line => line.startsWith('{')));
  assert.equal(observation.device, 1);
  assert.equal(observation.inode, module.FS.lookupPath(selected.path).node.id);
  assert.equal(observation.otherInode, module.FS.lookupPath(other.path).node.id);
  assert.notEqual(observation.inode, observation.otherInode);
  assert.equal(observation.size, 256);
  assert.deepEqual(reads, [{ name: 'selected', offset: 7, bytes: 1 }]);
  assert.throws(() => selected.remove(), /Close all native/);
  // The C++ loader handles have closed, but the retained dup still reads.
  assert.equal(module.ccall('lcb_identity_again', 'number', [], []), 1);
  assert.deepEqual(reads, [
    { name: 'selected', offset: 7, bytes: 1 },
    { name: 'selected', offset: 7, bytes: 1 },
  ]);
  assert.equal(module.ccall('lcb_identity_release', 'number', [], []), 0);
  selected.remove(); other.remove();
  assert.equal(module.FS.analyzePath(selected.path).exists, false);
  assert.equal(module.FS.analyzePath(other.path).exists, false);
  const openFiles = module.FS.streams.filter(stream => stream?.path === selected.path || stream?.path === other.path);
  assert.equal(openFiles.length, 0);
  console.log(JSON.stringify({ status: 'passed', actualWasmLibc: true, actualReadOnlyAdapter: true,
    ...observation, bytesRead: 2, retainedAfterLoaderClose: true, noOpenFiles: true }));
} finally {
  module.ccall('lcb_identity_release', 'number', [], []);
  selected.remove(); other.remove();
}
