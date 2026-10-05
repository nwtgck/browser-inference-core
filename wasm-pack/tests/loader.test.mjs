import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {selectWasmAssets} from '../runtime/catalog.mjs';
import {createWasmLoader,decompressNative,sha256} from '../runtime/loader.mjs';
async function fixture(){
  const raw=new Uint8Array([0,97,115,109,1,0,0,0, ...Array.from({length:500},(_,i)=>i%256)]);
  const compressed=new Uint8Array(gzipSync(raw,{level:9}));
  const identity=async b=>({bytes:b.length,sha256:await sha256(b)});
  const rid=await identity(raw),aid=await identity(compressed);
  const catalog={formatVersion:1,decoderApiVersion:1,assets:{data:{path:'data/a.gz',...aid,codec:'gzip',decoded:rid}},
    targets:{one:{identity:{runtime:'llama-cpp',source:'stable',profile:'cpu-wasm32',variant:'browser'},raw:{path:'raw/one.wasm',...rid},representations:[{id:'full','kind':'full',payload:'data'}]}},
    runtime:{entry:'runtime/loader.mjs',files:[{path:'runtime/loader.mjs',bytes:1,sha256:'a'.repeat(64),role:'always',codecs:['gzip']}]}};
  return {raw,compressed,plan:selectWasmAssets(catalog,{targets:['one'],codec:'gzip'})};
}
test('native gzip path fully restores original; unsupported target does not fetch',async()=>{
  const f=await fixture();let reads=0;
  const loader=createWasmLoader({plan:f.plan,readAsset:async()=>{reads++;return f.compressed;}});
  assert.deepEqual(await loader.load('one'),f.raw);assert.equal(reads,1);
  await assert.rejects(loader.load('missing'),/Unavailable/);assert.equal(reads,1);
});
test('caller cannot mutate selected plan after creating loader',async()=>{
  const f=await fixture();const external=JSON.parse(JSON.stringify(f.plan));
  const loader=createWasmLoader({plan:external,readAsset:async()=>f.compressed});
  external.targets.one.raw.sha256='0'.repeat(64);external.assets.data.sha256='0'.repeat(64);
  assert.deepEqual(await loader.load('one'),f.raw);
});
test('corrupt compressed input rejects before decompression, next request retries',async()=>{
  const f=await fixture();let bad=true,decodes=0;
  const loader=createWasmLoader({plan:f.plan,readAsset:async()=>{const b=f.compressed.slice();if(bad)b[0]^=1;return b;},
    decompress:async(...args)=>{decodes++;return decompressNative(...args);}});
  await assert.rejects(loader.load('one'),/hash/);assert.equal(decodes,0);bad=false;
  assert.deepEqual(await loader.load('one'),f.raw);assert.equal(decodes,1);
});
test('output is verified even when a platform adapter is supplied',async()=>{
  const f=await fixture();
  for(const result of [f.raw.slice(0,-1),new Uint8Array(f.raw.length)]){
    const l=createWasmLoader({plan:f.plan,readAsset:async()=>f.compressed,decompress:async()=>result});
    await assert.rejects(l.load('one'),/length\/hash/);
  }
});
test('abort before load does not fetch and does not poison later request',async()=>{
  const f=await fixture();const ac=new AbortController();ac.abort(new Error('caller abort'));let reads=0;
  const l=createWasmLoader({plan:f.plan,readAsset:async()=>{reads++;return f.compressed;}});
  await assert.rejects(l.load('one',{signal:ac.signal}),/caller abort/);assert.equal(reads,0);
  assert.deepEqual(await l.load('one'),f.raw);
});
test('abort while fetching and queued caller cancellation stay isolated',async()=>{
  const f=await fixture();let release;const ac=new AbortController();let first=true;
  const l=createWasmLoader({plan:f.plan,readAsset:async()=>{if(first){first=false;await new Promise(r=>release=r);}return f.compressed;}});
  const a=l.load('one',{signal:ac.signal});const b=l.load('one');
  while(!release)await new Promise(r=>setImmediate(r));ac.abort(new Error('only first'));release();
  await assert.rejects(a,/only first/);assert.deepEqual(await b,f.raw);
});
test('native stream rejects short/long outputs, malformed frames and unknown codec',async()=>{
  const f=await fixture();
  await assert.rejects(decompressNative(f.compressed,'gzip',f.raw.length-1),/overflow/);
  await assert.rejects(decompressNative(f.compressed,'gzip',f.raw.length+1),/length/);
  await assert.rejects(decompressNative(f.compressed.slice(0,-2),'gzip',f.raw.length));
  await assert.rejects(decompressNative(f.compressed,'zstd',f.raw.length),/Unsupported/);
  await assert.rejects(decompressNative(f.compressed,'gzip',Number.MAX_SAFE_INTEGER),/bound/);
});
test('empty selection emits nothing and rejects every load',async()=>{
  const f=await fixture();const empty={...f.plan,targets:{},assets:{},runtime:{entry:null,files:[]},files:[],bytes:0,rawBytes:0};
  const l=createWasmLoader({plan:empty,readAsset:async()=>{throw Error('must not fetch');}});
  await assert.rejects(l.load('one'),/Unavailable/);
});

// A stalled reader must not prevent an independent queued caller from cancelling.
// Abort does not allow the decoder to run concurrently with unfinished work.
test('queued and active abort reject promptly without breaking serialization', async () => {
  const f = await fixture();
  let release, reads = 0;
  const loader = createWasmLoader({plan: f.plan, readAsset: async () => {
    reads++;
    if (reads === 1) await new Promise(resolve => { release = resolve; });
    return f.compressed;
  }});
  const active = new AbortController(), queued = new AbortController();
  const first = loader.load('one', {signal: active.signal});
  const second = loader.load('one', {signal: queued.signal});
  let firstDone = false, secondDone = false;
  const checkedFirst = assert.rejects(first, /active cancelled/).then(() => { firstDone = true; });
  const checkedSecond = assert.rejects(second, /queued cancelled/).then(() => { secondDone = true; });
  while (!release) await new Promise(resolve => setImmediate(resolve));
  queued.abort(new Error('queued cancelled'));
  active.abort(new Error('active cancelled'));
  const third = loader.load('one');
  await new Promise(resolve => setImmediate(resolve));
  const settledBeforeRelease = [firstDone, secondDone];
  assert.equal(reads, 1, 'cancel must not release the serialization gate early');
  release();
  await Promise.all([checkedFirst, checkedSecond]);
  assert.deepEqual(await third, f.raw);
  assert.equal(reads, 2, 'cancelled queued work must never fetch');
  assert.deepEqual(settledBeforeRelease, [true, true]);
});

test('unknown and already-aborted requests fail before a stalled queue', async () => {
  const f = await fixture(); let release;
  const loader = createWasmLoader({plan: f.plan, readAsset: async () => {
    await new Promise(resolve => { release = resolve; }); return f.compressed;
  }});
  const pending = loader.load('one');
  while (!release) await new Promise(resolve => setImmediate(resolve));
  const controller = new AbortController(); controller.abort(new Error('already cancelled'));
  let unknownDone = false, abortedDone = false;
  const unknown = assert.rejects(loader.load('unknown'), /Unavailable/).then(() => { unknownDone = true; });
  const aborted = assert.rejects(loader.load('one', {signal: controller.signal}), /already cancelled/).then(() => { abortedDone = true; });
  await new Promise(resolve => setImmediate(resolve));
  const settled = [unknownDone, abortedDone];
  release(); await pending; await Promise.all([unknown, aborted]);
  assert.deepEqual(settled, [true, true]);
});
