import test from 'node:test';
import assert from 'node:assert/strict';
import {selectWasmAssets,validateCatalog,validatePlan} from '../runtime/catalog.mjs';
const hash='a'.repeat(64),file=(path,bytes,role='always')=>({path,bytes,sha256:hash,role,codecs:['gzip','brotli']});
function fixture(){
 const c={formatVersion:1,decoderApiVersion:1,assets:{},targets:{},runtime:{entry:'runtime/loader.mjs',files:[file('runtime/loader.mjs',5)]}};
 for(const id of ['cpu','gpu32','gpu64']) {
  const raw={path:`raw/${id}.wasm`,bytes:100,sha256:hash};
  const reps=[];
  for(const codec of ['gzip','brotli']) {
   const key=`${id}-${codec}`;
   c.assets[key]={path:`data/${key}`,bytes:80,sha256:hash,codec,decoded:{bytes:100,sha256:hash}};
   reps.push({id:`full-${codec}`,kind:'full',payload:key});
   if(id==='gpu64') {
    const k=`delta-${codec}`;c.assets[k]={path:`data/${k}`,bytes:10,sha256:hash,codec,decoded:{bytes:40,sha256:hash}};
    reps.push({id:`delta-${codec}`,kind:'delta',base:'cpu',payload:k});
   }
  }
  c.targets[id]={identity:{runtime:'llama-cpp',source:'stable',profile:id,variant:'browser'},raw,representations:reps};
 }
 return c;
}
test('same set shares a selected base, independent full remains available',()=>{
 const c=fixture(); const p=selectWasmAssets(c,{targets:['gpu64','cpu'],codec:'gzip'});
 assert.equal(p.bytes,95); assert.equal(p.targets.gpu64.representation.kind,'delta');validatePlan(p);
 assert.deepEqual(Object.keys(p.targets),['cpu','gpu64']);
});
test('standalone selection NEVER includes the excluded CPU base',()=>{
 const p=selectWasmAssets(fixture(),{targets:['gpu32','gpu64'],codec:'brotli'});
 assert.equal(p.bytes,165);assert.equal(p.targets.gpu64.representation.kind,'full');
 assert(p.files.every(f=>!f.path.includes('cpu')&&!f.path.includes('gzip')));validatePlan(p);
});
test('removing one profile chooses an independent representation',()=>{
 const p=selectWasmAssets(fixture(),{targets:['gpu64'],codec:'gzip'});
 assert.equal(p.bytes,85);assert.equal(p.targets.gpu64.representation.kind,'full');validatePlan(p);
});
test('force hot target full even when delta is smaller',()=>{
 const p=selectWasmAssets(fixture(),{targets:['gpu64','cpu'],codec:'gzip',fullTargets:['gpu64']});
 assert.equal(p.targets.gpu64.representation.kind,'full');validatePlan(p);
});
test('unavailable profile/codec, duplicate or malformed requests fail',()=>{
 for(const options of [{targets:['missing'],codec:'gzip'},{targets:['cpu','cpu'],codec:'gzip'},{targets:['cpu'],codec:'zstd'},{targets:['cpu'],codec:'gzip',fullTargets:['gpu64']}]) assert.throws(()=>selectWasmAssets(fixture(),options));
});
test('reject dangling bases, unknown format, duplicate assets, missing full',()=>{
 for(const mutate of [c=>c.formatVersion=2,c=>c.targets.gpu64.representations[1].base='missing',c=>c.assets['cpu-gzip'].path='../x',c=>c.assets['cpu-gzip'].path=c.assets['cpu-brotli'].path,c=>c.targets.gpu64.representations=c.targets.gpu64.representations.filter(r=>r.kind!=='full')]){
  const c=fixture();mutate(c);assert.throws(()=>validateCatalog(c));
 }
});
test('plan cannot smuggle excluded base, unused assets or changed costs',()=>{
 const original=selectWasmAssets(fixture(),{targets:['cpu','gpu64'],codec:'gzip'});
 for(const mutate of [p=>delete p.targets.cpu,p=>p.bytes++,p=>p.assets.hidden=fixture().assets['gpu32-brotli'],p=>p.targets.cpu.representation={kind:'delta',id:'x',base:'gpu64',payload:'delta-gzip'}]){
  const p=structuredClone(original);mutate(p);assert.throws(()=>validatePlan(p));
 }
});
test('planning is deterministic, detached and immutable',()=>{
 const c=fixture();const p=selectWasmAssets(c,{targets:['gpu64','cpu'],codec:'gzip'});c.assets['cpu-gzip'].bytes=999;
 assert.equal(p.bytes,95); assert(Object.isFrozen(p.targets.cpu.raw));
 assert.deepEqual(p,selectWasmAssets(fixture(),{targets:['cpu','gpu64'],codec:'gzip'}));
});
test('disabling the runtime emits no assets or decoder',()=>{
 const p=selectWasmAssets(fixture(),{targets:[],codec:'brotli'});
 assert.deepEqual(p.files,[]);assert.equal(p.bytes,0);assert.equal(p.runtime.entry,null);validatePlan(p);
});

test('codec entry metadata cannot refer to an absent or incompatible file', () => {
  const c = fixture();
  c.runtime.entries = {brotli: 'runtime/missing.mjs'};
  assert.throws(() => validateCatalog(c), /entry/);
  c.runtime.entries = {unknown: c.runtime.entry};
  assert.throws(() => validateCatalog(c), /codec/);
});

test('selected codec entry belongs to the emitted file closure', () => {
  const c = fixture();
  const original = c.runtime.files.find(f => f.path === c.runtime.entry);
  c.runtime.files.push({...original, path: 'runtime/brotli-entry.mjs', codecs: ['brotli']});
  c.runtime.entries = {brotli: 'runtime/brotli-entry.mjs'};
  const plan = selectWasmAssets(c, {targets: ['cpu'], codec: 'brotli'});
  assert.equal(plan.runtime.entry, 'runtime/brotli-entry.mjs');
  assert(plan.files.some(f => f.path === plan.runtime.entry));
  validatePlan(plan);
});
