// Optional real-GPU regression. Requires a separately installed native Dawn module.
// Standalone numerical smoke test; software-GPU results are not target-device speed.
import { readFile, writeFile, appendFile } from 'node:fs/promises';
const { create, globals } = await import(process.env.LCB_TEST_WEBGPU_MODULE ?? 'webgpu');
import { join } from 'node:path';
Object.assign(globalThis, globals);
const report={version:2,scope:'standalone HC4 kernels against staged GPU expressions; not compiled bicore/model inference',cases:[],errors:[]};
const gpu=create(['backend=vulkan']);
// dawn.node owns its native instance through create()'s returned object. A local
// binding used only for requestAdapter() is not an async lifetime guarantee.
// Keep an explicit root until cleanup and the standalone process exit.
const gpuLifetime = Symbol('HC Dawn instance lifetime');
globalThis[gpuLifetime] = gpu;
const forceGc = process.env.LCB_HC_FORCE_GC === '1';
report.forcedGc = forceGc;
report.node = process.version;
const caseStart=Number(process.argv[4]??0), caseCount=Number(process.argv[5]??288);
const reportPath=process.argv[3]??'hc-gpu-results.json';
const trace=(...x)=>{if(process.env.LCB_HC_TRACE)console.log(...x);};
await writeFile(reportPath+'.jsonl','');
let device;
const made=[];
try {
  if (!Number.isSafeInteger(caseStart) || !Number.isSafeInteger(caseCount) ||
      caseStart < 0 || caseCount <= 0 || caseStart + caseCount > 288) {
    throw new Error('case range must be integers inside [0, 288), with a positive count');
  }
  if (forceGc && typeof globalThis.gc !== 'function') throw new Error('LCB_HC_FORCE_GC requires node --expose-gc');
  const adapter=await gpu.requestAdapter(); if(!adapter)throw new Error('No Vulkan adapter');
  report.adapter={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description};
  device=await adapter.requestDevice();
  device.addEventListener('uncapturederror',e=>report.errors.push(String(e.error)));
  const root=process.argv[2]??new URL('.',import.meta.url).pathname;
  const codes=await Promise.all(['hc_pre.wgsl','hc_post.wgsl'].map(n=>readFile(join(root,n),'utf8')));
  const pipelines=[];
  for(let i=0;i<codes.length;i++){
    const module=device.createShaderModule({code:codes[i],label:i?'candidate HC POST':'candidate HC PRE'});
    const info=await module.getCompilationInfo();
    if(info.messages.some(m=>m.type==='error'))throw new Error(JSON.stringify(info.messages));
    pipelines.push(await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}));
  }
  const sentinel=0x4a123456;
  function storage(words){const b=device.createBuffer({size:words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});made.push(b);return b;}
  function uniform(words){const bytes=new Uint32Array(Math.ceil(words.length/4)*4);bytes.set(words);const b=device.createBuffer({size:bytes.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,bytes);made.push(b);return b;}
  async function read(buffer){const b=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});made.push(b);const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,b,0,buffer.size);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const copy=new Uint32Array(b.getMappedRange().slice(0));b.unmap();return copy;}
  function dispatch(pipeline,resources,count,split=false){const e=device.createCommandEncoder();const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:resources.map((resource,binding)=>({binding,resource}))}));const groups=Math.ceil(count/64);const limit=split?Math.max(2,Math.ceil(Math.sqrt(groups))):65535;
    const y=Math.max(1,Math.ceil(groups/limit));const x=Math.ceil(groups/y);p.dispatchWorkgroups(x,y);p.end();device.queue.submit([e.finish()]);}
  const cache=new Map();
  async function ref(kind,body,inputs,output,count,d,t){
    let p=cache.get(kind);
    if(!p){const declarations=inputs.map((_,i)=>`@group(0) @binding(${i}) var<storage,read_write> a${i}:array<f32>;`).join('\n');
      const code=`${declarations}\n@group(0) @binding(${inputs.length}) var<storage,read_write> o:array<f32>;\nstruct P{n:u32,d:u32,t:u32,z:u32}; @group(0) @binding(${inputs.length+1}) var<uniform> p:P; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=p.n){return;} ${body}}`;
      p=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code}),entryPoint:'main'}});cache.set(kind,p);}
    dispatch(p,[...inputs.map(buffer=>({buffer})),{buffer:output},{buffer:uniform([count,d,t,0])}],count);
  }
  function values(n,seed){const a=new Float32Array(n);let state=seed;for(let i=0;i<n;i++){state=(Math.imul(state,1664525)+1013904223)>>>0;a[i]=((state>>>8)/16777216-0.5)*8;}if(n>6){a[0]=0;a[1]=-0;a[2]=1e-30;a[3]=-1e-30;a[4]=seed===23?80:1e15;a[5]=seed===23?-80:-1e15;}return a;}
  function same(a,b){const f=new Float32Array(new Uint32Array([a,b]).buffer);return a===b||(Number.isNaN(f[0])&&Number.isNaN(f[1]));}
  let caseIndex=-1;
  for(const d of [1,3,63,64,65,3072])for(const t of [1,3])for(const pooled of [false,true])for(const residue of [0,1,63])for(const split of [false,true])for(const pre of [true,false]){
    ++caseIndex;if(caseIndex<caseStart||caseIndex>=caseStart+caseCount)continue;
    if (forceGc) { globalThis.gc(); await new Promise(resolve => setImmediate(resolve)); }
    console.log(JSON.stringify({start:caseIndex,d,t,pooled,residue,split,pre}));
    const n=d*t, inputs=pre?[values(4*n,17),values(4*n,23)]:[values(n,41),values(4*n,53),values(4*t,71)];
    if(!pre){for(let i=0;i<4*n;i+=17){const token=Math.floor(i/(4*d));const channel=Math.floor(i/d)%4;inputs[1][i]=-Math.fround(inputs[0][token*d+i%d]*inputs[2][token*4+channel]);}}
    const outN=pre?n:4*n, all=[...inputs,new Float32Array(outN)];
    const offsets=all.map((_,k)=>(residue+k*7)%64);
    const scale=residue===0?0.25:residue===1?-1.125:0.0;
    const spans=all.map((a,k)=>Math.ceil((a.length+offsets[k]+8)/64)*64);let total=spans.reduce((a,b)=>a+b,0);
    const pool=pooled?storage(total):undefined;const bindings=[];const buffers=[];const initial=[];let start=0;
    for(let k=0;k<all.length;k++){const b=pool??storage(spans[k]);const u=new Uint32Array(spans[k]).fill(sentinel);if(k<inputs.length)u.set(new Uint32Array(all[k].buffer),offsets[k]);device.queue.writeBuffer(b,pooled?start*4:0,u);bindings.push({buffer:b,offset:pooled?start*4:0,size:spans[k]*4});buffers.push(b);initial.push(u);start+=spans[k];}
    const f=new Float32Array([scale,0]);const bits=new Uint32Array(f.buffer);
    const words=pre?[...offsets,d,t,bits[0],bits[1]]:[...offsets,d,t];
    dispatch(pipelines[pre?0:1],[...bindings,{buffer:uniform(words)}],outN,split);trace('candidate submitted');
    const dstIndex=all.length-1;const targetData=await read(buffers[dstIndex]);trace('candidate read');const outOffset=pooled?spans.slice(0,-1).reduce((a,b)=>a+b,0):0;
    const actual=targetData.slice(outOffset+offsets[dstIndex],outOffset+offsets[dstIndex]+outN);
    const refs=inputs.map(a=>{const b=storage(a.length);device.queue.writeBuffer(b,0,a);return b;});
    const expected=storage(outN);
    if(pre){const sig=storage(4*n),prod=storage(4*n),acc=storage(n);await ref('sig','o[i]=1.0/(1.0+exp(-a0[i]));',[refs[1]],sig,4*n,d,t);
      await ref('prod','o[i]=a0[i]*a1[i];',[refs[0],sig],prod,4*n,d,t);
      await ref('first','let base=(i/p.d)*(4u*p.d)+i%p.d;o[i]=a0[base];',[prod],acc,n,d,t);
      for(let c=1;c<4;c++)await ref(`sum${c}`,`let base=(i/p.d)*(4u*p.d)+i%p.d;o[i]=o[i]+a0[base+${c}u*p.d];`,[prod],acc,n,d,t);
      // The uniform is the existing runtime scale/bias boundary, not a literal +0 simplification.
      await ref('scale','o[i]=a0[i]*a1[0]+a1[1];',[acc,(()=>{const b=storage(2);device.queue.writeBuffer(b,0,new Float32Array([scale,0]));return b;})()],expected,n,d,t);
    }else{const repeat=storage(4*n),mul=storage(4*n);await ref('repeat','o[i]=a0[(i/(4u*p.d))*p.d+i%p.d];',[refs[0]],repeat,4*n,d,t);
      await ref('postmul','o[i]=a0[i]*a1[i/p.d];',[repeat,refs[2]],mul,4*n,d,t);await ref('resadd','o[i]=a0[i]+a1[i];',[refs[1],mul],expected,4*n,d,t);}
    trace('reference submitted');const wanted=await read(expected);trace('reference read');let mismatches=0;for(let i=0;i<outN;i++)mismatches+=Number(!same(actual[i],wanted[i]));
    let guardErrors=0;
    for(let k=0,off=0;k<all.length;off+=spans[k++]){const data=pooled?targetData:await read(buffers[k]);const origin=pooled?off:0;for(let i=0;i<spans[k];i++){if(k===dstIndex&&i>=offsets[k]&&i<offsets[k]+outN)continue;guardErrors+=Number(data[origin+i]!==initial[k][i]);}}
    report.cases.push({index:caseIndex,d,t,pooled,residue,offsets,split,scale,operation:pre?'PRE':'POST',outputWords:outN,mismatches,guardErrors});
    await appendFile(reportPath+'.jsonl',JSON.stringify(report.cases.at(-1))+'\n');
    // Keep every submitted resource alive until this case's queue work is done.
    // A mapped readback is not used as a blanket lifetime fence for other buffers.
    await device.queue.onSubmittedWorkDone();
    for(const b of made.splice(0))b.destroy();
    if(report.cases.length%32===0)console.log(JSON.stringify({completed:report.cases.length}));
  }
  console.log(JSON.stringify({phase:"negative-alias"}));
  const alias=storage(256), aliasDst=storage(64);
  device.pushErrorScope('validation');
  dispatch(pipelines[0],[{buffer:alias,offset:0,size:256},{buffer:alias,offset:0,size:256},{buffer:aliasDst},{buffer:uniform([0,0,0,4,1,0x3e800000,0])}],4);
  const aliasError=await device.popErrorScope();
  report.rejectedAlias={rejected:aliasError!==null,message:aliasError?.message??null};
  if(!aliasError) report.errors.push('Expected writable-alias validation rejection was not observed');
  report.ok=report.errors.length===0&&report.cases.every(x=>x.mismatches===0&&x.guardErrors===0);
}catch(e){report.ok=false;report.errors.push(e.stack??String(e));}
finally{for(const b of made)b.destroy();device?.destroy();await writeFile(process.argv[3]??'hc-gpu-results.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({ok:report.ok,cases:report.cases.length,errors:report.errors}));}
// Native Dawn keeps its event loop alive: this is a standalone, fully awaited test process.
delete globalThis[gpuLifetime];
process.exit(report.ok?0:1);
