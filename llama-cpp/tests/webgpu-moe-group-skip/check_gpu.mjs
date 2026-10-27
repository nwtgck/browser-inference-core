// Opt-in actual upstream gather and canonical/prepared matrix validation.
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {performance} from 'node:perf_hooks';
if (!process.env.LCB_TEST_WEBGPU_MODULE) throw new Error('Set LCB_TEST_WEBGPU_MODULE to a local webgpu package URL');
const {create, globals}=await import(process.env.LCB_TEST_WEBGPU_MODULE);
Object.assign(globalThis, globals);
const lab=resolve(process.argv[2]), argv=process.argv.slice(3);
const value=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const output=resolve(value('--output',join(lab,'report.json'))), filter=value('--filter',''), limit=Number(value('--limit',Infinity));
const benchmark=argv.includes('--benchmark'), large=argv.includes('--large');
const hash=x=>createHash('sha256').update(x).digest('hex');
const invariant=(ok,message)=>{if(!ok)throw new Error(message);};
const source=JSON.parse(readFileSync(join(lab,large?'source-large.json':'source.json'))), manifestBytes=readFileSync(join(lab,large?'manifest-large.json':'manifest.json'));
invariant(hash(manifestBytes)===source.fixtureManifestSha256,'Manifest hash mismatch');
const specs=JSON.parse(manifestBytes).filter(x=>(large||!x.large)&&x.name.includes(filter)).slice(Number(value('--skip',0)),Number(value('--skip',0))+limit);
invariant(specs.length>0,'No fixtures selected');
const names=['baseline','candidate'];
const report={scope:'Actual upstream gather/factory/emitter; synthetic packed inputs; canonical/prepared byte validation; no model speed claim',status:'started',source,runnerSha256:hash(readFileSync(fileURLToPath(import.meta.url))),adapter:{},device:{},cases:[],benchmarks:[],errors:[]};
const owner=create(process.env.LCB_TEST_WEBGPU_BACKEND?['backend='+process.env.LCB_TEST_WEBGPU_BACKEND]:[]);globalThis[Symbol.for('ggml-moe-group-check.owner')]=owner;
let device;const live=new Set(),queries=new Set();
try{
 const adapter=await owner.requestAdapter();invariant(adapter,'No WebGPU adapter');
 report.adapter=Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize'].map(k=>[k,adapter.info[k]]));
 report.adapter.features=[...adapter.features];report.adapter.limits={maxComputeInvocationsPerWorkgroup:adapter.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupSizeX:adapter.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize};
 invariant(adapter.features.has('shader-f16'),'shader-f16 unavailable');
 const features=['shader-f16'];if(benchmark&&adapter.features.has('timestamp-query'))features.push('timestamp-query');
 device=await adapter.requestDevice({requiredFeatures:features,requiredLimits:{maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256}});
 report.device.features=[...device.features];report.device.limits={maxComputeInvocationsPerWorkgroup:device.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupSizeX:device.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupStorageSize:device.limits.maxComputeWorkgroupStorageSize};
 invariant(device.limits.maxComputeInvocationsPerWorkgroup===256,'Factory gather context/device limit mismatch');
 device.addEventListener('uncapturederror',e=>report.errors.push(e.error.message));
 device.lost.then(info=>{if(info.reason!=='destroyed')report.errors.push('device lost: '+info.message);});
 function allocate(bytes,usage,label){const buffer=device.createBuffer({size:Math.ceil(bytes/4)*4,usage,label});live.add(buffer);return buffer;}
 function upload(bytes,usage=GPUBufferUsage.STORAGE,label=''){const b=allocate(bytes.byteLength,usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label);device.queue.writeBuffer(b,0,bytes);return b;}
 async function read(b){const st=allocate(b.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,'readback');const en=device.createCommandEncoder();en.copyBufferToBuffer(b,0,st,0,b.size);device.queue.submit([en.finish()]);await st.mapAsync(GPUMapMode.READ);const data=Buffer.from(st.getMappedRange().slice(0));st.unmap();st.destroy();live.delete(st);return data;}
 function group(p,bs){return device.createBindGroup({layout:p.getBindGroupLayout(0),entries:bs.map((buffer,binding)=>({binding,resource:{buffer}}))});}
 const pipelineCache=new Map();
 async function pipeline(spec,name){
  const bytes=readFileSync(join(lab,'shaders',spec.name+'-'+name+'.wgsl'));invariant(hash(bytes)===spec.shaderHashes[name],'Shader hash mismatch');
  const key=hash(bytes);if(pipelineCache.has(key))return pipelineCache.get(key);
  device.pushErrorScope('validation');
  const module=device.createShaderModule({code:bytes.toString(),label:name});const info=await module.getCompilationInfo();
  const errors=info.messages.filter(m=>m.type==='error');invariant(!errors.length,'WGSL '+spec.name+'/'+name+': '+errors.map(x=>x.message).join('\n'));
  const p=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'},label:name});
  const error=await device.popErrorScope();invariant(!error,'Pipeline '+name+': '+error?.message);pipelineCache.set(key,p);return p;
 }
 function encode(p,g,grid,count=1,query,resolved){const en=device.createCommandEncoder();const pass=en.beginComputePass(query?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});pass.setPipeline(p);pass.setBindGroup(0,g);for(let i=0;i<count;i++)pass.dispatchWorkgroups(...grid);pass.end();if(query)en.resolveQuerySet(query,0,2,resolved,0);return en.finish();}
 function u32(bytes){return new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);}
 function f32(bytes){return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);}
 function makeIds(spec,base,round){const bytes=Buffer.from(base);if(round){const a=new Int32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4),p=spec.args;for(let t=0;t<spec.tokens;t++)for(let s=0;s<spec.used;s++){const i=p[10]+t*p[15]+s;if(a[i]!==-1)a[i]=(a[i]+3)%spec.experts;}}return bytes;}
 function gatherCheck(spec,idsBytes,posBytes,tileBytes,initialPos,initialTiles){
  const ids=new Int32Array(idsBytes.buffer,idsBytes.byteOffset,idsBytes.byteLength/4),pos=u32(posBytes),tiles=u32(tileBytes),p=spec.args,counts=Array(spec.experts).fill(0),slots=spec.used*spec.tokens;
  const at=i=>ids[p[10]+Math.floor(i/spec.used)*p[15]+i%spec.used];for(let i=0;i<slots;i++){const expert=at(i);invariant(expert===-1||(expert>=0&&expert<spec.experts),'Invalid fixture expert');if(expert!==-1)counts[expert]++;}
  let expectedTiles=0,base=0;const seen=new Set(),histogram={};
  for(let e=0;e<spec.experts;e++){
   for(let k=0;k<counts[e];k+=32){const n=Math.min(32,counts[e]-k),idx=1+3*expectedTiles;invariant(tiles[idx]===e&&tiles[idx+1]===base+k&&tiles[idx+2]===n,'Gather tile metadata mismatch');histogram[n]=(histogram[n]??0)+1;expectedTiles++;}
   for(let j=base;j<base+counts[e];j++){invariant(pos[j]<slots&&!seen.has(pos[j])&&at(pos[j])===e,'Gather slot membership/uniqueness mismatch');seen.add(pos[j]);}
   base+=counts[e];
  }
  invariant(tiles[0]===expectedTiles&&seen.size===base,'Gather count mismatch');
  for(let i=base;i<pos.length;i++)invariant(pos[i]===initialPos[i],'Gather position guard');
  for(let i=1+3*expectedTiles;i<tiles.length;i++)invariant(tiles[i]===initialTiles[i],'Gather tile guard');
  const physicalWg=spec.grid[0]*spec.grid[1],rowsWg=Math.ceil(spec.m/32);
  return {actualTiles:expectedTiles,maxTiles:spec.maxTiles,tileOccupancyHistogram:histogram,validWorkgroups:expectedTiles*rowsWg,physicalWorkgroups:physicalWg,invalidWorkgroups:physicalWg-expectedTiles*rowsWg,liveColumns:base,skippedColumns:slots-base,paddedColumns:expectedTiles*32,slotPositionsChecked:base};
 }
 function cpuCheck(spec,weights,inputBytes,idsBytes,actualBytes){
  const dq=f32(weights),x=f32(inputBytes),ids=new Int32Array(idsBytes.buffer,idsBytes.byteOffset,idsBytes.byteLength/4),out=f32(actualBytes),p=spec.args;
  let nonfinite=0,envelopeFailures=0,maxAbsError=0,maxNormalizedError=0,outputs=0;
  const unit=2**-11,nops=2*spec.k+4,gamma=nops*unit<1?nops*unit/(1-nops*unit):undefined;
  for(let t=0;t<spec.tokens;t++)for(let s=0;s<spec.used;s++)for(let m=0;m<spec.m;m++){
   const expert=ids[p[10]+t*p[15]+s],a=(expert*spec.m+m)*spec.k,b=p[8]+t*p[14]+s%p[6]*p[13];let dot=0,sumAbs=0;
   if(expert!==-1)for(let k=0;k<spec.k;k++){const product=dq[a+k]*x[b+k];dot+=product;sumAbs+=Math.abs(product);}
   const got=out[p[9]+(t*spec.used+s)*spec.m+m],err=Math.abs(got-dot);nonfinite+=Number(!Number.isFinite(got));maxAbsError=Math.max(maxAbsError,err);maxNormalizedError=Math.max(maxNormalizedError,err/(sumAbs+1e-30));if(gamma!==undefined)envelopeFailures+=Number(err>gamma*sumAbs+2**-20);outputs++;
   if(expert===-1)invariant(Object.is(got,0),'Skipped output must be exact positive zero');
  }
  return {outputs,nonfinite,envelopeFailures,maxAbsError,maxNormalizedError,f16RoundingEnvelopeGamma:gamma??null,oracle:'Unmodified upstream dequantizer + Float64 dot/sumabs; F16 conservative rounding envelope where n*u<1; baseline byte equality is separate'};
 }
 for(const spec of specs){
  const caseBuffers=new Set(live);const data={};
  for(const name of ['weights','input','ids','dequantized']){const bytes=readFileSync(join(lab,'fixtures',spec.name,name+'.bin'));invariant(hash(bytes)===spec.hashes[name],'Fixture hash mismatch '+name);data[name]=bytes;}
  invariant(spec.grid.every(n=>n<=device.limits.maxComputeWorkgroupsPerDimension),'Actual device grid limit exceeded');
  const p=spec.args,words=spec.m*spec.used*spec.tokens,initial=new Uint32Array(p[9]+words+16);initial.fill(0x7eabcdef);
  const iniPos=new Uint32Array(spec.used*spec.tokens+16);iniPos.fill(0xabcde071);const iniTiles=new Uint32Array(1+3*spec.maxTiles+16);iniTiles.fill(0xabcde073);
  const gparam=new Uint32Array(8);gparam.set(spec.gatherParams);const mparam=new Uint32Array(16);mparam.set(spec.mainParams);
  const wa=upload(data.weights,undefined,'weights'),xx=upload(data.input,undefined,'input'),ids=upload(data.ids,undefined,'ids'),pos=upload(iniPos,undefined,'slot-pos'),tiles=upload(iniTiles,undefined,'tiles'),gp=upload(gparam,GPUBufferUsage.UNIFORM,'gather-params'),mp=upload(mparam,GPUBufferUsage.UNIFORM,'main-params');
  const gatherOutput=upload(initial,undefined,'gather-output');
  const gather=await pipeline(spec,'gather'),gg=group(gather,[ids,pos,tiles,gatherOutput,gp]);const lanes=[];
  for(const name of names){const p=await pipeline(spec,name),dst=upload(initial,undefined,'output-'+name);lanes.push({name,p,dst,g:group(p,[wa,xx,dst,pos,tiles,mp])});}
  const row={name:spec.name,type:spec.type,pattern:spec.pattern,k:spec.k,m:spec.m,experts:spec.experts,used:spec.used,tokens:spec.tokens,sharedInput:spec.sharedInput,mainParams:spec.mainParams,gatherParams:spec.gatherParams,grid:spec.grid,tileK:spec.tileK,vectorized:spec.vectorized,hashes:spec.hashes,rounds:[]};report.cases.push(row);let lastOutputHash,lastInputs;
  for(let round=0;round<2;round++){
   const idsBytes=makeIds(spec,data.ids,round);device.queue.writeBuffer(ids,0,idsBytes);device.queue.writeBuffer(pos,0,iniPos);device.queue.writeBuffer(tiles,0,iniTiles);device.queue.writeBuffer(gatherOutput,0,initial);
   device.queue.submit([encode(gather,gg,[1,1])]);let posBytes=await read(pos),tilesBytes=await read(tiles);
   const gatheredOutput=await read(gatherOutput),gatheredWords=u32(gatheredOutput),liveIds=new Int32Array(idsBytes.buffer,idsBytes.byteOffset,idsBytes.byteLength/4);
   for(let i=0;i<gatheredWords.length;i++){const slot=Math.floor((i-p[9])/spec.m),inside=i>=p[9]&&i<p[9]+words;const skipped=inside&&liveIds[p[10]+Math.floor(slot/spec.used)*p[15]+slot%spec.used]===-1;invariant(gatheredWords[i]===(skipped?0:initial[i]),'Gather output initialization/guard mismatch');}
   const occupancy=gatherCheck(spec,idsBytes,posBytes,tilesBytes,iniPos,iniTiles);
   const zeroOverride=spec.zeroTilesOverride&&round===0;
   if(zeroOverride){device.queue.writeBuffer(tiles,0,new Uint32Array([0]));tilesBytes=await read(tiles);}
   const rr={round,idsSha256:hash(idsBytes),gather:occupancy,zeroTilesOverride:zeroOverride,variants:[]};row.rounds.push(rr);let baseline,baselineCpu;
   for(const lane of lanes){
    // Each matrix variant starts with the exact output state produced by the
    // canonical gather, including its skipped-slot zero initialization.
    device.queue.writeBuffer(lane.dst,0,gatheredOutput);device.queue.submit([encode(lane.p,lane.g,spec.grid)]);const out=await read(lane.dst),actual=u32(out);
    let guardErrors=0,bitDifferences=0;
    for(let i=0;i<actual.length;i++){if(i<p[9]||i>=p[9]+words||zeroOverride)guardErrors+=Number(actual[i]!==gatheredWords[i]);if(baseline)bitDifferences+=Number(actual[i]!==baseline[i]);}
    if(!baseline)baseline=Uint32Array.from(actual);
    const cpu=zeroOverride?undefined:lane.name==='baseline'?(baselineCpu=cpuCheck(spec,data.dequantized,data.input,idsBytes,out)):{...baselineCpu,inheritedFromBitIdenticalBaseline:true};
    const result={name:lane.name,shaderSha256:spec.shaderHashes[lane.name],outputSha256:hash(out),checkedOutputWords:zeroOverride?0:words,guardErrors,bitDifferences,cpu};rr.variants.push(result);
    invariant(!guardErrors&&!bitDifferences&&!cpu?.nonfinite&&!cpu?.envelopeFailures,'Numerical/guard/byte failure '+spec.name+'/'+lane.name);
   }
   for(const [label,b,expected] of [['weights',wa,data.weights],['input',xx,data.input],['ids',ids,idsBytes],['positions',pos,posBytes],['tiles',tiles,tilesBytes],['gatherParams',gp,Buffer.from(gparam.buffer)],['mainParams',mp,Buffer.from(mparam.buffer)]])invariant((await read(b)).equals(expected),'Input/metadata mutation '+label);
   lastOutputHash=rr.variants[0].outputSha256;lastInputs={ids:idsBytes,positions:posBytes,tiles:tilesBytes};rr.inputsAndMetadataUnchanged=true;rr.status='passed';
  }
  if(benchmark&&!spec.zeroTilesOverride){
   const hasTimestamp=device.features.has('timestamp-query'),query=hasTimestamp?device.createQuerySet({type:'timestamp',count:2}):undefined;if(query)queries.add(query);
   const resolved=query?allocate(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC,'timestamps'):undefined,count=Number(value('--dispatches',1)),measured=Number(value('--rounds',5));invariant(count>=1&&count<=64&&measured>=1&&measured<=20,'Benchmark bounds');
   for(const candidate of lanes.slice(1)){
    const b={case:spec.name,baseline:'baseline',candidate:candidate.name,warmups:2,measured,dispatchesPerPass:count,includesGather:false,inputReuse:'same uploaded synthetic inputs; warm kernel throughput, not model/Mac prediction',pairs:[]};report.benchmarks.push(b);
    for(let i=0;i<measured+2;i++){
     const order=i%2?[candidate,lanes[0]]:[lanes[0],candidate],rr={order:order.map(x=>x.name),values:{}};
     for(const lane of order){const start=performance.now(),cmd=encode(lane.p,lane.g,spec.grid,count,query,resolved),encoded=performance.now();device.queue.submit([cmd]);await device.queue.onSubmittedWorkDone();const done=performance.now();let gpuMsPerDispatch;
      if(query){const ts=await read(resolved);gpuMsPerDispatch=Number(ts.readBigUInt64LE(8)-ts.readBigUInt64LE(0))/1e6/count;}
      rr.values[lane.name]={gpuMsPerDispatch,cpuEncodeMs:encoded-start,submitWaitMs:done-encoded};}
     if(i>=2)b.pairs.push(rr);
    }
    const median=a=>{a=[...a].sort((x,y)=>x-y);return (a[Math.floor(a.length/2)]+a[Math.floor((a.length-1)/2)])/2;};
    b.summary=Object.fromEntries(['baseline',candidate.name].map(name=>[name,Object.fromEntries(['gpuMsPerDispatch','cpuEncodeMs','submitWaitMs'].map(key=>[key,b.pairs[0].values[name][key]===undefined?undefined:median(b.pairs.map(p=>p.values[name][key]))]))]));
    b.gpuSpeedup=b.summary.baseline.gpuMsPerDispatch/b.summary[candidate.name].gpuMsPerDispatch;
    console.log(JSON.stringify({case:spec.name,candidate:candidate.name,gpuSpeedup:b.gpuSpeedup,summary:b.summary}));
   }
   for(const lane of lanes)invariant(hash(await read(lane.dst))===lastOutputHash,'Benchmark changed output/guards '+lane.name);
   for(const [label,b,expected] of [['weights',wa,data.weights],['input',xx,data.input],['ids',ids,lastInputs.ids],['positions',pos,lastInputs.positions],['tiles',tiles,lastInputs.tiles],['gatherParams',gp,Buffer.from(gparam.buffer)],['mainParams',mp,Buffer.from(mparam.buffer)]])invariant((await read(b)).equals(expected),'Benchmark mutated input '+label);
   row.afterBenchmarkOutputsAndInputsUnchanged=true;
   if(query){query.destroy();queries.delete(query);}if(resolved){resolved.destroy();live.delete(resolved);}
  }
  row.status='passed';console.log(JSON.stringify({case:spec.name,status:row.status,rounds:row.rounds.length,variants:names.length,tiles:row.rounds[0].gather.actualTiles,invalidWg:row.rounds[0].gather.invalidWorkgroups}));
  await device.queue.onSubmittedWorkDone();for(const b of [...live])if(!caseBuffers.has(b)){b.destroy();live.delete(b);}
 }
 invariant(!report.errors.length,'Uncaptured GPU validation error');report.pipelineCount=pipelineCache.size;report.status='passed';
}catch(error){report.status='failed';report.failure={message:error.message,stack:error.stack};}
finally{for(const q of queries)q.destroy();for(const b of live)b.destroy();device?.destroy();writeFileSync(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,cases:report.cases.length,benchmarks:report.benchmarks.length,failure:report.failure}));}
process.exitCode=report.status==='passed'?0:1;
