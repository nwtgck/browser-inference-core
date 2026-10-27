// Actual factory WGSL and untouched staged RMS/SCALE references. No browser.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
const [directory, output, mode, scaleWgArgument] = process.argv.slice(2);
const scaleWorkgroupSize=Number(scaleWgArgument);
if (!directory || !output || !['numerical','benchmark'].includes(mode) || process.argv.length !== 6 || !Number.isInteger(scaleWorkgroupSize) || scaleWorkgroupSize<1 || scaleWorkgroupSize>1024) throw new Error('Usage: check_gpu.mjs SHADERS REPORT MODE SCALE_WORKGROUP_SIZE');
const report = { schema:1,scope:'Actual prepared RMS/scale shader vs unchanged upstream staged shaders; standalone, no browser/model/target-device claim',stagedLayout:'RMS writes the final allocation and SCALE runs in place, matching allocator reuse',tolerance:{absolute:3e-6,relative:3e-5},node:process.version,nonfinitePolicy:"Only observed staged/fused classification parity is required for overflow, nonfinite and zero-epsilon probes; CPU classes are recorded without claiming universal WGSL nonfinite behavior",cases:[],benchmarks:[],errors:[],ok:false };
const hash=x=>createHash('sha256').update(x).digest('hex'),invariant=(ok,message)=>{if(!ok)throw new Error(message);};
const buffers=new Set(),ownerKey=Symbol('RMS/scale Dawn owner');let device;
try{
 const{create,globals}=await import(process.env.LCB_TEST_WEBGPU_MODULE);Object.assign(globalThis,globals);globalThis[ownerKey]=create([]);
 const adapter=await globalThis[ownerKey].requestAdapter();invariant(adapter,'No WebGPU adapter');
 report.adapter=Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize'].map(k=>[k,adapter.info[k]]));
 invariant(scaleWorkgroupSize===adapter.limits.maxComputeInvocationsPerWorkgroup && scaleWorkgroupSize<=adapter.limits.maxComputeWorkgroupSizeX,'Reference SCALE WG must match the production adapter maximum: '+adapter.limits.maxComputeInvocationsPerWorkgroup);
 report.workgroupSizes={rms:128,fused:128,scale:scaleWorkgroupSize};
 const timed=mode==='benchmark';invariant(!timed||adapter.features.has('timestamp-query'),'Missing timestamp-query');
 device=await adapter.requestDevice({requiredFeatures:timed?['timestamp-query']:[],requiredLimits:{minStorageBufferOffsetAlignment:adapter.limits.minStorageBufferOffsetAlignment,maxComputeInvocationsPerWorkgroup:scaleWorkgroupSize,maxComputeWorkgroupSizeX:scaleWorkgroupSize}});
 device.addEventListener('uncapturederror',e=>report.errors.push(e.error.message));
 const alignment=device.limits.minStorageBufferOffsetAlignment,programs=new Map();report.shaderSha256={};report.storageAlignment=alignment;
 for(const name of ['fused-0','fused-1','rms','rms-inplace','scale','scale-inplace']){
  const code=await readFile(join(directory,name+'.wgsl'),'utf8'),module=device.createShaderModule({code}),info=await module.getCompilationInfo();
  invariant(!info.messages.some(m=>m.type==='error'),name+': '+info.messages.map(m=>m.message).join('\n'));
  report.shaderSha256[name]=hash(code);programs.set(name,await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}));
 }
 function allocate(size,usage){const b=device.createBuffer({size:Math.ceil(size/4)*4,usage});buffers.add(b);return b;}
 function upload(data,usage=GPUBufferUsage.STORAGE){const b=allocate(data.byteLength,usage|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(b,0,data);return b;}
 async function read(b){const stage=allocate(b.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(b,0,stage,0,b.size);device.queue.submit([encoder.finish()]);await stage.mapAsync(GPUMapMode.READ);const bytes=new Uint8Array(stage.getMappedRange().slice(0));stage.unmap();stage.destroy();buffers.delete(stage);return bytes;}
 async function release(){await device.queue.onSubmittedWorkDone();for(const b of buffers)b.destroy();buffers.clear();}
 const align=(x)=>Math.ceil(x/alignment)*alignment;
 function fixture(spec){
  const{width,heads,tokens,sequences,padding,placement,prefix,probe,coefficients}=spec,n=width*heads*tokens*sequences;
  const nb1=width+(padding===1?3:0),nb2=nb1*heads+(padding?width*heads*2:0),nb3=nb2*tokens+(padding?5:0),span=(sequences-1)*nb3+(tokens-1)*nb2+(heads-1)*nb1+width;
  const srcOffset=prefix+(padding===2?width*heads*4:0),dstOffset=placement==='inplace'?srcOffset:placement==='same-buffer-disjoint'?align(srcOffset+span*4+alignment)+prefix%alignment:prefix+16;
  const same=placement!=='separate',sizeSrc=srcOffset+span*4+32,sizeDst=dstOffset+n*4+32;
  const source=new Uint8Array(same?Math.max(sizeSrc,sizeDst):sizeSrc).fill(0x39),destination=same?source:new Uint8Array(sizeDst).fill(0x39),logical=new Float32Array(n);
  const view=new DataView(source.buffer);let index=0;
  for(let i3=0;i3<sequences;i3++)for(let i2=0;i2<tokens;i2++)for(let i1=0;i1<heads;i1++)for(let col=0;col<width;col++){
   let value=probe==='zero'?(col%2?-0:0):probe==='tiny'?Math.fround(((col%7)-3)*2**-100):probe==='infinity'?(col===0?Infinity:col===1?-Infinity:1):probe==='nan'?NaN:probe==='large'?Math.fround(3.402823466e38):Math.fround(Math.sin(col*0.43+i1*0.71+i2*0.13+i3*0.29)*3.7);
   logical[index++]=value;view.setFloat32(srcOffset+(i3*nb3+i2*nb2+i1*nb1+col)*4,value,true);
  }
  const initial=new Uint32Array(destination.buffer);for(let i=0;i<n;i++)initial[dstOffset/4+i]=0x7eabcdef;
  if(placement==='inplace'){index=0;for(let i3=0;i3<sequences;i3++)for(let i2=0;i2<tokens;i2++)for(let i1=0;i1<heads;i1++)for(let col=0;col<width;col++)view.setFloat32(srcOffset+(i3*nb3+i2*nb2+i1*nb1+col)*4,logical[index++],true);}
  const sourceStart=Math.floor(srcOffset/alignment)*alignment,destinationStart=Math.floor(dstOffset/alignment)*alignment;
  const [eps,postScale,postBias]=coefficients.map(Math.fround),cpu=new Float64Array(n);
  for(let r=0;r<n/width;r++){let sum=0;for(let c=0;c<width;c++)sum+=logical[r*width+c]**2;const normalizer=1/Math.sqrt(sum/width+eps);for(let c=0;c<width;c++)cpu[r*width+c]=(normalizer*(logical[r*width+c]-0))*postScale+postBias;}
  return{n,span,nb1,nb2,nb3,srcOffset,dstOffset,sourceStart,destinationStart,source,destination,cpu,same,rows:heads*tokens*sequences,eps,postScale,postBias};
 }
 function params(spec,data,sourceIndex,destinationIndex,sourceStrides,destinationStrides,fused){const p=new Uint32Array([sourceIndex,destinationIndex,...sourceStrides,...destinationStrides,spec.width,spec.heads,spec.tokens,spec.sequences,0,...(fused?[0,0]:[])]);const f=new Float32Array(p.buffer);f[12]=data.eps;if(fused){f[13]=data.postScale;f[14]=data.postBias;}return p;}
 function scaleParams(n,source,destination,data){const p=new Uint32Array([source,destination,n,n,n,n,n,n,n,n,1,1,0,0]);const f=new Float32Array(p.buffer);f[12]=data.postScale;f[13]=data.postBias;return p;}
 function group(name,ranges,words){const pipeline=programs.get(name),u=upload(words,GPUBufferUsage.UNIFORM);return{pipeline,group:device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...ranges.map((resource,binding)=>({binding,resource})),{binding:ranges.length,resource:{buffer:u}}]})};}
 function commands(entries,repeat,query=null,resolve=null){const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass(query?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});for(let i=0;i<repeat;i++)for(const e of entries){pass.setPipeline(e.pipeline);pass.setBindGroup(0,e.group);pass.dispatchWorkgroups(e.x,e.y);}pass.end();if(query)encoder.resolveQuerySet(query,0,2,resolve,0);return encoder.finish();}
 function setup(spec,data,fused){
  const src=upload(data.source),dst=data.same?src:upload(data.destination),sourceRange={buffer:src,offset:data.sourceStart,size:data.srcOffset-data.sourceStart+data.span*4},destinationRange={buffer:dst,offset:data.destinationStart,size:data.dstOffset-data.destinationStart+data.n*4};
  const sourceIndex=(data.srcOffset-data.sourceStart)/4,destinationIndex=(data.dstOffset-data.destinationStart)/4,sourceStrides=[data.nb1,data.nb2,data.nb3],destinationStrides=[spec.width,spec.width*spec.heads,spec.width*spec.heads*spec.tokens],inplace=spec.placement==='inplace';
  const entries=[];
  function append(entry,groups){const y=Math.ceil(groups/7),x=Math.ceil(groups/y);entries.push({...entry,x,y});}
  if(fused)append(group('fused-'+Number(inplace),inplace?[sourceRange]:[sourceRange,destinationRange],params(spec,data,sourceIndex,destinationIndex,sourceStrides,destinationStrides,true)),data.rows);
  else if(inplace){append(group('rms-inplace',[sourceRange],params(spec,data,sourceIndex,sourceIndex,sourceStrides,sourceStrides,false)),data.rows);append(group('scale-inplace',[sourceRange],scaleParams(data.n,sourceIndex,sourceIndex,data)),Math.ceil(data.n/scaleWorkgroupSize));}
  else{append(group('rms',[sourceRange,destinationRange],params(spec,data,sourceIndex,destinationIndex,sourceStrides,destinationStrides,false)),data.rows);append(group('scale-inplace',[destinationRange],scaleParams(data.n,destinationIndex,destinationIndex,data)),Math.ceil(data.n/scaleWorkgroupSize));}
  return{src,dst,entries};
 }
 async function check(spec,data,entry){const destination=await read(entry.dst),actual=new Float32Array(destination.buffer,data.dstOffset,data.n),bits=new Uint32Array(destination.buffer,data.dstOffset,data.n),reference= data.destination;let guardErrors=0,inputErrors=0,cpuErrors=0,nonfinite=0,unwritten=0,maxAbsolute=0,maxRelative=0;
  for(let i=0;i<destination.length;i++)if(i<data.dstOffset||i>=data.dstOffset+data.n*4)guardErrors+=Number(destination[i]!==reference[i]);
  if(!data.same){const original=await read(entry.src);for(let i=0;i<data.source.length;i++)inputErrors+=Number(original[i]!==data.source[i]);}
  for(let i=0;i<data.n;i++){const error=Math.abs(actual[i]-data.cpu[i]);maxAbsolute=Math.max(maxAbsolute,error);maxRelative=Math.max(maxRelative,error/Math.max(1e-30,Math.abs(data.cpu[i])));unwritten+=Number(bits[i]===0x7eabcdef);nonfinite+=Number(!Number.isFinite(actual[i]));cpuErrors+=Number(error>report.tolerance.absolute+report.tolerance.relative*Math.abs(data.cpu[i]));}
  return{actual:actual.slice(),bits:bits.slice(),guardErrors,inputErrors,cpuErrors,nonfinite,unwritten,maxAbsolute,maxRelative};
 }
 const cases=[];for(const width of [1,7,127,128,129,257])for(const heads of [1,3,16])for(const variant of [0,1,2,3,4,5]){
  const padding=variant===3?1:variant>=4?2:0,placement=variant===2?'inplace':variant===1||variant===5?'same-buffer-disjoint':'separate';
  const tokens=heads===3?3:1,sequences=heads===3?2:1,prefix=[0,4,alignment-4,alignment+4][cases.length%4];
  const coefficients=cases.length%2?[1e-6/width,-0.3,0.07]:[1e-6/width,1/Math.sqrt(width),0];cases.push({width,heads,tokens,sequences,padding,placement,prefix,probe:'normal',coefficients});
 }
 for(const probe of ['zero','tiny'])for(const placement of ['inplace','separate'])cases.push({width:128,heads:16,tokens:1,sequences:1,padding:0,placement,prefix:4,probe,coefficients:[1e-6,1,-0]});
 for(const tokens of [3,17])cases.push({width:128,heads:16,tokens,sequences:2,padding:2,placement:'separate',prefix:4,probe:'normal',coefficients:[1e-6/128,1/Math.sqrt(128),0]});
 for(const [probe,coefficients] of [['zero',[0,1,0]],['zero',[-1,1,0]],['tiny',[0,1,0]],['infinity',[1e-6,1,0]],['nan',[1e-6,1,0]],['large',[1e-6,1,0]],['normal',[Infinity,Infinity,0]]])for(const placement of ['inplace','separate'])cases.push({width:128,heads:3,tokens:1,sequences:1,padding:0,placement,prefix:4,probe,coefficients,observedOnly:true});
 const classification=value=>Number.isNaN(value)?'nan':value===Infinity?'+inf':value===-Infinity?'-inf':Object.is(value,-0)?'-zero':value===0?'+zero':'finite';
 for(const[caseIndex,spec]of cases.entries()){
  const data=fixture(spec),old=setup(spec,data,false),candidate=setup(spec,data,true);device.queue.submit([commands(old.entries,1),commands(candidate.entries,1)]);
  const a=await check(spec,data,old),b=await check(spec,data,candidate);let bitDifferences=0;for(let i=0;i<data.n;i++)bitDifferences+=Number(a.bits[i]!==b.bits[i]);
  const clean=r=>({guardErrors:r.guardErrors,inputErrors:r.inputErrors,cpuErrors:r.cpuErrors,nonfinite:r.nonfinite,unwritten:r.unwritten,maxAbsolute:r.maxAbsolute,maxRelative:r.maxRelative});
  const stagedClasses=[...a.actual].map(classification),fusedClasses=[...b.actual].map(classification),classificationDifferences=fusedClasses.reduce((count,value,i)=>count+Number(value!==stagedClasses[i]),0);
  report.cases.push({...spec,caseIndex,elements:data.n,bitDifferences,staged:clean(a),fused:clean(b),...(spec.observedOnly?{stagedClasses,fusedClasses,cpuClasses:[...data.cpu].map(classification),classificationDifferences}:{})});
  invariant(!a.guardErrors&&!b.guardErrors&&!a.inputErrors&&!b.inputErrors&&!a.unwritten&&!b.unwritten&&(spec.observedOnly?classificationDifferences===0:!a.cpuErrors&&!b.cpuErrors&&!a.nonfinite&&!b.nonfinite),'Numerical/range failure: '+JSON.stringify(report.cases.at(-1)));
  await release();
 }
 if(timed){const median=a=>{const s=[...a].sort((x,y)=>x-y),n=s.length;return(s[Math.floor(n/2)]+s[Math.ceil(n/2)-1])/2;};for(const tokens of [1,17,128]){
  const spec={width:128,heads:16,tokens,sequences:1,padding:2,placement:'separate',prefix:4,probe:'normal',coefficients:[1e-6/128,1/Math.sqrt(128),0]},data=fixture(spec),a=setup(spec,data,false),b=setup(spec,data,true),entries=[a,b],query=device.createQuerySet({type:'timestamp',count:2}),resolve=allocate(16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),pairs=[],times=[[],[]];
  for(let rep=0;rep<16;rep++){const pair={};for(const index of rep%2?[1,0]:[0,1]){device.queue.submit([commands(entries[index].entries,32,query,resolve)]);const stamps=new BigUint64Array((await read(resolve)).buffer),interval=stamps[1]-stamps[0];invariant(interval>=0n&&interval<=BigInt(Number.MAX_SAFE_INTEGER),'Invalid timestamp');pair[index?'fusedMs':'stagedMs']=Number(interval)/1e6/32;}if(rep>=4){pairs.push(pair);times[0].push(pair.stagedMs);times[1].push(pair.fusedMs);}}
  query.destroy();const old=await check(spec,data,a),fused=await check(spec,data,b);let bits=0;for(let i=0;i<data.n;i++)bits+=Number(old.bits[i]!==fused.bits[i]);
  invariant(!old.cpuErrors&&!fused.cpuErrors&&!old.nonfinite&&!fused.nonfinite&&!old.unwritten&&!fused.unwritten&&!old.guardErrors&&!fused.guardErrors&&!old.inputErrors&&!fused.inputErrors,'Post-timing validation');report.benchmarks.push({width:128,heads:16,tokens,warmupPairs:4,measuredPairs:12,chainsPerPass:32,ordering:'AB/BA alternating, same logical input, separate per-variant buffers',pairs,stagedMedianMs:median(times[0]),fusedMedianMs:median(times[1]),ratio:median(times[0])/median(times[1]),postTimingBits:bits,postTiming:{staged:{maxAbsolute:old.maxAbsolute,maxRelative:old.maxRelative,cpuErrors:old.cpuErrors,nonfinite:old.nonfinite,unwritten:old.unwritten,guardErrors:old.guardErrors,inputErrors:old.inputErrors},fused:{maxAbsolute:fused.maxAbsolute,maxRelative:fused.maxRelative,cpuErrors:fused.cpuErrors,nonfinite:fused.nonfinite,unwritten:fused.unwritten,guardErrors:fused.guardErrors,inputErrors:fused.inputErrors}}});await release();
 }}
 invariant(!report.errors.length,'WebGPU validation errors');report.finiteBitDifferences=report.cases.filter(c=>!c.observedOnly).reduce((n,c)=>n+c.bitDifferences,0);report.ok=true;
}catch(error){report.errors.push(error.stack??String(error));}
finally{for(const b of buffers)b.destroy();device?.destroy();await writeFile(output,JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({ok:report.ok,cases:report.cases.length,finiteBitDifferences:report.finiteBitDifferences,benchmarks:report.benchmarks.length,errors:report.errors}));
process.exit(report.ok?0:1);
