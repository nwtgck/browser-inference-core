#!/usr/bin/env python3
"""Generate self-authored packed inputs; no model fixtures or downloads."""
from pathlib import Path
import hashlib,json,struct,subprocess,sys
from types import SimpleNamespace
from emit_factory import function

HERE=Path(__file__).resolve().parent

def generate(upstream: Path, work: Path, factory: Path, *, large: bool=False) -> Path:
    UP,LAB=upstream,work
    LAB.mkdir(parents=True,exist_ok=True)
    identity=json.loads((HERE/'inputs.json').read_text())
    for name,expected in identity['oracle_inputs_sha256'].items():
        if hashlib.sha256((UP/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('CPU fixture oracle contract changed: '+name)
    args=SimpleNamespace(large=large,large_type='both',large_pattern='paired',sweep=True)
    variants={'baseline':None,'candidate':None}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    # Source-bound packed-byte oracle: unchanged upstream decoders only.
    qt=(UP/'ggml/src/ggml-quants.c').read_text();oracle='#define GGML_COMMON_DECL_C\n#define GGML_COMMON_IMPL_C\n#include "ggml-common.h"\n#include "ggml-impl.h"\n'
    for typ in ['q8_0','iq3_s']:oracle+=function(qt,'void dequantize_row_'+typ+'(')+'\n'
    oracle+=r'''
    #include <stdio.h>
    int main(int argc,char**argv){if(argc!=6)return 2;int kind=atoi(argv[1]);size_t k=strtoull(argv[2],0,10),rows=strtoull(argv[3],0,10);FILE*f=fopen(argv[4],"rb"),*o=fopen(argv[5],"wb");if(!f||!o)return 3;size_t bytes=kind==1?k/32*34:k/256*110;void*packed=malloc(bytes);float*decoded=malloc(k*4);for(size_t r=0;r<rows;r++){if(fread(packed,1,bytes,f)!=bytes)return 4;if(kind==1)dequantize_row_q8_0(packed,decoded,k);else dequantize_row_iq3_s(packed,decoded,k);fwrite(decoded,4,k,o);}free(packed);free(decoded);fclose(f);fclose(o);return 0;}
    '''
    (LAB/'oracle.c').write_text(oracle)
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-ffp-contract=off','-isystem',str(UP/'ggml/include'),'-isystem',str(UP/'ggml/src'),str(LAB/'oracle.c'),'-lm','-o',str(LAB/'oracle')],check=True)
    shaders=LAB/'shaders';shaders.mkdir(exist_ok=True);fixtures=LAB/'fixtures';fixtures.mkdir(exist_ok=True)
    cases=[]
    for typ in ['f32','q8','iq3']:
     for pattern in ['uniform','concentrated','skewed','duplicate']:
      cases.append((typ,pattern,35 if typ=='f32' else 256,7 if pattern in ['uniform','skewed'] else 36,11,4,9,pattern in ['uniform','duplicate'],True))
     cases.extend([(typ,'fulltiles',32 if typ=='f32' else 256,32,4,4,32,False,False),(typ,'zero',32 if typ=='f32' else 256,7,7,3,5,True,True)])
    for typ in ['f32','q8','iq3']:
     for count in [1,3,4,5,31,33]:cases.append((typ,'uniform' if count==1 else 'concentrated',32 if typ=='f32' else 256,7,2 if count==1 else 1,1,2 if count==1 else count,True,True))
     cases.append((typ,'uniform',32 if typ=='f32' else 256,7,512,10,2,False,True))
    cases.append(('f32','uniform',36,36,11,4,9,False,True))
    for typ in ['f32','q8','iq3']:
     for pattern in ['mixed-skipped','all-skipped']:
      cases.append((typ,pattern,35 if typ=='f32' else 256,7,11,4,9,True,True))
    if args.large:
     for typ in (['q8','iq3'] if args.large_type=='both' else [args.large_type]):
      for pattern in (['uniform','concentrated','skewed','duplicate'] if args.large_pattern=='all' else ['uniform','concentrated'] if args.large_pattern=='paired' else [args.large_pattern]):
       for k in ([256,1024,2560] if args.sweep else [2560]):cases.append((typ,pattern,k,32,512,10,128,True,False))
    manifest_rows=[]
    for seed,(typ,pattern,k,m,e,u,t,shared,padded) in enumerate(cases):
     kind=['f32','q8','iq3'].index(typ);vec=typ=='f32' and k%4==0 and m%4==0;block=1 if typ=='f32' else 32 if typ=='q8' else 256;bs=4 if typ=='f32' else 34 if typ=='q8' else 110
     offa=(4 if vec else 1) if padded else 0;offx=(4 if vec else 3) if padded else 0;offd=(8 if vec else 7) if padded else 0;offids=3 if padded else 0
     rowblocks=k//block if typ!='f32' else k;s01=rowblocks+(4 if vec else 1) if padded else rowblocks;s02=m*s01+(4 if vec else 3) if padded else m*s01;bn=1 if shared else u;s11=k+(4 if vec else 5) if padded else k;s12=bn*s11+(4 if vec else 3) if padded else bn*s11;sid=u+2 if padded else u
     total=(offa+e*s02)*bs+16;weights=bytearray([0x42])*((total+3)//4*4);packed=bytearray()
     for ex in range(e):
      for row in range(m):
       raw=bytearray()
       for b in range(rowblocks):
        ph=(ex*7+row*3+b+seed)
        if typ=='f32':chunk=struct.pack('<f',((ph%7)-3)/16)
        elif typ=='q8':chunk=struct.pack('<e',1/64)+bytes(((j*3+ph)%15-7)&255 for j in range(32))
        else:
         chunk=bytearray(110);struct.pack_into('<e',chunk,0,1/1024);chunk[2:66]=bytes((j*17+ph*13)%256 for j in range(64));chunk[66:74]=bytes((j*31+ph)%256 for j in range(8));chunk[74:106]=bytes((j*43+ph*19)%256 for j in range(32));chunk[106:110]=bytes((ph+j)%16 for j in range(4))
        raw+=chunk
       base=(offa+ex*s02+row*s01)*bs;weights[base:base+len(raw)]=raw;packed+=raw
     inp=bytearray(struct.pack('<f',99.5)*(offx+t*s12+16))
     for token in range(t):
      for slot in range(bn):
       for col in range(k):struct.pack_into('<f',inp,4*(offx+token*s12+slot*s11+col),(((col*3+token*5+slot*7+seed)%7)-3)/16)
     ids=[0x7fffffff]*(offids+t*sid+8)
     for token in range(t):
      for slot in range(u):
       if pattern in ['concentrated','fulltiles']:ex=slot%e
       elif pattern=='duplicate':ex=(token//2)%e
       elif pattern=='skewed':ex=0 if (token*u+slot)%5 else (token*3+slot)%e
       else:ex=(token*u+slot)*5%e
       if pattern=='all-skipped' or (pattern=='mixed-skipped' and (token*u+slot)%3==0):ex=-1
       ids[offids+token*sid+slot]=ex
     name=f'{typ}-{pattern}-k{k}-m{m}-e{e}-u{u}-t{t}';d=fixtures/name;d.mkdir(exist_ok=True)
     (d/'weights.bin').write_bytes(weights);(d/'input.bin').write_bytes(inp);(d/'ids.bin').write_bytes(struct.pack('<'+'i'*len(ids),*ids));(d/'packed.bin').write_bytes(packed)
     if typ=='f32':(d/'dequantized.bin').write_bytes(packed)
     else:subprocess.run([str(LAB/'oracle'),str(kind),str(k),str(e*m),str(d/'packed.bin'),str(d/'dequantized.bin')],check=True)
     p=[kind,k,m,e,u,t,bn,offa,offx,offd,offids,s01,s02,s11,s12,sid]
     actual=json.loads(subprocess.check_output([str(factory),str(shaders),name,*map(str,p)],text=True))
     row=dict(name=name,type=typ,pattern=pattern,k=k,m=m,experts=e,used=u,tokens=t,sharedInput=shared,zeroTilesOverride=pattern=='zero',args=p,**actual,hashes={key:sha(d/(key+'.bin')) for key in ['weights','input','ids','dequantized']},shaderHashes={v:sha(shaders/(name+'-'+v+'.wgsl')) for v in ['gather','gather-baseline',*variants]},large=k>512 or (e==512 and t==128))
     manifest_rows.append(row)
    manifest=LAB/('manifest-large.json' if large else 'manifest.json')
    manifest.write_text(json.dumps(manifest_rows,indent=2)+'\n')
    return manifest
