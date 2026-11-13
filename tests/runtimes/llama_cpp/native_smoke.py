#!/usr/bin/env python3
"""Test a native-linked llama.cpp build; this does not validate the Wasm runtime."""
from __future__ import annotations
import argparse
import ctypes as c
import json
import hashlib
from pathlib import Path
import math


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True); p.add_argument('--schema',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); lib=c.CDLL(str(a.library.resolve())); schema=json.loads(a.schema.read_text())
    def native(name, result, params):
        f=getattr(lib,'lcb_'+name); f.restype=result; f.argtypes=params; return f
    u64=c.c_uint64; u32=c.c_uint32; i32=c.c_int32
    alloc=native('malloc',u64,[u64]); free=native('free',None,[u64])
    size=native('sizeof_record',u64,[u32]); off=native('offsetof_field',u64,[u32,u32]); fieldsize=native('sizeof_field',u64,[u32,u32])
    byname={x['name']:x for x in schema['records']}; checks=[]
    def check(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok: raise AssertionError(name)
    def record(name): return alloc(size(byname[name]['id']))
    def set32(name,ptr,field,value):
        r=byname[name]; f=next(x for x in r['fields'] if x['name']==field)
        check('field-size:'+name+'.'+field,fieldsize(r['id'],f['id'])==4)
        c.c_int32.from_address(ptr+off(r['id'],f['id'])).value=value
    check('all-generated-exports-resolve',all(hasattr(lib,f['export'][1:]) for f in schema['functions']))
    hash_pointer=native('schema_hash',u64,[])()
    check('schema-hash-matches',c.string_at(hash_pointer).decode()==hashlib.sha256(a.schema.read_bytes()).hexdigest())
    native('llama_backend_init',None,[])()
    mp=record('llama_model_params'); native('llama_model_default_params',None,[u64])(mp)
    set32('llama_model_params',mp,'n_gpu_layers',0)
    set32('llama_model_params',mp,'load_mode',0)
    set32('llama_model_params',mp,'lazy_mode',0)
    path=c.create_string_buffer(str(a.model.resolve()).encode())
    model=native('llama_model_load_from_file',u64,[u64,u64])(c.addressof(path),mp)
    check('synthetic-gguf-loaded',model!=0)
    cp=record('llama_context_params'); native('llama_context_default_params',None,[u64])(cp)
    for name,value in [('n_ctx',128),('n_batch',16),('n_ubatch',16),('n_threads',1),('n_threads_batch',1)]:
        set32('llama_context_params',cp,name,value)
    ctx=native('llama_init_from_model',u64,[u64,u64])(model,cp)
    check('context-created',ctx!=0)
    tokens=(i32*1)(1); batch=record('llama_batch')
    native('llama_batch_get_one',None,[u64,u64,i32])(batch,c.addressof(tokens),1)
    check('decode-one-token',native('llama_decode',i32,[u64,u64])(ctx,batch)==0)
    vocab=native('llama_model_get_vocab',u64,[u64])(model)
    count=native('llama_vocab_n_tokens',i32,[u64])(vocab)
    logits=native('llama_get_logits_ith',u64,[u64,i32])(ctx,-1)
    check('finite-logits',all(math.isfinite(x) for x in (c.c_float*count).from_address(logits)))
    sampler=native('llama_sampler_init_greedy',u64,[])()
    token=native('llama_sampler_sample',i32,[u64,u64,i32])(sampler,ctx,-1)
    check('sampled-token-in-range',0<=token<count)
    state_size=native('llama_state_get_size',u64,[u64])(ctx)
    state=alloc(state_size)
    used=native('llama_state_get_data',u64,[u64,u64,u64])(ctx,state,state_size)
    check('state-saved',0<used<=state_size)
    restored=native('llama_state_set_data',u64,[u64,u64,u64])(ctx,state,used)
    check('state-restored',restored==used)
    native('llama_sampler_free',None,[u64])(sampler)
    native('llama_free',None,[u64])(ctx)
    native('llama_model_free',None,[u64])(model)
    for ptr in (mp,cp,batch,state): free(ptr)
    native('llama_backend_free',None,[])()
    result={'scope':'native-linked llama.cpp; synthetic untrained model; not a browser/Wasm validation',
            'checks':checks,'passed':len(checks),'failed':0,'functions':len(schema['functions']),
            'sampledToken':token,'stateBytes':used}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
