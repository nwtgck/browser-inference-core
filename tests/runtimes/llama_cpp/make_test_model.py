#!/usr/bin/env python3
"""Generate a small untrained GGUF fixture using only the standard library."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import struct


def string(value):
    data=value.encode('utf-8')
    return struct.pack('<Q',len(data))+data

def scalar(kind,value):
    if kind==8: return string(value)
    return struct.pack({4:'<I',5:'<i',6:'<f',7:'<?'}[kind],value)

def create(path: Path):
    vocab=['<unk>','<s>','</s>']+[f'<0x{i:02X}>' for i in range(256)]
    metadata=[('general.architecture',8,'llama'),('general.name',8,'synthetic-boundary-test-untrained'),
              ('llama.context_length',4,128),('llama.embedding_length',4,32),
              ('llama.block_count',4,1),('llama.feed_forward_length',4,64),
              ('llama.attention.head_count',4,4),('llama.attention.head_count_kv',4,4),
              ('llama.attention.layer_norm_rms_epsilon',6,1e-5),('llama.rope.dimension_count',4,8),
              ('tokenizer.ggml.model',8,'llama'),('tokenizer.ggml.tokens',9,(8,vocab)),
              ('tokenizer.ggml.scores',9,(6,[0.0]*len(vocab))),
              ('tokenizer.ggml.token_type',9,(5,[2,3,3]+[6]*256)),
              ('tokenizer.ggml.bos_token_id',4,1),('tokenizer.ggml.eos_token_id',4,2),
              ('tokenizer.ggml.unknown_token_id',4,0),('tokenizer.ggml.add_bos_token',7,True)]
    shapes=[('token_embd.weight',[32,len(vocab)]),('output_norm.weight',[32]),
            ('output.weight',[32,len(vocab)]),('blk.0.attn_norm.weight',[32]),
            *[(f'blk.0.attn_{part}.weight',[32,32]) for part in ('q','k','v','output')],
            ('blk.0.ffn_norm.weight',[32]),('blk.0.ffn_gate.weight',[32,64]),
            ('blk.0.ffn_down.weight',[64,32]),('blk.0.ffn_up.weight',[32,64])]
    data=bytearray(); descriptors=bytearray(); rng=random.Random(1847)
    for name,dims in shapes:
        data.extend(b'\0'*((-len(data))%32)); offset=len(data)
        count=1
        for dim in dims: count*=dim
        values=[1.0]*count if 'norm.weight' in name else [rng.uniform(-0.1,0.1) for _ in range(count)]
        data.extend(struct.pack('<'+'f'*count,*values))
        descriptors.extend(string(name)+struct.pack('<I',len(dims)))
        descriptors.extend(struct.pack('<'+'Q'*len(dims),*dims))
        descriptors.extend(struct.pack('<IQ',0,offset)) # GGML_TYPE_F32
    header=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(shapes),len(metadata)))
    for name,kind,value in metadata:
        header.extend(string(name)+struct.pack('<I',kind))
        if kind==9:
            subtype,items=value
            header.extend(struct.pack('<IQ',subtype,len(items)))
            for item in items: header.extend(scalar(subtype,item))
        else: header.extend(scalar(kind,value))
    header.extend(descriptors); header.extend(b'\0'*((-len(header))%32))
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(header+data)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    a=p.parse_args(); print(create(a.output))
