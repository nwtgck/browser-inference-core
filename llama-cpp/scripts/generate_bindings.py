#!/usr/bin/env python3
"""Generate mechanical ABI bindings, types, and layout queries from Clang declarations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

EXTRA_CONSTANTS = [
    'LLAMA_DEFAULT_SEED', 'LLAMA_TOKEN_NULL',
    'LLAMA_FILE_MAGIC_GGLA', 'LLAMA_FILE_MAGIC_GGSN', 'LLAMA_FILE_MAGIC_GGSQ',
    'LLAMA_SESSION_MAGIC', 'LLAMA_SESSION_VERSION',
    'LLAMA_STATE_SEQ_MAGIC', 'LLAMA_STATE_SEQ_VERSION',
    'LLAMA_STATE_SEQ_FLAGS_NONE', 'LLAMA_STATE_SEQ_FLAGS_SWA_ONLY',
    'LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY', 'LLAMA_STATE_SEQ_FLAGS_ON_DEVICE',
    'GGUF_VERSION', 'GGUF_DEFAULT_ALIGNMENT',
    'GGML_FILE_MAGIC', 'GGML_FILE_VERSION', 'GGML_QNT_VERSION',
    'GGML_QNT_VERSION_FACTOR', 'GGML_MAX_DIMS', 'GGML_MAX_PARAMS',
    'GGML_MAX_SRC', 'GGML_MAX_N_THREADS', 'GGML_MAX_OP_PARAMS', 'GGML_MAX_NAME',
    'GGML_TENSOR_SIZE',
    'GGML_DEFAULT_N_THREADS', 'GGML_DEFAULT_GRAPH_SIZE', 'GGML_MEM_ALIGN',
    'GGML_EXIT_SUCCESS', 'GGML_EXIT_ABORTED',
    'GGML_ROPE_TYPE_NORMAL', 'GGML_ROPE_TYPE_NEOX', 'GGML_ROPE_TYPE_MROPE',
    'GGML_ROPE_TYPE_VISION', 'GGML_ROPE_TYPE_IMROPE', 'GGML_MROPE_SECTIONS',
    'GGML_N_TASKS_MAX', 'GGML_BACKEND_META_MAX_DEVICES',
    'EINVAL', 'EIO', 'EOVERFLOW', 'EROFS', 'ENOENT',
    'SEEK_SET', 'SEEK_CUR', 'SEEK_END',
]

# The pinned header declares this function but no upstream translation unit
# defines it. Keep the reason visible in schema.excluded; reconsider this entry
# when updating the upstream pin, rather than inventing a private implementation.
UNIMPLEMENTED_PUBLIC_FUNCTIONS = {
    'ggml_threadpool_get_n_threads':
        'declared in ggml-cpu.h but not implemented by the pinned upstream',
}

# Explicit synchronous callback-only metadata surface. Normal generated APIs
# retain their JSPI/Promise behavior. These public getters inspect registry,
# operation and buffer metadata; none submits GPU work or reads tensor values.
# Audit this bounded list against the pinned backends when updating upstream.
CALLBACK_METADATA_GETTERS = {
    'ggml_op_desc': ('pointer', ('pointer',)),
    'ggml_backend_buffer_name': ('pointer', ('pointer',)),
    'ggml_backend_buffer_is_host': ('boolean', ('pointer',)),
    'ggml_backend_dev_count': ('u64', ()),
    'ggml_backend_dev_get': ('pointer', ('u64',)),
    'ggml_backend_dev_name': ('pointer', ('pointer',)),
    'ggml_backend_dev_supports_op': ('boolean', ('pointer', 'pointer')),
}

def walk(node):
    yield node
    for child in node.get('inner', []):
        yield from walk(child)

def generate(source: Path, output: Path, compiler: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    unit = output / 'headers.c'
    unit.write_text('#include "llama.h"\n#include "gguf.h"\n#include "ggml.h"\n'
                    '#include "ggml-alloc.h"\n#include "ggml-backend.h"\n'
                    '#include "mtmd.h"\n#include "mtmd-helper.h"\n')
    command = [compiler, '-x', 'c', '-std=c11', '-fsyntax-only',
               '-I'+str(source/'include'), '-I'+str(source/'ggml/include'),
               '-I'+str(source/'tools/mtmd'),
               '-Xclang', '-ast-dump=json', str(unit)]
    ast = json.loads(subprocess.check_output(command, text=True))
    nodes = list(walk(ast))
    aliases = {n['name']: n['type'].get('desugaredQualType', n['type']['qualType'])
               for n in nodes if n.get('kind') == 'TypedefDecl' and 'type' in n}
    def resolve(typ):
        old = None
        while typ != old:
            old = typ
            typ = aliases.get(typ, typ)
        return typ
    def classify(typ):
        # The public BF16 scalar is an anonymous struct typedef passed by value.
        if typ == 'ggml_bf16_t': return 'record'
        t = resolve(typ).strip()
        if '*' in t or t.endswith(']') and '(' in t:
            return 'pointer'
        if re.match(r'^(const )?(struct|union) ', t):
            return 'record'
        if '[' in t:
            return 'array'
        if t == 'void': return 'void'
        if t in ('bool', '_Bool'): return 'boolean'
        if t.startswith('enum '): return 'signed'
        if t in ('float', 'double'): return 'float'
        if typ in ('size_t', 'uintptr_t', 'uint64_t') or t == 'unsigned long' or 'unsigned long long' in t:
            return 'u64'
        if typ in ('ssize_t', 'intptr_t', 'int64_t', 'off_t') or t == 'long' or t == 'long long':
            return 'i64'
        if t in ('unsigned int', 'unsigned short', 'unsigned char', 'unsigned short int'):
            return 'unsigned'
        if t in ('int', 'short', 'short int', 'char', 'signed char'): return 'signed'
        raise ValueError(f'Unsupported C type: {typ!r} -> {t!r}; extend the generator explicitly')
    def js_type(kind):
        if kind in ('pointer', 'record', 'u64', 'i64'): return 'bigint'
        if kind == 'void': return 'void'
        # C bool crosses the low-level boundary as 0 / 1, not JavaScript boolean.
        return 'number'
    def wire_type(kind, typ):
        if kind in ('pointer', 'record', 'u64'): return 'uint64_t'
        if kind == 'i64': return 'int64_t'
        if kind == 'boolean': return 'int32_t'
        if kind == 'signed': return 'int32_t'
        if kind == 'unsigned': return 'uint32_t'
        return typ
    functions, excluded = [], []
    for n in nodes:
        name = n.get('name', '')
        if n.get('kind') != 'FunctionDecl' or not name.startswith(('llama_', 'gguf_', 'ggml_', 'mtmd_')):
            continue
        if name in UNIMPLEMENTED_PUBLIC_FUNCTIONS:
            excluded.append({'name': name, 'reason': UNIMPLEMENTED_PUBLIC_FUNCTIONS[name]})
            continue
        if name.startswith('mtmd_helper_video_') and name != 'mtmd_helper_video_init_params_default':
            excluded.append({'name': name, 'reason': 'requires subprocess video support (MTMD_VIDEO=OFF)'})
            continue
        if any(x.get('kind') == 'DeprecatedAttr' for x in n.get('inner', [])) or n.get('variadic'):
            excluded.append({'name': name, 'reason': 'deprecated' if not n.get('variadic') else 'variadic'})
            continue
        result = n['type']['qualType'].split(' (', 1)[0]
        rk = classify(result)
        params = []
        for i, p in enumerate(x for x in n.get('inner', []) if x.get('kind') == 'ParmVarDecl'):
            typ = p['type']['qualType']; kind = classify(typ)
            if kind == 'array': raise ValueError(f'Unlowered array argument: {name}')
            params.append({'name': p.get('name', f'arg{i}'), 'cType': typ, 'kind': kind})
        functions.append({'name': name, 'export': '_lcb_'+name, 'returnType': result,
                          'returnKind': rk, 'parameters': params})
    functions = sorted({f['name']: f for f in functions}.values(), key=lambda f: f['name'])
    records = []
    for n in nodes:
        name = n.get('name', '')
        if n.get('kind') != 'RecordDecl' or not n.get('completeDefinition') or not name.startswith(('llama_', 'gguf_', 'ggml_', 'mtmd_')):
            continue
        fields = []
        for p in n.get('inner', []):
            if p.get('kind') not in ('FieldDecl', 'IndirectFieldDecl') or not p.get('name'): continue
            if p.get('isBitfield'): raise ValueError(f'Bitfield requires explicit support: {name}.{p["name"]}')
            field_type = p.get('type')
            if field_type is None:
                matches = [f for f in walk(n) if f.get('kind') == 'FieldDecl' and f.get('name') == p['name'] and 'type' in f]
                if len(matches) != 1: raise ValueError(f'Ambiguous anonymous field: {name}.{p["name"]}')
                field_type = matches[0]['type']
            typ = field_type['qualType']
            kind = 'array' if re.search(r'\[[0-9]*\]$', typ) else classify(typ)
            fields.append({'name': p['name'], 'cType': typ, 'kind': kind})
        records.append({'name': name, 'cType': n.get('tagUsed', 'struct')+' '+name, 'fields': fields})
    records.append({'name': 'ggml_bf16_t', 'cType': 'ggml_bf16_t', 'fields': [
        {'name': 'bits', 'cType': 'uint16_t', 'kind': 'unsigned'},
    ]})
    records = sorted({r['name']: r for r in records}.values(), key=lambda r: r['name'])
    enum_names = sorted({n['name'] for n in nodes if n.get('kind') == 'EnumConstantDecl'
                         and n.get('name','').startswith(('LLAMA_', 'GGML_', 'GGUF_', 'MTMD_'))})
    constants = sorted(set(enum_names + EXTRA_CONSTANTS))
    cpp = ['// Generated; do not edit.', '#include "llama.h"', '#include "gguf.h"',
           '#include "ggml.h"', '#include "ggml-alloc.h"', '#include "ggml-backend.h"',
           '#include "mtmd.h"', '#include "mtmd-helper.h"',
           '#include <stdint.h>', '#include <stddef.h>',
           '#include <errno.h>', '#include <stdlib.h>', '#include <stdexcept>',
           'static uintptr_t lcb_checked_pointer(uint64_t p) {',
           '  if (p > UINTPTR_MAX) throw std::range_error("pointer/size exceeds address space");',
           '  return (uintptr_t)p;', '}', 'extern "C" {',
           'uint64_t lcb_malloc(uint64_t size) { return (uint64_t)(uintptr_t)malloc(lcb_checked_pointer(size)); }',
           'void lcb_free(uint64_t ptr) { free((void *)lcb_checked_pointer(ptr)); }',
           'uint32_t lcb_pointer_bytes(void) { return sizeof(void *); }',
           'uint32_t lcb_abi_version(void) { return 1; }',
           'uint32_t lcb_callback_metadata_version(void) { return 1; }']
    missing = CALLBACK_METADATA_GETTERS.keys() - {f['name'] for f in functions}
    if missing:
        raise ValueError(f'Callback metadata getters need upstream review: {sorted(missing)}')
    for f in functions:
        args, callargs, guards = [], [], []
        if f['returnKind'] == 'record':
            args.append('uint64_t out')
            guards.append('if (!out) throw std::invalid_argument("null return storage");')
        for i, p in enumerate(f['parameters']):
            typ, kind = p['cType'], p['kind']; arg = 'a'+str(i)
            args.append(wire_type(kind,typ)+' '+arg)
            if kind == 'record':
                guards.append(f'if (!{arg}) throw std::invalid_argument("null record argument");')
                qualified = typ if typ.startswith('const ') else 'const ' + typ
                callargs.append(f'*(({qualified} *)lcb_checked_pointer({arg}))')
            elif kind == 'pointer': callargs.append(f'({typ})lcb_checked_pointer({arg})')
            elif kind == 'u64' and typ in ('size_t', 'uintptr_t'):
                callargs.append(f'({typ})lcb_checked_pointer({arg})')
            else: callargs.append(f'({typ}){arg}')
        call = f'{f["name"]}({", ".join(callargs)})'
        rk=f['returnKind']; rt=f['returnType']
        if rk == 'record': statement=f'*(({rt} *)lcb_checked_pointer(out)) = {call};'; wt='void'
        elif rk == 'void': statement=call+';'; wt='void'
        elif rk == 'pointer': statement=f'return (uint64_t)(uintptr_t){call};'; wt='uint64_t'
        else: wt=wire_type(rk,rt); statement=f'return ({wt}){call};'
        cpp += [f'{wt} lcb_{f["name"]}({", ".join(args) or "void"}) {{',
                *['  '+g for g in guards], '  '+statement, '}']
        if f['name'] in CALLBACK_METADATA_GETTERS:
            # A separate export avoids depending on WebAssembly.promising()
            # behavior, or changing the ordinary low-level API's contract.
            if (rk, tuple(p['kind'] for p in f['parameters'])) != CALLBACK_METADATA_GETTERS[f['name']]:
                raise ValueError(f'Unexpected callback getter signature: {f["name"]}')
            cpp += [f'{wt} lcb_callback_{f["name"]}({", ".join(args) or "void"}) {{',
                    *['  '+g for g in guards], '  '+statement, '}']

    for i,r in enumerate(records): r['id']=i
    for name, expr in [('sizeof_record', 'sizeof({ctype})'), ('alignof_record', 'alignof({ctype})')]:
        cpp += [f'uint64_t lcb_{name}(uint32_t record) {{ switch(record) {{']
        cpp += [f'case {r["id"]}: return {expr.format(ctype=r["cType"])};' for r in records]
        cpp += ['default: return UINT64_MAX; } }']
    for name, expr in [('offsetof_field','offsetof({ctype}, {field})'), ('sizeof_field','sizeof((({ctype} *)0)->{field})')]:
        cpp += [f'uint64_t lcb_{name}(uint32_t record, uint32_t field) {{ switch(record) {{']
        for r in records:
            cpp += [f'case {r["id"]}: switch(field) {{']
            for i,p in enumerate(r['fields']):
                p['id']=i
                cpp += [f'case {i}: return {expr.format(ctype=r["cType"], field=p["name"])};']
            cpp += ['default: return UINT64_MAX; }']
        cpp += ['default: return UINT64_MAX; } }']
    cpp += ['int64_t lcb_constant(uint32_t id) { switch(id) {']
    cpp += [f'case {i}: return (int64_t)({name});' for i,name in enumerate(constants)]
    cpp += ['default: throw std::out_of_range("constant id"); } }', '}']
    helper_names = ['malloc','free','pointer_bytes','abi_version','sizeof_record','alignof_record','offsetof_field','sizeof_field','constant','schema_hash','callback_metadata_version']
    exports = ['_lcb_'+n for n in helper_names] + [f['export'] for f in functions]
    exports += ['_lcb_callback_'+name for name in sorted(CALLBACK_METADATA_GETTERS)]
    # Direct exports remain available for expert callers bound to this exact native ABI.
    exports += ['_'+f['name'] for f in functions] + ['_malloc','_free']
    (output/'exports.json').write_text(json.dumps(exports,indent=2)+'\n')
    (output/'jspi-exports.json').write_text(json.dumps([f['export'][1:] for f in functions] + [f['name'] for f in functions],indent=2)+'\n')
    schema={'abiVersion':1,'pointerRepresentation':'bigint-u64','functions':functions,
            'records':records,'constants':constants,'excluded':excluded,
            'callbackMetadata': {'version': 1, 'getters': [
                {'name': name, 'export': '_lcb_callback_'+name}
                for name in sorted(CALLBACK_METADATA_GETTERS)]}}
    schema_text=json.dumps(schema,indent=2)+'\n'
    schema_hash=hashlib.sha256(schema_text.encode()).hexdigest()
    cpp.insert(-1, f'uint64_t lcb_schema_hash(void) {{ return (uint64_t)(uintptr_t)"{schema_hash}"; }}')
    (output/'bindings.cpp').write_text(('\n'.join(cpp)+'\n').replace('_Bool', 'bool'))
    (output/'schema.json').write_text(schema_text)
    (output/'schema.mjs').write_text('export default '+json.dumps({**schema, 'schemaSha256':schema_hash},separators=(',',':'))+';\n')
    ts=['/** Generated from the pinned headers. Pointer and size arguments use bigint. */',
        'export interface LowLevelFunctions {']
    for f in functions:
        args=[]
        if f['returnKind']=='record': args.append('out: bigint')
        for i,p in enumerate(f['parameters']): args.append(f'arg{i}: {js_type(p["kind"])}')
        ret='void' if f['returnKind']=='record' else js_type(f['returnKind'])
        ts += [f'  {f["name"]}({", ".join(args)}): Promise<{ret}>;']
    ts += ['}']
    (output/'functions.d.ts').write_text('\n'.join(ts)+'\n')
    return schema

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clang',default='clang')
    a=p.parse_args(); s=generate(a.source.resolve(),a.output.resolve(),a.clang)
    print(json.dumps({'functions':len(s['functions']),'records':len(s['records']),
                      'constants':len(s['constants']),'excluded':len(s['excluded'])}))
if __name__=='__main__': main()
