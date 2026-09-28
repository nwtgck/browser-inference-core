#!/usr/bin/env python3
"""Generate thin bindings from the pinned SD and selected public ggml C headers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ABI_VERSION = 2
EXTRA_CONSTANTS = ['EINVAL', 'EIO', 'EOVERFLOW', 'EROFS', 'ENOENT',
                   'SEEK_SET', 'SEEK_CUR', 'SEEK_END']
GGML_HEADERS = {
    'ggml.h': '''
        ggml_version ggml_commit ggml_status_to_string
        ggml_fp16_to_fp32 ggml_fp32_to_fp16
        ggml_bf16_to_fp32 ggml_fp32_to_bf16 ggml_set_abort_callback
        ggml_nelements ggml_nrows ggml_nbytes ggml_nbytes_pad
        ggml_blck_size ggml_type_size ggml_row_size ggml_type_name
        ggml_op_name ggml_op_symbol ggml_unary_op_name ggml_glu_op_name
        ggml_op_desc ggml_element_size ggml_is_quantized
        ggml_ftype_to_ggml_type ggml_tensor_overhead ggml_validate_row_data
        ggml_is_transposed ggml_is_permuted ggml_is_empty ggml_is_view
        ggml_is_scalar ggml_is_vector ggml_is_matrix ggml_is_3d
        ggml_n_dims ggml_is_contiguous ggml_is_contiguous_0
        ggml_is_contiguous_1 ggml_is_contiguous_2
        ggml_is_contiguous_to_1 ggml_is_contiguous_to_2
        ggml_is_contiguous_to_3 ggml_are_same_shape ggml_are_same_stride
        ggml_get_name ggml_get_data ggml_get_data_f32
        ggml_quantize_requires_imatrix ggml_get_type_traits
    ''',
    'ggml-backend.h': '''
        ggml_backend_buft_name ggml_backend_buft_get_alignment
        ggml_backend_buft_get_max_size ggml_backend_buft_get_alloc_size
        ggml_backend_buft_is_host ggml_backend_buft_get_device
        ggml_backend_buffer_name ggml_backend_buffer_get_base
        ggml_backend_buffer_get_size ggml_backend_buffer_get_alignment
        ggml_backend_buffer_get_max_size ggml_backend_buffer_get_alloc_size
        ggml_backend_buffer_is_host ggml_backend_buffer_get_usage
        ggml_backend_buffer_get_type ggml_backend_name
        ggml_backend_free ggml_backend_alloc_buffer
        ggml_backend_buffer_free
        ggml_backend_get_default_buffer_type ggml_backend_get_alignment
        ggml_backend_get_max_size ggml_backend_get_device
        ggml_backend_dev_name ggml_backend_dev_description
        ggml_backend_dev_memory ggml_backend_dev_type
        ggml_backend_dev_get_props ggml_backend_dev_backend_reg
        ggml_backend_dev_init
        ggml_backend_dev_buffer_type ggml_backend_dev_host_buffer_type
        ggml_backend_dev_supports_op ggml_backend_dev_supports_buft
        ggml_backend_dev_offload_op
        ggml_backend_reg_name ggml_backend_reg_dev_count
        ggml_backend_reg_dev_get ggml_backend_reg_count
        ggml_backend_reg_get ggml_backend_reg_by_name
        ggml_backend_dev_count ggml_backend_dev_get
        ggml_backend_dev_by_name ggml_backend_dev_by_type
        ggml_backend_init_by_name ggml_backend_init_by_type
        ggml_backend_init_best
        ggml_backend_load_all
    ''',
    'gguf.h': '''
        gguf_init_empty gguf_init_from_file gguf_init_from_buffer
        gguf_init_from_callback gguf_free gguf_type_name
        gguf_get_version gguf_get_alignment gguf_get_data_offset
        gguf_get_n_kv gguf_find_key gguf_get_key gguf_get_kv_type
        gguf_get_arr_type gguf_get_val_u8 gguf_get_val_i8
        gguf_get_val_u16 gguf_get_val_i16 gguf_get_val_u32
        gguf_get_val_i32 gguf_get_val_f32 gguf_get_val_u64
        gguf_get_val_i64 gguf_get_val_f64 gguf_get_val_bool
        gguf_get_val_str gguf_get_val_data gguf_get_arr_n
        gguf_get_arr_data gguf_get_arr_str gguf_get_n_tensors
        gguf_find_tensor gguf_get_tensor_offset gguf_get_tensor_name
        gguf_get_tensor_ne gguf_get_tensor_type gguf_get_tensor_size
        gguf_get_meta_size gguf_get_meta_data
    ''',
    'ggml-cpu.h': '''
        ggml_is_numa ggml_cpu_has_sse3 ggml_cpu_has_ssse3
        ggml_cpu_has_avx ggml_cpu_has_avx_vnni ggml_cpu_has_avx2
        ggml_cpu_has_bmi2 ggml_cpu_has_f16c ggml_cpu_has_fma
        ggml_cpu_has_avx512 ggml_cpu_has_avx512_vbmi
        ggml_cpu_has_avx512_vnni ggml_cpu_has_avx512_bf16
        ggml_cpu_has_amx_int8 ggml_cpu_has_neon ggml_cpu_has_arm_fma
        ggml_cpu_has_fp16_va ggml_cpu_has_dotprod
        ggml_cpu_has_matmul_int8 ggml_cpu_has_sve ggml_cpu_get_sve_cnt
        ggml_cpu_has_sme ggml_cpu_has_sme2 ggml_cpu_has_riscv_v
        ggml_cpu_get_rvv_vlen ggml_cpu_has_vsx ggml_cpu_has_vxe
        ggml_cpu_has_wasm_simd ggml_cpu_has_llamafile
    ''',
}
GGML_TAG_RECORDS = {
    'ggml.h': {'ggml_tensor', 'ggml_type_traits'},
    'ggml-backend.h': {'ggml_backend_dev_caps', 'ggml_backend_dev_props'},
    'gguf.h': {'gguf_init_params'},
}
GGML_MACRO_CONSTANTS = (
    'GGML_MAX_DIMS', 'GGML_MAX_SRC', 'GGML_DEFAULT_GRAPH_SIZE',
    'GGUF_VERSION', 'GGUF_DEFAULT_ALIGNMENT',
)
GGML_EXCLUSION_REASONS = {
    'ggml_abort': 'variadic fatal-abort API',
    'ggml_fopen': 'FILE* lifetime is outside the browser core',
    'gguf_init_from_file_ptr': 'FILE* ownership is outside the browser core',
}

def direct_declarations(ast: dict, unit: Path) -> list[dict]:
    return [n for n in ast['inner']
            if n.get('loc', {}).get('includedFrom', {}).get('file') == str(unit)]

def select_ggml_functions(header: str, declarations: list[dict], names: set[str]) -> tuple[list[dict], list[dict], dict]:
    """Make every omitted public declaration visible, and reject stale selections."""
    available = {n['name']: n for n in declarations if n.get('kind') == 'FunctionDecl' and n.get('name')}
    missing = names - available.keys()
    if missing:
        raise ValueError(f'{header}: selected public functions missing: {sorted(missing)}')
    selected, excluded = [], []
    for name, node in sorted(available.items()):
        deprecated = any(x.get('kind') == 'DeprecatedAttr' for x in node.get('inner', []))
        reason = ('deprecated' if deprecated else 'variadic' if node.get('variadic') else
                  GGML_EXCLUSION_REASONS.get(name, 'not-selected'))
        if name in names:
            if deprecated or node.get('variadic'):
                raise ValueError(f'{header}: selected function is {reason}: {name}')
            selected.append(node)
        else:
            excluded.append({'name': name, 'header': header, 'reason': reason})
    return selected, excluded, {'available': len(available), 'selected': len(selected), 'excluded': len(excluded)}

def walk(node):
    yield node
    for child in node.get('inner', []):
        yield from walk(child)

def generate(source: Path, output: Path, compiler: str, ggml_source: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    unit = output / 'headers.c'
    unit.write_text('#include "stable-diffusion.h"\n')
    command = [compiler, '-x', 'c', '-std=c11', '-fsyntax-only',
               '-I'+str(source/'include'), '-Xclang', '-ast-dump=json', str(unit)]
    ast = json.loads(subprocess.check_output(command, text=True))
    declarations = direct_declarations(ast, unit)
    declaration_groups = [list(declarations)]
    nodes = list(walk(ast))
    ggml_selected, ggml_excluded, coverage = [], [], {}
    ggml_declarations = {}
    if ggml_source is not None:
        if not (ggml_source/'include/ggml.h').is_file():
            raise ValueError('Missing pinned ggml public headers')
        for header, selection in GGML_HEADERS.items():
            header_unit = output / (header.replace('.', '-') + '.c')
            header_unit.write_text(f'#include "{header}"\n')
            header_ast = json.loads(subprocess.check_output([
                compiler, '-x', 'c', '-std=c11', '-fsyntax-only',
                '-I'+str(ggml_source/'include'), '-Xclang', '-ast-dump=json', str(header_unit)
            ], text=True))
            direct = direct_declarations(header_ast, header_unit)
            ggml_declarations[header] = direct
            declaration_groups.append(direct)
            chosen, omitted, count = select_ggml_functions(header, direct, set(selection.split()))
            ggml_selected.extend(chosen)
            ggml_excluded.extend(omitted)
            coverage[header] = count
            nodes.extend(walk(header_ast))
            declarations.extend(direct)
    aliases = {n['name']: n['type'].get('desugaredQualType', n['type']['qualType'])
               for n in nodes if n.get('kind') == 'TypedefDecl' and 'type' in n}
    named_records = {}
    record_types = {}
    # Clang node IDs are process-local; do not match a typedef in one AST dump
    # against a coincidentally identical record ID from another header unit.
    for group in declaration_groups:
        record_declarations = {n['id']: n for n in group
                               if n.get('kind') == 'RecordDecl' and n.get('completeDefinition')}
        for n in group:
            if n.get('kind') != 'TypedefDecl' or not n['type']['qualType'].startswith(('struct ', 'union ')): continue
            for child in walk(n):
                owned = child.get('ownedTagDecl', {})
                if owned.get('id') in record_declarations:
                    named_records[n['name']] = record_declarations[owned['id']]
                    record_types[n['name']] = n['name']
    for header, names in GGML_TAG_RECORDS.items():
        if ggml_source is None: break
        records = {n.get('name'): n for n in ggml_declarations[header]
                   if n.get('kind') == 'RecordDecl' and n.get('completeDefinition')}
        missing = names - records.keys()
        if missing:
            raise ValueError(f'{header}: selected public records missing: {sorted(missing)}')
        for name in names:
            named_records[name] = records[name]
            record_types[name] = 'struct ' + name
    for name in named_records: aliases[name] = 'struct ' + name
    def resolve(typ):
        old = None
        while typ != old:
            old = typ
            typ = aliases.get(typ, typ)
        return typ
    def classify(typ):
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
    selected_ggml_ids = {id(n) for n in ggml_selected}
    for n in declarations:
        name = n.get('name', '')
        if n.get('kind') != 'FunctionDecl': continue
        if ggml_source is not None and name.startswith(('ggml_', 'gguf_')) and id(n) not in selected_ggml_ids:
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
        functions.append({'name': name, 'export': '_sdc_'+name, 'returnType': result,
                          'returnKind': rk, 'parameters': params})
    functions = sorted({f['name']: f for f in functions}.values(), key=lambda f: f['name'])
    records = []
    for name, n in sorted(named_records.items()):
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
        records.append({'name': name, 'cType': record_types[name], 'fields': fields})
    records = sorted({r['name']: r for r in records}.values(), key=lambda r: r['name'])
    enum_names = sorted({child['name'] for n in declarations if n.get('kind') == 'EnumDecl'
                         for child in n.get('inner', []) if child.get('kind') == 'EnumConstantDecl'})
    constants = sorted(set(enum_names + EXTRA_CONSTANTS +
                           (list(GGML_MACRO_CONSTANTS) if ggml_source is not None else [])))
    cpp = ['// Generated from the public header; do not edit.', '#include "stable-diffusion.h"',
           *([f'#include "{header}"' for header in GGML_HEADERS] if ggml_source is not None else []),
           '#include <stdint.h>', '#include <stddef.h>', '#include <stdio.h>',
           '#include <errno.h>', '#include <stdlib.h>', '#include <stdexcept>',
           'static uintptr_t sdc_checked_pointer(uint64_t p) {',
           '  if (p > UINTPTR_MAX) throw std::range_error("pointer/size exceeds address space");',
           '  return (uintptr_t)p;', '}', 'extern "C" {',
           'uint64_t sdc_malloc(uint64_t size) { return (uint64_t)(uintptr_t)malloc(sdc_checked_pointer(size)); }',
           'void sdc_free(uint64_t ptr) { free((void *)sdc_checked_pointer(ptr)); }',
           'uint32_t sdc_pointer_bytes(void) { return sizeof(void *); }',
           f'uint32_t sdc_abi_version(void) {{ return {ABI_VERSION}; }}',
           'uint32_t sdc_model_io_capabilities(void) { return 3; }']
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
                callargs.append(f'*(({qualified} *)sdc_checked_pointer({arg}))')
            elif kind == 'pointer': callargs.append(f'({typ})sdc_checked_pointer({arg})')
            elif kind == 'u64' and typ in ('size_t', 'uintptr_t'):
                callargs.append(f'({typ})sdc_checked_pointer({arg})')
            else: callargs.append(f'({typ}){arg}')
        call = f'{f["name"]}({", ".join(callargs)})'
        rk=f['returnKind']; rt=f['returnType']
        if rk == 'record': statement=f'*(({rt} *)sdc_checked_pointer(out)) = {call};'; wt='void'
        elif rk == 'void': statement=call+';'; wt='void'
        elif rk == 'pointer': statement=f'return (uint64_t)(uintptr_t){call};'; wt='uint64_t'
        else: wt=wire_type(rk,rt); statement=f'return ({wt}){call};'
        cpp += [f'{wt} sdc_{f["name"]}({", ".join(args) or "void"}) {{',
                *['  '+g for g in guards], '  '+statement, '}']
    for i,r in enumerate(records): r['id']=i
    for name, expr in [('sizeof_record', 'sizeof({ctype})'), ('alignof_record', 'alignof({ctype})')]:
        cpp += [f'uint64_t sdc_{name}(uint32_t record) {{ switch(record) {{']
        cpp += [f'case {r["id"]}: return {expr.format(ctype=r["cType"])};' for r in records]
        cpp += ['default: return UINT64_MAX; } }']
    for name, expr in [('offsetof_field','offsetof({ctype}, {field})'), ('sizeof_field','sizeof((({ctype} *)0)->{field})')]:
        cpp += [f'uint64_t sdc_{name}(uint32_t record, uint32_t field) {{ switch(record) {{']
        for r in records:
            cpp += [f'case {r["id"]}: switch(field) {{']
            for i,p in enumerate(r['fields']):
                p['id']=i
                cpp += [f'case {i}: return {expr.format(ctype=r["cType"], field=p["name"])};']
            cpp += ['default: return UINT64_MAX; }']
        cpp += ['default: return UINT64_MAX; } }']
    cpp += ['int64_t sdc_constant(uint32_t id) { switch(id) {']
    cpp += [f'case {i}: return (int64_t)({name});' for i,name in enumerate(constants)]
    cpp += ['default: throw std::out_of_range("constant id"); } }', '}']
    helper_names = ['malloc','free','pointer_bytes','abi_version','model_io_capabilities','sizeof_record','alignof_record','offsetof_field','sizeof_field','constant','schema_hash']
    exports = ['_sdc_'+n for n in helper_names] + [f['export'] for f in functions]
    # Direct exports remain available for expert callers bound to this exact native ABI.
    exports += ['_'+f['name'] for f in functions] + ['_malloc','_free']
    (output/'exports.json').write_text(json.dumps(exports,indent=2)+'\n')
    (output/'jspi-exports.json').write_text(json.dumps([f['export'][1:] for f in functions] + [f['name'] for f in functions],indent=2)+'\n')
    schema={'abiVersion':ABI_VERSION,'pointerRepresentation':'bigint-u64','functions':functions,
            'records':records,'constants':constants,'excluded':excluded + ggml_excluded,
            'ggmlCoverage':coverage}
    schema_text=json.dumps(schema,indent=2)+'\n'
    schema_hash=hashlib.sha256(schema_text.encode()).hexdigest()
    cpp.insert(-1, f'uint64_t sdc_schema_hash(void) {{ return (uint64_t)(uintptr_t)"{schema_hash}"; }}')
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
    p.add_argument('--ggml-source',type=Path,
                   help='Prepared pinned ggml source; explicitly selected public APIs only')
    p.add_argument('--clang',default='clang')
    a=p.parse_args(); s=generate(a.source.resolve(),a.output.resolve(),a.clang,
                                 a.ggml_source.resolve() if a.ggml_source else None)
    print(json.dumps({'functions':len(s['functions']),'records':len(s['records']),
                      'constants':len(s['constants']),'excluded':len(s['excluded'])}))
if __name__=='__main__': main()
