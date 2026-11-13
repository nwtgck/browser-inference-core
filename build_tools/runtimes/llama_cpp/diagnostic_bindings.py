"""Generate an optional typed extension without changing any base ABI files."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

PREFIX = 'lcb_webgpu_q8_diagnostic_'
RECORD_PREFIX = 'ggml_webgpu_q8_diagnostic_'
CONSTANT_PREFIX = 'GGML_WEBGPU_Q8_DIAGNOSTIC_'
RECORDS = {RECORD_PREFIX + 'request_v1', RECORD_PREFIX + 'report_v1'}
SIGNATURES = {
    PREFIX + 'version': ('uint32_t', ('uint64_t',)),
    PREFIX + 'run_v1': ('int32_t', ('uint64_t', 'uint64_t', 'uint64_t', 'uint64_t', 'uint32_t')),
}
HELPERS = ['schema_hash', 'sizeof_record', 'alignof_record', 'offsetof_field', 'sizeof_field', 'constant']
KINDS = {'uint32_t': 'unsigned', 'int32_t': 'signed', 'uint64_t': 'u64'}


def walk(node):
    yield node
    for child in node.get('inner', []):
        yield from walk(child)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value):
        raise ValueError('Unsupported identifier')
    return value


def generate(ast, output, contract, base_exports, base_jspi):
    nodes = list(walk(ast))
    records, functions = {}, {}
    constants = sorted({identifier(n['name']) for n in nodes
                        if n.get('kind') == 'EnumConstantDecl' and n.get('name', '').startswith(CONSTANT_PREFIX)})
    if not constants:
        raise ValueError('No diagnostic constants')
    for n in nodes:
        name = n.get('name', '')
        if n.get('kind') == 'RecordDecl' and n.get('completeDefinition') and name in RECORDS:
            fields = []
            for field in n.get('inner', []):
                if field.get('kind') != 'FieldDecl':
                    continue
                typ = field['type']['qualType']
                if field.get('isBitfield') or typ not in ('uint32_t', 'uint8_t[32]'):
                    raise ValueError('Unsupported diagnostic field: ' + typ)
                fields.append({'id': len(fields), 'name': identifier(field['name']),
                               'cType': typ, 'kind': 'array' if '[' in typ else 'unsigned'})
            records[name] = {'name': name, 'cType': 'struct ' + name, 'fields': fields}
        if n.get('kind') == 'FunctionDecl' and name in SIGNATURES:
            params = [p for p in n.get('inner', []) if p.get('kind') == 'ParmVarDecl']
            ret = n['type']['qualType'].split(' (', 1)[0]
            signature = (ret, tuple(p['type']['qualType'] for p in params))
            if signature != SIGNATURES[name] or n.get('variadic'):
                raise ValueError('Diagnostic bridge signature changed: ' + name)
            functions[name] = {'name': name, 'export': '_' + name, 'returnType': ret,
                               'returnKind': KINDS[ret],
                               'suspension': 'jspi' if name.endswith('run_v1') else 'none',
                               'parameters': [{'name': identifier(p['name']), 'cType': p['type']['qualType'],
                                               'kind': KINDS[p['type']['qualType']]} for p in params]}
    if records.keys() != RECORDS or functions.keys() != SIGNATURES.keys():
        raise ValueError('Missing diagnostic records or functions')
    record_list = [dict(records[name], id=i) for i, name in enumerate(sorted(records))]
    exports = ['_' + name for name in sorted(functions)] + ['_' + PREFIX + name for name in HELPERS]
    if len(set(exports)) != len(exports) or set(exports).intersection(base_exports):
        raise ValueError('Export collision')
    schema = {'abiVersion': 1, 'extensionId': 'webgpu-q8-diagnostic-v1', 'baseAbiVersion': 1,
              'pointerRepresentation': 'bigint-u64', 'versionExport': '_' + PREFIX + 'version',
              'schemaHashExport': '_' + PREFIX + 'schema_hash',
              'recordQueryNamespace': '_' + PREFIX,
              'functions': [functions[n] for n in sorted(functions)],
              'records': record_list, 'constants': constants}
    canonical = json.dumps(schema, indent=2) + '\n'
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    cpp = ['// Generated extension metadata; not part of the base schema.',
           '#include ' + json.dumps(contract.name), '#include <stddef.h>', '#include <stdint.h>', 'extern "C" {',
           'uint64_t ' + PREFIX + 'schema_hash(void) { return (uint64_t)(uintptr_t)"' + fingerprint + '"; }']
    for name, expr in [('sizeof_record', 'sizeof({type})'), ('alignof_record', 'alignof({type})')]:
        cpp.append('uint64_t ' + PREFIX + name + '(uint32_t record) { switch(record) {')
        for r in record_list:
            cpp.append('case ' + str(r['id']) + ': return ' + expr.format(type=r['cType']) + ';')
        cpp.append('default: return UINT64_MAX; } }')
    for name, expr in [('offsetof_field', 'offsetof({type}, {field})'), ('sizeof_field', 'sizeof((({type} *)0)->{field})')]:
        cpp.append('uint64_t ' + PREFIX + name + '(uint32_t record, uint32_t field) { switch(record) {')
        for r in record_list:
            cpp.append('case ' + str(r['id']) + ': switch(field) {')
            for field in r['fields']:
                cpp.append('case ' + str(field['id']) + ': return ' + expr.format(type=r['cType'], field=field['name']) + ';')
            cpp.append('default: return UINT64_MAX; }')
        cpp.append('default: return UINT64_MAX; } }')
    cpp.append('int32_t ' + PREFIX + 'constant(uint32_t id) { switch(id) {')
    for i, name in enumerate(constants):
        cpp.append('case ' + str(i) + ': return (int32_t)' + name + ';')
    cpp.append('default: return INT32_MIN; } }')
    cpp.append('}')
    jspi = base_jspi + [PREFIX + 'run_v1']
    if len(set(jspi)) != len(jspi):
        raise ValueError('JSPI export collision')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'q8-diagnostic-metadata.cpp').write_text('\n'.join(cpp) + '\n')
    (output / 'q8-diagnostic-v1.schema.json').write_text(json.dumps(dict(schema, schemaSha256=fingerprint), indent=2) + '\n')
    (output / 'q8-diagnostic-exports.json').write_text(json.dumps(base_exports + exports, indent=2) + '\n')
    (output / 'q8-diagnostic-jspi-exports.json').write_text(json.dumps(jspi, indent=2) + '\n')
    types = ['/** Generated optional extension; no private native function pointers. */',
             'export interface Q8DiagnosticFunctions {',
             '  version(backend: bigint): number;',
             '  runV1(backend: bigint, dst: bigint, request: bigint, report: bigint, reportCapacity: number): Promise<number>;',
             '}']
    (output / 'q8-diagnostic-v1.functions.d.ts').write_text('\n'.join(types) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ast-json', type=Path)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--clang', default='clang')
    p.add_argument('--contract-header', type=Path, required=True)
    p.add_argument('--base-exports', type=Path, required=True)
    p.add_argument('--base-jspi', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ast = (json.loads(a.ast_json.read_text()) if a.ast_json else json.loads(subprocess.check_output(
        [a.clang, '-x', 'c', '-std=c11', '-fsyntax-only', '-I' + str(a.source / 'ggml/include'),
         '-Xclang', '-ast-dump=json', str(a.contract_header)], text=True)))
    generate(ast, a.output, a.contract_header,
             json.loads(a.base_exports.read_text()), json.loads(a.base_jspi.read_text()))


if __name__ == '__main__':
    main()
