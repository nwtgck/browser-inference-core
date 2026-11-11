#!/usr/bin/env python3
"""Generate a versioned, typed leaf extension; leave the base ABI unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

PREFIX = 'lcb_model_graph_'
TYPES = {'uint64_t': 'bigint', 'uint32_t': 'number', 'int32_t': 'number', 'void': 'void'}
SIGNATURES = {
    'version': ('uint32_t', ()),
    'schema_hash': ('uint64_t', ()),
    'last_error': ('uint64_t', ()),
    'architecture_known': ('int32_t', ('uint64_t',)),
    'vocab_new': ('uint64_t', ('uint64_t',)),
    'vocab_get': ('uint64_t', ('uint64_t',)),
    'vocab_free': ('void', ('uint64_t',)),
    'sampler_workspace_new': ('uint64_t', ('uint32_t',)),
    'sampler_workspace_free': ('void', ('uint64_t',)),
    'sample': ('int32_t', ('uint64_t', 'uint64_t', 'uint64_t', 'uint32_t')),
}


def walk(node):
    yield node
    for child in node.get('inner', []):
        yield from walk(child)


def generate(header: Path, output: Path, base_exports: Path, clang: str) -> dict:
    ast = json.loads(subprocess.check_output([clang, '-x', 'c', '-std=c11', '-fsyntax-only', '-Xclang', '-ast-dump=json', str(header)], text=True))
    functions = {}
    for node in walk(ast):
        name = node.get('name', '')
        if node.get('kind') != 'FunctionDecl' or not name.startswith(PREFIX):
            continue
        short = name[len(PREFIX):]
        params = [p for p in node.get('inner', []) if p.get('kind') == 'ParmVarDecl']
        ret = node['type']['qualType'].split(' (', 1)[0]
        signature = (ret, tuple(p['type']['qualType'] for p in params))
        if short not in SIGNATURES or signature != SIGNATURES[short] or node.get('variadic'):
            raise ValueError('Review the model graph ABI before changing: ' + name)
        for p in params:
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', p.get('name', '')):
                raise ValueError('Unnamed or invalid model graph parameter')
        functions[name] = {'name': name, 'export': '_' + name, 'returnType': ret,
                           'parameters': [{'name': p['name'], 'cType': p['type']['qualType']} for p in params]}
    if set(functions) != {PREFIX + name for name in SIGNATURES}:
        raise ValueError('Missing model graph boundary function')
    schema = {'extensionId': 'model-graph-v1', 'version': 1, 'pointerRepresentation': 'bigint-u64',
              'suspension': 'none', 'functions': [functions[name] for name in sorted(functions)]}
    fingerprint = hashlib.sha256((json.dumps(schema, indent=2) + '\n').encode()).hexdigest()
    exports = json.loads(base_exports.read_text())
    additions = [f['export'] for f in schema['functions']]
    if set(exports).intersection(additions):
        raise ValueError('Duplicate model graph export')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model-graph-exports.json').write_text(json.dumps(exports + additions, indent=2) + '\n')
    (output / 'model-graph-v1.schema.json').write_text(json.dumps(dict(schema, schemaSha256=fingerprint), indent=2) + '\n')
    (output / 'model-graph-metadata.cpp').write_text('#include "model-graph.h"\nextern "C" uint64_t lcb_model_graph_schema_hash(void) { return (uint64_t)(uintptr_t)"' + fingerprint + '"; }\n')
    types = ['/** Generated from bridge/model-graph.h; synchronous optional extension. */',
             'export interface ModelGraphNativeExports {']
    for f in schema['functions']:
        args = ', '.join(p['name'] + ': ' + TYPES[p['cType']] for p in f['parameters'])
        types.append('  ' + f['export'] + '(' + args + '): ' + TYPES[f['returnType']] + ';')
    types.append('}')
    (output / 'model-graph-v1.functions.d.ts').write_text('\n'.join(types) + '\n')
    return dict(schema, schemaSha256=fingerprint)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--header', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-exports', type=Path, required=True)
    p.add_argument('--clang', default='clang')
    args = p.parse_args()
    generate(args.header, args.output, args.base_exports, args.clang)
