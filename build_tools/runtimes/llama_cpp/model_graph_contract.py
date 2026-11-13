"""Check the optional model-graph extension without changing the base ABI.

Legacy packages may omit the extension entirely. Once a build records it,
missing or inconsistent types/schema/provenance must fail packaging, not become
an apparently compatible runtime. This does not certify actual Wasm execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_tools.runtimes.llama_cpp.generate_model_graph import PREFIX, SIGNATURES, TYPES

SCHEMA_FILE = 'model-graph-v1.schema.json'
TYPES_FILE = 'model-graph-v1.functions.d.ts'


def declaration(schema: dict) -> str:
    lines = ['/** Generated from native/model-graph/model-graph.h; synchronous optional extension. */',
             'export interface ModelGraphNativeExports {']
    for function in schema['functions']:
        args = ', '.join(p['name'] + ': ' + TYPES[p['cType']] for p in function['parameters'])
        lines.append('  ' + function['export'] + '(' + args + '): ' + TYPES[function['returnType']] + ';')
    return '\n'.join(lines + ['}']) + '\n'


def validate(runtime: Path, provenance: dict) -> str | None:
    record = provenance.get('modelGraphBoundary')
    schema_path, types_path = runtime / SCHEMA_FILE, runtime / TYPES_FILE
    present = schema_path.exists() or types_path.exists()
    if record is None and not present:
        return None
    if not isinstance(record, dict) or not schema_path.is_file() or not types_path.is_file():
        raise ValueError('Incomplete model graph boundary artifacts/provenance')
    if schema_path.is_symlink() or types_path.is_symlink():
        raise ValueError('Linked model graph boundary artifact')
    schema = json.loads(schema_path.read_text())
    if not isinstance(schema, dict) or set(schema) != {
        'extensionId', 'version', 'pointerRepresentation', 'suspension', 'functions', 'schemaSha256'
    }:
        raise ValueError('Invalid model graph boundary schema')
    if (schema['extensionId'] != 'model-graph-v1' or schema['version'] != 1 or
            schema['pointerRepresentation'] != 'bigint-u64' or schema['suspension'] != 'none'):
        raise ValueError('Unsupported model graph boundary contract')
    functions = schema['functions']
    if not isinstance(functions, list) or len(functions) != len(SIGNATURES):
        raise ValueError('Incomplete model graph boundary function inventory')
    import re
    names = []
    for function in functions:
        if not isinstance(function, dict) or set(function) != {'name', 'export', 'returnType', 'parameters'}:
            raise ValueError('Invalid model graph boundary function')
        name = function['name']
        if not isinstance(name, str) or not name.startswith(PREFIX):
            raise ValueError('Invalid model graph boundary function name')
        short = name[len(PREFIX):]
        params = function['parameters']
        if not isinstance(params, list) or any(not isinstance(p, dict) or set(p) != {'name', 'cType'} or
                not isinstance(p['name'], str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', p['name']) for p in params):
            raise ValueError('Invalid model graph boundary parameters')
        if len({p['name'] for p in params}) != len(params):
            raise ValueError('Duplicate model graph parameter')
        if short not in SIGNATURES or (function['returnType'], tuple(p['cType'] for p in params)) != SIGNATURES[short]:
            raise ValueError('Changed model graph boundary signature')
        if function['export'] != '_' + name:
            raise ValueError('Changed model graph boundary export')
        names.append(name)
    if names != sorted(PREFIX + name for name in SIGNATURES):
        raise ValueError('Duplicate, missing or reordered model graph function')
    # Match the generator's canonical order, independent of input JSON ordering.
    canonical = {key: schema[key] for key in ('extensionId', 'version', 'pointerRepresentation', 'suspension', 'functions')}
    fingerprint = hashlib.sha256((json.dumps(canonical, indent=2) + '\n').encode()).hexdigest()
    if schema['schemaSha256'] != fingerprint or record.get('schemaSha256') != fingerprint:
        raise ValueError('Model graph boundary fingerprint mismatch')
    if types_path.read_text() != declaration(schema):
        raise ValueError('Model graph boundary TypeScript declarations differ from schema')
    return fingerprint
