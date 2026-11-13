"""Select the current matvec dispatch block using Clang's semantic source ranges.

The complete provider is type-checked before selecting a block. This remains a
native fragment test, not execution of the full backend or a GPU. Unlike text
markers, comments, brace formatting and string literals cannot select code.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

from build_tools.runtimes.llama_cpp import webgpu_plan
from tests.support.llama_cpp.current_webgpu import dawn_package

Node = dict[str, Any]


def _objects(text: str) -> Iterator[Node]:
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            return
        value, offset = decoder.raw_decode(text, offset)
        if not isinstance(value, dict):
            raise ValueError('Clang AST root must be an object')
        yield value


def _walk(node: Node) -> Iterator[Node]:
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(reversed(current.get('inner', [])))


def _references(node: Node) -> set[str]:
    return {child.get('referencedDecl', {}).get('id', '') for child in _walk(node)
            if child.get('kind') == 'DeclRefExpr'}


def _source_range(node: Node, source: Path, data: bytes) -> str:
    locations = node.get('range', {})
    begin, end = locations.get('begin', {}), locations.get('end', {})
    for location in (begin, end):
        if 'spellingLoc' in location or 'expansionLoc' in location:
            raise ValueError('Dispatch boundaries must not be macro-expanded')
        if 'file' in location and Path(location['file']).resolve() != source.resolve():
            raise ValueError('Dispatch range points outside the current provider')
    try:
        first, last = begin['offset'], end['offset'] + end['tokLen']
    except KeyError as error:
        raise ValueError('Clang did not provide a complete physical source range') from error
    if not 0 <= first < last <= len(data):
        raise ValueError('Clang source range is outside the provider bytes')
    # Clang offsets are bytes, not Unicode code points.
    return data[first:last].decode('utf-8')


def function_ast(source: Path, name: str, arguments: list[str], *,
                 include_template: bool = False) -> tuple[Node, bytes]:
    """Type-check a unit and require one definition in that exact source file.

    Complete consumers can retain the primary template declaration; block
    consumers still receive the FunctionDecl, never an instantiated clone.
    """
    source = source.resolve()
    compiler = shutil.which(os.environ.get('CLANGXX', 'clang++'))
    if compiler is None:
        raise RuntimeError('Current dispatch checks need Clang C++ (CLANGXX or clang++)')
    before = source.read_bytes()
    command = [compiler, *arguments, '-fsyntax-only', '-Xclang', '-ast-dump=json',
               '-Xclang', '-ast-dump-filter=' + name, str(source)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr)
        raise
    if source.read_bytes() != before:
        raise ValueError('Current provider changed while Clang was reading it')
    definitions = []
    for root in _objects(result.stdout):
        if root.get('name') != name:
            continue
        # Profiled code owns a function template plus several instantiations.
        # Select the primary definition, not an arbitrary instantiated body.
        candidates = ([root] if root.get('kind') == 'FunctionDecl' else
                      root.get('inner', []) if root.get('kind') == 'FunctionTemplateDecl' else [])
        for node in candidates:
            children = node.get('inner', [])
            location = node.get('loc', {}).get('file', root.get('loc', {}).get('file', ''))
            if (node.get('kind') == 'FunctionDecl' and node.get('name') == name
                    and any(child.get('kind') == 'CompoundStmt' for child in children)
                    and not any(child.get('kind') == 'TemplateArgument' for child in children)
                    and Path(location).resolve() == source):
                definitions.append(root if include_template and root.get('kind') == 'FunctionTemplateDecl' else node)
    if len(definitions) != 1:
        raise ValueError(f'Expected one current provider definition of {name}; got {len(definitions)}')
    return definitions[0], before


def macro_definition(source: Path, name: str, arguments: list[str]) -> str:
    """Use the compiler's active definition, never a handwritten math substitute.

    Reject redefinition/undefinition rather than guessing which lifetime applies
    to a selected range. Matching preprocessor directives is not parsing C++.
    """
    compiler = shutil.which(os.environ.get('CLANGXX', 'clang++'))
    if not compiler:
        raise RuntimeError('Clang is required for current provider macro definitions')
    before = source.read_bytes()
    result = subprocess.run([compiler, *arguments, '-E', '-dD', str(source)],
                            check=True, capture_output=True, text=True, timeout=120)
    if source.read_bytes() != before:
        raise ValueError('Provider changed while preprocessing current macros')
    pattern = re.compile(r'^#(define|undef)\s+' + re.escape(name) + r'(?=\(|\s|$)')
    definitions = [line for line in result.stdout.splitlines() if pattern.match(line)]
    if len(definitions) != 1 or not definitions[0].startswith('#define '):
        raise ValueError('Expected one unambiguous current macro definition: ' + name)
    return definitions[0]


def dispatch_block(function: Node) -> Node:
    # Names select this explicit contract, but declaration identities decide
    # references. A shadow variable must not masquerade as the dispatch state.
    declarations = {}
    for name in ('use_mat_vec', 'wg_x', 'wg_y'):
        matches = {node['id'] for node in _walk(function)
                   if node.get('kind') in ('VarDecl', 'ParmVarDecl') and node.get('name') == name}
        if len(matches) != 1:
            raise ValueError(f'Expected one dispatch declaration of {name}; got {len(matches)}')
        declarations[name] = matches.pop()
    candidates = []
    for node in _walk(function):
        if node.get('kind') != 'IfStmt' or node.get('hasInit') or node.get('hasVar'):
            continue
        children = node.get('inner', [])
        if len(children) < 2:
            continue
        condition = children[0]
        while condition.get('kind') in ('ImplicitCastExpr', 'ParenExpr') and len(condition.get('inner', [])) == 1:
            condition = condition['inner'][0]
        if (condition.get('kind') == 'DeclRefExpr'
                and condition.get('referencedDecl', {}).get('id') == declarations['use_mat_vec']
                and children[1].get('kind') == 'CompoundStmt'
                and {declarations['wg_x'], declarations['wg_y']} <= _references(children[1])):
            candidates.append(children[1])
    if len(candidates) != 1:
        raise ValueError(f'Expected one matvec workgroup branch; got {len(candidates)}')
    return candidates[0]


def extract_dispatch(source: Path, arguments: list[str], *,
                     matmul_name: str = 'ggml_webgpu_mul_mat') -> tuple[str, str]:
    compute, first = function_ast(source, 'compute_2d_workgroups', arguments)
    matmul, second = function_ast(source, matmul_name, arguments)
    if first != second:
        raise ValueError('Current provider changed between Clang passes')
    return (_source_range(compute, source, first),
            _source_range(dispatch_block(matmul), source, second))


def current_dispatch(upstream: Path, prepared: Path, *, namespace: str = '') -> str:
    if namespace not in ('', 'candidate'):
        raise ValueError('Unsupported recording namespace')
    sdk = dawn_package()
    receipt = json.loads((prepared / webgpu_plan.RECEIPT).read_text())
    arguments = ['-std=c++20', '-D__wasi__', '-pthread', '-DGGML_SCHED_MAX_COPIES=4',
                 '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS']
    if receipt['gpuProfile']:
        arguments += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
    arguments += ['-I' + str(path) for path in
                  (prepared, upstream / webgpu_plan.BASE, upstream / 'ggml/include', upstream / 'ggml/src')]
    arguments += ['-isystem', str(sdk / 'webgpu/include'), '-isystem', str(sdk / 'webgpu_cpp/include')]
    original = (prepared / 'ggml-webgpu.cpp').read_bytes()
    matmul_name = 'ggml_webgpu_mul_mat_impl' if receipt['gpuProfile'] else 'ggml_webgpu_mul_mat'
    compute, branch = extract_dispatch(prepared / 'ggml-webgpu.cpp', arguments, matmul_name=matmul_name)
    definition = macro_definition(prepared / 'ggml-webgpu.cpp', 'CEIL_DIV', arguments)
    if (prepared / 'ggml-webgpu.cpp').read_bytes() != original:
        raise ValueError('Current provider changed between AST and macro passes')
    prefix = namespace + '::' if namespace else ''
    return ('#undef CEIL_DIV\n' + definition + '\n' + compute
            + '\nstatic std::array<uint32_t, 2> production_dispatch('
            + prefix + 'webgpu_pipeline pipeline, ggml_tensor * dst, uint32_t max_wg_per_dim) {\n'
            + ('using ggml_webgpu_mul_mat_vec_shader_decisions = candidate::ggml_webgpu_mul_mat_vec_shader_decisions;\n'
               if namespace else '')
            + 'uint32_t wg_x = 1, wg_y = 1;\n' + branch + '\nreturn {wg_x, wg_y};\n}\n')
