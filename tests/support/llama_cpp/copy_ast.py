"""Type-checked source selections for native copy/queue behavioral models.

The model executes selected functions, not a complete backend. Every selection
comes from a successful full translation-unit check of the current prepared
source. Frozen composer guards use their own historical fixtures instead.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from tests.support.llama_cpp.dispatch_ast import Node, _objects, _references, _source_range, _walk


class Declarations:
    """One checked translation unit, never shared between inputs or options."""

    def __init__(self, source: Path, arguments: list[str], names: set[str], *,
                 headers: tuple[Path, ...] = (), name_filter: str = 'ggml_'):
        self.source = source.resolve()
        self.inputs = {p.resolve(): p.read_bytes() for p in (source, *headers)}
        compiler = shutil.which(os.environ.get('CLANGXX', 'clang++'))
        if compiler is None:
            raise RuntimeError('Current copy checks require Clang (CLANGXX or clang++)')
        command = [compiler, *arguments, '-fsyntax-only', '-Xclang', '-ast-dump=json',
                   '-Xclang', '-ast-dump-filter=' + name_filter, str(self.source)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as error:
            error.add_note(error.stderr)
            raise
        self.assert_unchanged()
        self.nodes: dict[str, list[Node]] = {name: [] for name in names}
        # Filtered Clang output is a sequence of JSON roots. Retain only requested
        # declarations, not the much larger bodies of unrelated compute kernels.
        for node in _objects(result.stdout):
            if node.get('name') in self.nodes:
                self.nodes[node['name']].append(node)

    def assert_unchanged(self) -> None:
        if any(p.read_bytes() != data for p, data in self.inputs.items()):
            raise ValueError('Current copy source changed while selecting declarations')

    def definition(self, name: str, kind: str = 'FunctionDecl', *, owner: Path | None = None) -> Node:
        owner = (owner or self.source).resolve()
        matches = []
        for node in self.nodes.get(name, []):
            loc = node.get('loc', {})
            if ('spellingLoc' in loc or 'expansionLoc' in loc
                    or Path(loc.get('file', '')).resolve() != owner or node.get('kind') != kind):
                continue
            if kind == 'FunctionDecl' and not any(c.get('kind') == 'CompoundStmt' for c in node.get('inner', [])):
                continue
            if kind in ('RecordDecl', 'CXXRecordDecl') and not node.get('completeDefinition'):
                continue
            matches.append(node)
        if len(matches) != 1:
            raise ValueError(f'Expected one current {kind} definition of {name}; got {len(matches)}')
        return matches[0]

    def text(self, node: Node, *, owner: Path | None = None) -> str:
        self.assert_unchanged()
        owner = (owner or self.source).resolve()
        return _source_range(node, owner, self.inputs[owner])

    def function(self, name: str) -> str:
        return self.text(self.definition(name)) + '\n'

    def registered_function(self, variable: str, record: str, field: str, header: Path) -> str:
        """Pair a semantic aggregate initializer with its actual record fields."""
        declaration = self.definition(variable, 'VarDecl')
        record_node = self.definition(record, 'CXXRecordDecl', owner=header)
        fields = [n for n in record_node.get('inner', []) if n.get('kind') == 'FieldDecl']
        initializers = [n for n in declaration.get('inner', []) if n.get('kind') == 'InitListExpr']
        if len(initializers) != 1:
            raise ValueError('Expected a typed interface aggregate initializer: ' + variable)
        initializer = initializers[0]
        # Clang desugars aliases; strip only C++ cv/elaborated-type words, not a
        # substring that could accidentally accept a different record type.
        actual_type = initializer['type'].get('desugaredQualType', initializer['type']['qualType'])
        if [word for word in actual_type.split() if word not in ('const', 'volatile', 'struct', 'class')] != [record]:
            raise ValueError('Interface initializer has an unexpected record type')
        entries = initializer.get('inner', [])
        indexes = [i for i, node in enumerate(fields) if node.get('name') == field]
        if len(indexes) != 1 or len(entries) != len(fields):
            raise ValueError('Interface initializer does not match its declared fields')
        expression = entries[indexes[0]]
        while (expression.get('kind') in ('ImplicitCastExpr', 'ParenExpr')
               or expression.get('kind') == 'UnaryOperator' and expression.get('opcode') == '&'):
            children = expression.get('inner', [])
            if len(children) != 1:
                raise ValueError('Ambiguous interface function initializer')
            expression = children[0]
        reference = expression.get('referencedDecl', {})
        if expression.get('kind') != 'DeclRefExpr' or reference.get('kind') != 'FunctionDecl':
            raise ValueError('Interface field must refer directly to a current function')
        name = reference['name']
        function = self.definition(name)
        if reference.get('id') not in (function.get('id'), function.get('previousDecl')):
            raise ValueError('Interface references a different function declaration')
        self.text(expression)  # reject macro/foreign ranges, even if the type matches
        return name

    def constructor_discovery(self, header: Path | None = None) -> str:
        constructor = self.definition('ggml_backend_sched_new')
        parameters = [n for n in constructor.get('inner', [])
                      if n.get('kind') == 'ParmVarDecl' and n.get('name') == 'n_backends']
        record = self.definition('ggml_backend_sched', 'CXXRecordDecl', owner=header)
        fields = [n for n in record.get('inner', [])
                  if n.get('kind') == 'FieldDecl' and n.get('name') == 'host_input_snapshot']
        if len(parameters) != 1 or len(fields) != 1:
            raise ValueError('Scheduler discovery contract declarations are ambiguous')
        body = next(n for n in constructor['inner'] if n.get('kind') == 'CompoundStmt')
        loops = [n for n in body.get('inner', []) if n.get('kind') == 'ForStmt'
                 and parameters[0]['id'] in _references(n)
                 and any(c.get('kind') == 'MemberExpr' and c.get('referencedMemberDecl') == fields[0]['id']
                         for c in _walk(n))]
        if len(loops) != 1:
            raise ValueError(f'Expected one scheduler backend discovery loop; got {len(loops)}')
        return self.text(loops[0])
