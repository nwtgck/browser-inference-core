import importlib.util
from pathlib import Path
import tempfile
import shutil
import subprocess
import json
import re
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sd_generate_bindings', ROOT/'scripts/generate_bindings.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

class Bindings(unittest.TestCase):
    def test_selected_names_and_layouts_exist_in_pinned_public_headers(self):
        include = ROOT/'vendor/ggml-webgpu-source/ggml/include'
        for header, selection in generator.GGML_HEADERS.items():
            contents = (include/header).read_text()
            for name in selection.split():
                self.assertRegex(contents, r'\b'+re.escape(name)+r'\s*\(')
            for name in generator.GGML_TAG_RECORDS.get(header, ()):
                self.assertRegex(contents, r'\bstruct\s+'+re.escape(name)+r'\s*\{')
        all_headers = '\n'.join((include/header).read_text() for header in generator.GGML_HEADERS)
        for name in generator.GGML_MACRO_CONSTANTS:
            self.assertRegex(all_headers, r'(?m)^\s*#\s*define\s+'+re.escape(name)+r'\b')

    def test_ggml_selection_is_complete_and_exclusions_are_explained(self):
        declarations = [
            {'kind': 'FunctionDecl', 'name': 'ggml_version'},
            {'kind': 'FunctionDecl', 'name': 'ggml_abort', 'variadic': True},
            {'kind': 'FunctionDecl', 'name': 'ggml_unused'},
        ]
        selected, excluded, coverage = generator.select_ggml_functions(
            'ggml.h', declarations, {'ggml_version'})
        self.assertEqual([node['name'] for node in selected], ['ggml_version'])
        self.assertEqual(coverage, {'available': 3, 'selected': 1, 'excluded': 2})
        self.assertEqual({item['name']: item['reason'] for item in excluded}, {
            'ggml_abort': 'variadic', 'ggml_unused': 'not-selected',
        })
        with self.assertRaisesRegex(ValueError, 'selected public functions missing'):
            generator.select_ggml_functions('ggml.h', declarations, {'ggml_missing'})
        with self.assertRaisesRegex(ValueError, 'selected function is variadic'):
            generator.select_ggml_functions('ggml.h', declarations, {'ggml_abort'})

    def test_ggml_tagged_and_anonymous_records_and_signatures_without_clang(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sd = root/'sd'; ggml = root/'ggml'; output = root/'out'
            (sd/'include').mkdir(parents=True)
            (ggml/'include').mkdir(parents=True)
            (sd/'include/stable-diffusion.h').write_text('')
            for header in generator.GGML_HEADERS:
                (ggml/'include'/header).write_text('')

            def function(name, unit, result='int', params=()):
                return {'kind': 'FunctionDecl', 'name': name,
                        'loc': {'includedFrom': {'file': str(unit)}},
                        'type': {'qualType': result+' (void)'},
                        'inner': [{'kind': 'ParmVarDecl', 'name': 'p'+str(i),
                                   'type': {'qualType': typ}} for i, typ in enumerate(params)]}

            def record(name, unit, fields):
                return {'kind': 'RecordDecl', 'id': name, 'name': name,
                        'completeDefinition': True,
                        'loc': {'includedFrom': {'file': str(unit)}},
                        'inner': [{'kind': 'FieldDecl', 'name': field,
                                   'type': {'qualType': typ}} for field, typ in fields]}

            def ast_for(unit):
                header = unit.read_text().split('"')[1]
                if header == 'stable-diffusion.h':
                    return {'inner': [function('sd_version', unit, 'const char *'),
                            record('anonymous-bf16', unit, [('sd_field', 'uint16_t')]),
                            {'kind': 'TypedefDecl', 'name': 'sd_example_t',
                             'loc': {'includedFrom': {'file': str(unit)}},
                             'type': {'qualType': 'struct sd_example_t'},
                             'inner': [{'ownedTagDecl': {'id': 'anonymous-bf16'}}]},
                            {'kind': 'TypedefDecl', 'name': 'uint16_t',
                             'type': {'qualType': 'unsigned short'}}]}
                functions = [function(name, unit) for name in generator.GGML_HEADERS[header].split()]
                overrides = {
                    'ggml_fp32_to_bf16': ('ggml_bf16_t', ('float',)),
                    'ggml_bf16_to_fp32': ('float', ('ggml_bf16_t',)),
                    'ggml_set_abort_callback': ('ggml_abort_callback_t', ('ggml_abort_callback_t',)),
                    'ggml_backend_dev_get_props': ('void', ('ggml_backend_dev_t', 'struct ggml_backend_dev_props *')),
                    'ggml_get_type_traits': ('const struct ggml_type_traits *', ('enum ggml_type',)),
                    'gguf_init_from_buffer': ('struct gguf_context *', ('const void *', 'size_t', 'struct gguf_init_params')),
                    'gguf_get_val_u64': ('uint64_t', ('const struct gguf_context *', 'int64_t')),
                }
                functions = [function(n['name'], unit, *overrides[n['name']]) if n['name'] in overrides else n
                             for n in functions]
                functions.append(function('not_selected', unit))
                records = [record(name, unit, [('caps', 'struct ggml_backend_dev_caps')]
                                  if name == 'ggml_backend_dev_props' else
                                  [('to_float', 'ggml_to_float_t')]
                                  if name == 'ggml_type_traits' else [('bits', 'uint16_t')])
                           for name in generator.GGML_TAG_RECORDS.get(header, ())]
                records.append({'kind': 'TypedefDecl', 'name': 'uint16_t',
                                'type': {'qualType': 'unsigned short'}})
                if header == 'ggml.h':
                    bf16_record = record('anonymous-bf16', unit, [('bits', 'uint16_t')])
                    records.extend([bf16_record,
                                    {'kind': 'TypedefDecl', 'name': 'ggml_bf16_t',
                                     'loc': {'includedFrom': {'file': str(unit)}},
                                     'type': {'qualType': 'struct ggml_bf16_t'},
                                     'inner': [{'ownedTagDecl': {'id': 'anonymous-bf16'}}]},
                                    {'kind': 'TypedefDecl', 'name': 'ggml_abort_callback_t',
                                     'type': {'qualType': 'void (*)(const char *)'}},
                                    {'kind': 'TypedefDecl', 'name': 'ggml_to_float_t',
                                     'type': {'qualType': 'void (*)(const void *, float *, int64_t)'}}])
                if header == 'ggml-backend.h':
                    records.append({'kind': 'TypedefDecl', 'name': 'ggml_backend_dev_t',
                                    'type': {'qualType': 'struct ggml_backend_device *'}})
                if header == 'gguf.h':
                    # A separate Clang AST process may reuse a node ID.
                    records.append(record('anonymous-bf16', unit, [('wrong', 'uint16_t')]))
                return {'inner': functions+records}

            def fake_ast(command, text):
                return json.dumps(ast_for(Path(command[-1])))

            with patch.object(generator.subprocess, 'check_output', side_effect=fake_ast):
                schema = generator.generate(sd, output, 'unused-clang', ggml)
            self.assertEqual(schema['ggmlCoverage']['ggml.h']['selected'],
                             len(generator.GGML_HEADERS['ggml.h'].split()))
            self.assertTrue(all(x['reason'] == 'not-selected' for x in schema['excluded']))
            functions = {item['name']: item for item in schema['functions']}
            self.assertEqual(functions['ggml_fp32_to_bf16']['returnKind'], 'record')
            self.assertEqual(functions['ggml_bf16_to_fp32']['parameters'][0]['kind'], 'record')
            self.assertEqual(functions['ggml_set_abort_callback']['returnKind'], 'pointer')
            self.assertEqual(functions['gguf_init_from_buffer']['parameters'][2]['kind'], 'record')
            self.assertEqual(functions['gguf_get_val_u64']['returnKind'], 'u64')
            records = {record['name']: record for record in schema['records']}
            self.assertEqual(records['ggml_bf16_t']['cType'], 'ggml_bf16_t')
            self.assertEqual(records['ggml_bf16_t']['fields'][0]['name'], 'bits')
            self.assertEqual(records['sd_example_t']['fields'][0]['name'], 'sd_field')
            self.assertEqual(records['ggml_backend_dev_props']['cType'], 'struct ggml_backend_dev_props')
            self.assertEqual(records['ggml_backend_dev_props']['fields'][0]['kind'], 'record')
            self.assertEqual(records['ggml_type_traits']['fields'][0]['kind'], 'pointer')
            self.assertEqual(records['gguf_init_params']['cType'], 'struct gguf_init_params')
            self.assertIn('GGUF_VERSION', schema['constants'])
            self.assertIn('_sdc_ggml_backend_dev_get_props', json.loads((output/'exports.json').read_text()))
            self.assertIn('ggml_set_abort_callback', (output/'functions.d.ts').read_text())

    @unittest.skipUnless(shutil.which('clang') and shutil.which('c++'), 'native compilers required')
    def test_anonymous_records_callbacks_and_full_width_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'include').mkdir()
            (root/'include/stable-diffusion.h').write_text('''#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
typedef struct { int64_t seed; const char *path; bool enabled; size_t count; } params_t;
typedef void (*progress_t)(int, int, float, void *);
void configure(params_t *p);
void set_progress(progress_t cb, void *data);
''')
            subprocess.run(['python3',str(ROOT/'scripts/generate_bindings.py'),'--source',str(root),'--output',str(root/'out')],check=True,capture_output=True)
            schema = json.loads((root/'out/schema.json').read_text())
            self.assertEqual([r['name'] for r in schema['records']], ['params_t'])
            self.assertEqual(schema['abiVersion'],2)
            self.assertEqual(len(schema['functions']),2)
            self.assertIn('uint64_t', (root/'out/bindings.cpp').read_text())
            subprocess.run(['c++','-std=c++17','-fsyntax-only','-I'+str(root/'include'),str(root/'out/bindings.cpp')],check=True,capture_output=True)
if __name__ == '__main__': unittest.main()
