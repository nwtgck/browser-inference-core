"""Exercise public GGML binding generation without configuring a runtime build."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_bindings


def function(name, result, *parameters, variadic=False):
    return {
        'kind': 'FunctionDecl', 'name': name,
        'type': {'qualType': result + ' ('},
        'inner': [
            {'kind': 'ParmVarDecl', 'name': f'arg{i}', 'type': {'qualType': typ}}
            for i, typ in enumerate(parameters)
        ],
        'variadic': variadic,
    }


class GgmlBindings(unittest.TestCase):
    def test_unimplemented_threadpool_query_is_explained_and_not_exported(self):
        source = ROOT / 'vendor/llama.cpp'
        missing = 'ggml_threadpool_get_n_threads'
        ast = {'inner': [
            function(missing, 'int', 'struct ggml_threadpool *'),
            function('ggml_threadpool_pause', 'void', 'struct ggml_threadpool *'),
            function('ggml_threadpool_resume', 'void', 'struct ggml_threadpool *'),
        ]}
        with tempfile.TemporaryDirectory(prefix='lcb-unimplemented-binding-') as temp, \
             patch.object(generate_bindings.subprocess, 'check_output', return_value=json.dumps(ast)):
            output = Path(temp)
            schema = generate_bindings.generate(source, output, 'clang')
            self.assertEqual({entry['name'] for entry in schema['functions']},
                             {'ggml_threadpool_pause', 'ggml_threadpool_resume'})
            self.assertEqual(schema['excluded'], [{
                'name': missing,
                'reason': 'declared in ggml-cpu.h but not implemented by the pinned upstream',
            }])
            for name in ('exports.json', 'jspi-exports.json', 'bindings.cpp', 'functions.d.ts'):
                self.assertNotIn(missing, (output / name).read_text(), name)
            for name in ('ggml_threadpool_pause', 'ggml_threadpool_resume'):
                self.assertIn('_lcb_' + name, json.loads((output / 'exports.json').read_text()))
                self.assertIn('_' + name, json.loads((output / 'exports.json').read_text()))
            if shutil.which('g++'):
                subprocess.run([
                    'g++', '-std=c++17', '-fsyntax-only',
                    '-I' + str(source / 'include'),
                    '-I' + str(source / 'ggml/include'),
                    '-I' + str(source / 'tools/mtmd'), str(output / 'bindings.cpp'),
                ], check=True, capture_output=True, text=True)

    def test_public_ggml_functions_records_and_exports(self):
        source = ROOT / 'vendor/llama.cpp'
        ast = {'inner': [
            {'kind': 'TypedefDecl', 'name': 'ggml_backend_buffer_t',
             'type': {'qualType': 'ggml_backend_buffer_t',
                      'desugaredQualType': 'struct ggml_backend_buffer *'}},
            {'kind': 'TypedefDecl', 'name': 'ggml_backend_buffer_type_t',
             'type': {'qualType': 'ggml_backend_buffer_type_t',
                      'desugaredQualType': 'struct ggml_backend_buffer_type *'}},
            {'kind': 'TypedefDecl', 'name': 'ggml_gallocr_t',
             'type': {'qualType': 'ggml_gallocr_t',
                      'desugaredQualType': 'struct ggml_gallocr *'}},
            {'kind': 'TypedefDecl', 'name': 'ggml_fp16_t',
             'type': {'qualType': 'ggml_fp16_t', 'desugaredQualType': 'unsigned short'}},
            {'kind': 'TypedefDecl', 'name': 'ggml_bf16_t',
             'type': {'qualType': 'ggml_bf16_t',
                      'desugaredQualType': 'struct (unnamed struct at ggml.h:377:13)'}},
            {'kind': 'TypedefDecl', 'name': 'ggml_abort_callback_t',
             'type': {'qualType': 'ggml_abort_callback_t',
                      'desugaredQualType': 'void (*)(const char *)'}},
            function('ggml_used_mem', 'size_t', 'const struct ggml_context *'),
            function('ggml_init', 'struct ggml_context *', 'struct ggml_init_params'),
            function('ggml_add', 'struct ggml_tensor *', 'struct ggml_context *',
                     'struct ggml_tensor *', 'struct ggml_tensor *'),
            function('ggml_fp16_to_fp32', 'float', 'ggml_fp16_t'),
            function('ggml_fp32_to_bf16', 'ggml_bf16_t', 'float'),
            function('ggml_bf16_to_fp32', 'float', 'ggml_bf16_t'),
            function('ggml_set_abort_callback', 'ggml_abort_callback_t',
                     'ggml_abort_callback_t'),
            function('ggml_tallocr_new', 'struct ggml_tallocr', 'ggml_backend_buffer_t'),
            function('ggml_gallocr_get_buffer_size', 'size_t', 'ggml_gallocr_t', 'int'),
            function('ggml_backend_alloc_ctx_tensors_from_buft_size', 'size_t',
                     'struct ggml_context *', 'ggml_backend_buffer_type_t'),
            function('ggml_format_name', 'struct ggml_tensor *', 'struct ggml_tensor *',
                     'const char *', variadic=True),
            {'kind': 'RecordDecl', 'name': 'ggml_init_params', 'tagUsed': 'struct',
             'completeDefinition': True, 'inner': [
                 {'kind': 'FieldDecl', 'name': 'mem_size', 'type': {'qualType': 'size_t'}},
                 {'kind': 'FieldDecl', 'name': 'no_alloc', 'type': {'qualType': 'bool'}},
             ]},
            {'kind': 'RecordDecl', 'name': 'ggml_tallocr', 'tagUsed': 'struct',
             'completeDefinition': True, 'inner': [
                 {'kind': 'FieldDecl', 'name': 'alignment', 'type': {'qualType': 'size_t'}},
             ]},
        ]}
        with tempfile.TemporaryDirectory(prefix='lcb-ggml-bindings-') as temp, \
             patch.object(generate_bindings.subprocess, 'check_output',
                          return_value=json.dumps(ast)):
            output = Path(temp)
            schema = generate_bindings.generate(source, output, 'clang')
            names = {entry['name'] for entry in schema['functions']}
            self.assertEqual(names, {
                'ggml_used_mem', 'ggml_init', 'ggml_add',
                'ggml_fp16_to_fp32', 'ggml_fp32_to_bf16', 'ggml_bf16_to_fp32',
                'ggml_set_abort_callback', 'ggml_tallocr_new',
                'ggml_gallocr_get_buffer_size',
                'ggml_backend_alloc_ctx_tensors_from_buft_size',
            })
            self.assertEqual(schema['excluded'], [
                {'name': 'ggml_format_name', 'reason': 'variadic'},
            ])
            self.assertEqual({entry['name'] for entry in schema['records']},
                             {'ggml_bf16_t', 'ggml_init_params', 'ggml_tallocr'})
            self.assertIn('GGML_MAX_DIMS', schema['constants'])
            self.assertIn('GGML_MEM_ALIGN', schema['constants'])
            exports = json.loads((output / 'exports.json').read_text())
            jspi = json.loads((output / 'jspi-exports.json').read_text())
            for name in names:
                self.assertIn('_lcb_' + name, exports)
                self.assertIn('_' + name, exports)
                self.assertIn('lcb_' + name, jspi)
                self.assertIn(name, jspi)
                self.assertIn(name + '(', (output / 'functions.d.ts').read_text())
            self.assertIn('#include "ggml-alloc.h"', (output / 'headers.c').read_text())
            if shutil.which('g++'):
                subprocess.run([
                    'g++', '-std=c++17', '-fsyntax-only',
                    '-I' + str(source / 'include'),
                    '-I' + str(source / 'ggml/include'),
                    '-I' + str(source / 'tools/mtmd'),
                    str(output / 'bindings.cpp'),
                ], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
