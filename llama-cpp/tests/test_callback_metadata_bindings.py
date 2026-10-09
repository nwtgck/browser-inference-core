"""Check the bounded callback ABI without building a browser runtime.

Clang input is a deliberately small public-declaration fixture. The host test
executes generated wrappers against fake metadata getters, not real GPU work.
"""
import hashlib
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
from test_generate_bindings_ggml import function


def declarations():
    return [
        function('ggml_op_desc', 'const char *', 'const struct ggml_tensor *'),
        function('ggml_backend_buffer_name', 'const char *', 'struct ggml_backend_buffer *'),
        function('ggml_backend_buffer_is_host', 'bool', 'struct ggml_backend_buffer *'),
        function('ggml_backend_dev_count', 'size_t'),
        function('ggml_backend_dev_get', 'struct ggml_backend_device *', 'size_t'),
        function('ggml_backend_dev_name', 'const char *', 'struct ggml_backend_device *'),
        function('ggml_backend_dev_supports_op', 'bool', 'struct ggml_backend_device *',
                 'const struct ggml_tensor *'),
        # An ordinary operation must remain JSPI-only, never grow a sync alias.
        function('ggml_add', 'struct ggml_tensor *', 'struct ggml_context *',
                 'struct ggml_tensor *', 'struct ggml_tensor *'),
    ]


class CallbackMetadataBindings(unittest.TestCase):
    def generate(self, output, nodes=None):
        ast = {'inner': declarations() if nodes is None else nodes}
        with patch.object(generate_bindings.subprocess, 'check_output', return_value=json.dumps(ast)):
            return generate_bindings.generate(ROOT / 'vendor/llama.cpp', output, 'clang')

    def test_aliases_are_explicit_sync_exports_and_normal_apis_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            schema = self.generate(out)
            exports = json.loads((out / 'exports.json').read_text())
            jspi = json.loads((out / 'jspi-exports.json').read_text())
            self.assertEqual(len(exports), len(set(exports)))
            self.assertIn('_lcb_callback_metadata_version', exports)
            self.assertNotIn('lcb_callback_metadata_version', jspi)
            expected = {'_lcb_callback_' + name for name in generate_bindings.CALLBACK_METADATA_GETTERS}
            self.assertEqual({name for name in exports if name.startswith('_lcb_callback_')},
                             expected | {'_lcb_callback_metadata_version'})
            self.assertEqual(schema['callbackMetadata'], {
                'version': 1, 'getters': [
                    {'name': name, 'export': '_lcb_callback_' + name}
                    for name in sorted(generate_bindings.CALLBACK_METADATA_GETTERS)]})
            self.assertFalse(any('callback_' in name for name in jspi))
            for name in generate_bindings.CALLBACK_METADATA_GETTERS:
                self.assertIn('lcb_' + name, jspi)
                self.assertIn(name, jspi)
                self.assertIn(name + '(', (out / 'functions.d.ts').read_text())
            # The new native contract participates in the existing source hash.
            digest = hashlib.sha256((out / 'schema.json').read_bytes()).hexdigest()
            self.assertIn(digest, (out / 'bindings.cpp').read_text())
            self.assertIn(digest, (out / 'schema.mjs').read_text())

    def test_missing_or_deprecated_getters_require_an_explicit_review(self):
        for name in generate_bindings.CALLBACK_METADATA_GETTERS:
            for deprecated in (False, True):
                with self.subTest(name=name, deprecated=deprecated), tempfile.TemporaryDirectory() as temporary:
                    nodes = declarations()
                    if deprecated:
                        next(n for n in nodes if n['name'] == name)['inner'].append({'kind': 'DeprecatedAttr'})
                    else:
                        nodes = [node for node in nodes if node['name'] != name]
                    with self.assertRaisesRegex(ValueError, 'need upstream review'):
                        self.generate(Path(temporary), nodes)
                    self.assertFalse((Path(temporary) / 'exports.json').exists())

    def test_changed_parameter_or_return_contract_is_rejected(self):
        for node in (
            function('ggml_op_desc', 'int', 'const struct ggml_tensor *'),
            function('ggml_backend_dev_get', 'struct ggml_backend_device *', 'size_t', 'size_t'),
            function('ggml_backend_dev_supports_op', 'bool', 'size_t', 'const struct ggml_tensor *'),
        ):
            with self.subTest(node=node['name']), tempfile.TemporaryDirectory() as temporary:
                nodes = [node if old['name'] == node['name'] else old for old in declarations()]
                with self.assertRaisesRegex(ValueError, 'Unexpected callback getter signature'):
                    self.generate(Path(temporary), nodes)

    def test_host_generated_wrappers_preserve_pointer_and_boolean_values(self):
        compiler = shutil.which('g++') or shutil.which('clang++')
        if not compiler:
            self.skipTest('A host C++ compiler is required')
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            self.generate(out)
            cpp = (out / 'bindings.cpp').read_text()
            # Compile the generated public wrappers, not invented hand-written
            # aliases. Record/constant queries are unrelated to this fixture.
            body = cpp[cpp.index('static uintptr_t lcb_checked_pointer'):cpp.index('uint64_t lcb_sizeof_record')]
            fixture = r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdexcept>
#include <assert.h>
#include <string.h>
struct ggml_tensor { int tag; };
struct ggml_backend_buffer { bool host; };
struct ggml_backend_device { int tag; };
struct ggml_context {};
static ggml_backend_device device{17};
const char *ggml_op_desc(const ggml_tensor *) { return "MUL_MAT"; }
const char *ggml_backend_buffer_name(ggml_backend_buffer *) { return "fixture"; }
bool ggml_backend_buffer_is_host(ggml_backend_buffer *b) { return b->host; }
size_t ggml_backend_dev_count() { return 1; }
ggml_backend_device *ggml_backend_dev_get(size_t i) { return i ? nullptr : &device; }
const char *ggml_backend_dev_name(ggml_backend_device *) { return "fixture-device"; }
bool ggml_backend_dev_supports_op(ggml_backend_device *d, const ggml_tensor *t) { return d == &device && t->tag == 4; }
ggml_tensor *ggml_add(ggml_context *, ggml_tensor *a, ggml_tensor *) { return a; }
'''
            main = r'''
}
int main() {
  ggml_tensor tensor{4}; ggml_backend_buffer buffer{true};
  auto t = (uint64_t)(uintptr_t)&tensor, b = (uint64_t)(uintptr_t)&buffer;
  assert(lcb_callback_metadata_version() == 1);
  assert(lcb_callback_ggml_backend_dev_count() == 1);
  auto d = lcb_callback_ggml_backend_dev_get(0);
  assert(d == (uint64_t)(uintptr_t)&device);
  assert(lcb_callback_ggml_backend_dev_get(1) == 0);
  assert(strcmp((const char *)(uintptr_t)lcb_callback_ggml_op_desc(t), "MUL_MAT") == 0);
  assert(strcmp((const char *)(uintptr_t)lcb_callback_ggml_backend_buffer_name(b), "fixture") == 0);
  assert(strcmp((const char *)(uintptr_t)lcb_callback_ggml_backend_dev_name(d), "fixture-device") == 0);
  assert(lcb_callback_ggml_backend_buffer_is_host(b) == 1);
  buffer.host = false; assert(lcb_callback_ggml_backend_buffer_is_host(b) == 0);
  assert(lcb_callback_ggml_backend_dev_supports_op(d, t) == 1);
  tensor.tag = 0; assert(lcb_callback_ggml_backend_dev_supports_op(d, t) == 0);
}
'''
            (out / 'fixture.cpp').write_text(fixture + body + main)
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            str(out / 'fixture.cpp'), '-o', str(out / 'fixture')],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(out / 'fixture')], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
