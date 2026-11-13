"""Cheap C++ overload checks; these do not replace the real Wasm chat smoke test."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')

# Minimal type-deduction surface of Embind, not its runtime implementation.
# The production registration statement below is compiled unchanged against
# both upstream signatures, so this test catches overload ambiguity without an
# Emscripten installation. Real Embind/JavaScript marshalling is checked by
# tests/chat-surface.mjs after the runtime has been built.
EMBIND_TYPES = r'''
#include <string>
#include <type_traits>
namespace emscripten {
    template <typename Signature>
    typename std::add_pointer<Signature>::type select_overload(
        typename std::add_pointer<Signature>::type function) { return function; }

    template <typename ReturnType, typename... Args, typename... Policies>
    void function(const char * name, ReturnType (*fn)(Args...), Policies...);
}
'''

JSON_OVERLOAD = r'''
std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false) {
    seen_schema = &schema;
    return force_gbnf ? "forced" : "automatic";
}
'''
DOCUMENT_OVERLOAD = r'''
std::string json_schema_to_grammar(const common_chat_schema_document &) {
    return "wrong overload";
}
'''


class ChatEmbindTests(unittest.TestCase):
    def setUp(self):
        self.compiler = shutil.which('clang++') or shutil.which('g++')
        if not self.compiler:
            self.skipTest('A native C++ compiler is needed for overload regression checks')
        self.tmp = tempfile.TemporaryDirectory(prefix='lcb-chat-embind-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        source = (ROOT / 'native/chat/chat-embind.cpp').read_text()
        matches = re.findall(r'\bfunction\(\s*"json_schema_to_grammar"\s*,[^;]*;', source)
        self.assertEqual(len(matches), 1, 'Expected one JSON schema function registration')
        self.registration = matches[0]

    def compile(self, source, *, syntax_only=False, includes=()):
        file = self.directory / 'probe.cpp'
        file.write_text(source)
        command = [self.compiler, '-std=c++17', '-Wall', '-Wextra']
        if not syntax_only:
            command.append('-Werror')  # Fixtures are strict; unrelated upstream warnings are not errors.
        command += ['-I' + str(path) for path in includes]
        command += ['-fsyntax-only'] if syntax_only else ['-o', str(self.directory / 'probe')]
        return subprocess.run(command + [str(file)], capture_output=True, text=True, timeout=30)

    def fixture(self, declarations, registration=None):
        return EMBIND_TYPES + r'''
#include <cassert>
struct common_json {};
struct common_chat_schema_document {};
const common_json * seen_schema = nullptr;
''' + declarations + r'''
namespace emscripten {
    template <typename ReturnType, typename... Args, typename... Policies>
    void function(const char * name, ReturnType (*fn)(Args...), Policies...) {
        using expected = std::string (*)(const common_json &, bool);
        static_assert(std::is_same<decltype(fn), expected>::value,
                      "The JavaScript binding must keep the JSON + bool signature");
        assert(std::string(name) == "json_schema_to_grammar");
        const common_json schema;
        assert(fn(schema, false) == "automatic");
        assert(seen_schema == &schema);
        assert(fn(schema, true) == "forced");
        assert(seen_schema == &schema);
    }
}
int main() {
    using namespace emscripten;
''' + (registration if registration is not None else self.registration) + '\n}\n'

    def assert_compiles_and_runs(self, source):
        result = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(self.directory / 'probe')], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_single_overload_keeps_json_and_both_bool_values(self):
        # 3057bb66: common/json-schema-to-grammar.h declares only JSON + bool.
        self.assert_compiles_and_runs(self.fixture(JSON_OVERLOAD))

    def test_schema_document_overload_does_not_change_the_binding(self):
        # v0.4.1 (b29c606e) also declares a one-argument schema-document overload.
        # Neither declaration order should influence the selected signature.
        for declarations in (JSON_OVERLOAD + DOCUMENT_OVERLOAD, DOCUMENT_OVERLOAD + JSON_OVERLOAD):
            with self.subTest(declarations=declarations):
                self.assert_compiles_and_runs(self.fixture(declarations))

    def test_old_unqualified_registration_reproduces_the_compiler_failure(self):
        result = self.compile(self.fixture(JSON_OVERLOAD + DOCUMENT_OVERLOAD,
            'function("json_schema_to_grammar", &json_schema_to_grammar);'))
        self.assertNotEqual(result.returncode, 0, 'The ambiguity fixture must reject the old binding')
        self.assertIn('function', result.stderr)

    def test_missing_json_overload_cannot_silently_bind_the_document_version(self):
        result = self.compile(self.fixture(DOCUMENT_OVERLOAD))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('json_schema_to_grammar', result.stderr)

    def test_checked_out_upstream_header_accepts_the_registration(self):
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (upstream / 'common/json-schema-to-grammar.h').is_file():
            self.skipTest('Initialize the pinned submodule or set LCB_TEST_LLAMA_SOURCE')
        source = ('#include "common/json-schema-to-grammar.h"\n' + EMBIND_TYPES
                  + '\nvoid check_registration() { using namespace emscripten;\n'
                  + self.registration + '\n}\n')
        result = self.compile(source, syntax_only=True,
            includes=(upstream, upstream / 'include', upstream / 'ggml/include'))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
