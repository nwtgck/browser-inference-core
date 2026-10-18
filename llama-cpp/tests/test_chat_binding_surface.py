"""Compile the entire bridge against real headers, with a type-only Embind shim.

Unlike selector-only probes this also checks fields, constructors and adapter
bodies. It is not an Emscripten link or a JavaScript marshalling test.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHIM = r'''
#pragma once
#include <type_traits>
#define EMSCRIPTEN_BINDINGS(name) void name()
namespace emscripten {
template<class Signature> Signature * select_overload(Signature * f) { return f; }
template<class Signature, class Class>
auto select_overload(Signature Class::* f) -> decltype(f) { return f; }
template<class T> struct class_ {
    explicit class_(const char *) {}
    template<class... Args> class_ & constructor() {
        static_assert(std::is_constructible<T, Args...>::value, "unconstructible bound class");
        return *this;
    }
    template<class Member> class_ & property(const char *, Member T::*) {
        static_assert(std::is_copy_constructible<Member>::value, "noncopyable bound property");
        return *this;
    }
    template<class F> class_ & function(const char *, F) { return *this; }
    template<class F> class_ & class_function(const char *, F) { return *this; }
};
template<class T> struct enum_ {
    explicit enum_(const char *) {}
    enum_ & value(const char *, T) { return *this; }
};
template<class F> void function(const char *, F) {}
template<class T> void register_vector(const char *) {
    static_assert(std::is_copy_constructible<T>::value, "noncopyable vector element");
}
template<class K, class V> void register_map(const char *) {}
}
'''


class ChatBindingSurface(unittest.TestCase):
    def setUp(self):
        self.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        self.compiler = os.environ.get('LCB_TEST_CXX') or shutil.which('clang++') or shutil.which('g++')
        if not self.compiler or not (self.upstream / 'common/chat.h').is_file():
            self.skipTest('A native C++ compiler and upstream checkout are required')
        self.tmp = tempfile.TemporaryDirectory(prefix='lcb-chat-surface-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / 'emscripten').mkdir()
        (self.work / 'emscripten/bind.h').write_text(SHIM)
        self.bridge = (ROOT / 'bridge/chat-embind.cpp').read_text()

    def compile(self, content):
        source = self.work / 'bridge.cpp'
        source.write_text(content)
        return subprocess.run([self.compiler, '-std=c++17', '-fsyntax-only',
                               '-I' + str(self.work), '-I' + str(self.upstream),
                               '-I' + str(self.upstream / 'include'), '-I' + str(self.upstream / 'ggml/include'),
                               str(source)], capture_output=True, text=True, timeout=45)

    def test_all_real_fields_constructors_and_adapter_bodies_compile(self):
        result = self.compile(self.bridge)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_removed_parser_field_is_detected_outside_selector_probes(self):
        old = 'FIELD(common_chat_parser_params, debug)'
        self.assertIn(old, self.bridge)
        result = self.compile(self.bridge.replace(old, 'FIELD(common_chat_parser_params, reasoning_format)'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('reasoning_format', result.stderr)

    def test_removed_arena_serializer_cannot_be_reintroduced_by_an_adapter(self):
        old = 'return self.value.parser();'
        self.assertIn(old, self.bridge)
        result = self.compile(self.bridge.replace(old, 'return self.value.parser().save();'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('save', result.stderr)


if __name__ == '__main__':
    unittest.main()
