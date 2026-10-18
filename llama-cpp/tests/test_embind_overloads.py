"""Keep named Embind callables explicitly typed as upstream overload sets evolve."""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / 'bridge/chat-embind.cpp'

# The two standard select_overload signatures from Embind. This deliberately
# models only C++ overload selection, not JS marshalling or the Embind runtime.
# tests/chat-surface.mjs remains the real generated-module execution test.
SELECTORS = r'''
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>
namespace emscripten {
    template <typename Signature>
    Signature * select_overload(Signature * fn) { return fn; }
    template <typename Signature, typename Class>
    auto select_overload(Signature (Class::*fn)) -> decltype(fn) { return fn; }
}
'''

REGISTRATION = re.compile(r'(?P<member>\.)?\b(?P<kind>class_function|function)\s*\(\s*"(?P<name>[^"\n]+)"\s*,\s*')
SELECTION = re.compile(r'select_overload<(?P<signature>.*?)>\s*\(\s*&(?P<target>[\w:]+)\s*\)', re.DOTALL)


def without_comments(source):
    # Retain string/character literals so comment markers in a literal do not
    # eat a registration. This is a guard for the bridge's documented style,
    # not a general C++ parser or an attempt to infer upstream signatures.
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/', re.DOTALL)
    return tokens.sub(lambda match: ' ' if match[0].startswith(('//', '/*')) else match[0], source)


@dataclass(frozen=True)
class Binding:
    name: str
    kind: str
    signature: str
    owner: str | None
    target: str
    expression: str


def selector_arguments(text):
    """Split top-level template arguments, preserving commas inside function types."""
    arguments, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in '<([{':
            depth += 1
        elif char in '>)]}':
            depth -= 1
        elif char == ',' and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return arguments


def named_bindings(source):
    """Collect the actual selectors; reject untyped/implicitly inferred callables."""
    source = without_comments(source)
    result = []
    for call in REGISTRATION.finditer(source):
        argument = source[call.end():]
        if re.match(r'\+\s*\[\s*\]', argument):
            # Typed captureless adapters perform pointer/ownership conversions.
            # They are not unqualified addresses of an upstream overload set.
            continue
        selected = SELECTION.match(argument)
        if selected is None:
            raise ValueError(f'{call["name"]}: named callable requires select_overload<explicit signature>')
        arguments = selector_arguments(selected['signature'])
        signature = ' '.join(arguments[0].split())
        if '(' not in signature or re.search(r'\b(auto|decltype)\b', signature):
            raise ValueError(f'{call["name"]}: signature must explicitly state return/argument types and method qualifiers')
        kind = 'member' if call['member'] and call['kind'] == 'function' else 'static' if '::' in selected['target'] else 'free'
        owner = selected['target'].rsplit('::', 1)[0] if kind == 'member' else None
        if (kind == 'member' and (len(arguments) != 2 or arguments[1] != owner)
                or kind != 'member' and len(arguments) != 1):
            raise ValueError(f'{call["name"]}: member selectors must also explicitly name their owning class')
        result.append(Binding(call['name'], kind, signature, owner, selected['target'], selected[0]))
    if not result:
        raise ValueError('No named bindings found; update the guard if registration syntax changes')
    return result


# Distinct value types keep fixture signatures meaningful without fetching any
# upstream headers. They are not implementations of chat, JSON, or parser APIs.
TYPES = r'''
struct common_json {};
struct common_chat_msg {};
struct common_chat_msg_diff {};
struct common_chat_parser_params {};
struct common_chat_input {};
using llama_token = int32_t;
struct common_peg_arena {};
struct common_chat_tool {};
struct common_chat_msg_delimiters {};
struct common_chat_msg_spans {};
enum common_chat_role { role_fixture };
enum common_chat_continuation { continuation_fixture };
enum common_chat_tool_choice { choice_fixture };
enum common_reasoning_format { format_fixture };
using llama_tokens = std::vector<int32_t>;
struct added_argument {};
struct other_result {};
template <typename Function> void require_unambiguous(Function);
'''


def overload_cases(bindings, *, present=True, extras=True, reverse=False, bare=False, deduce_class=False):
    """Exercise every production signature against synthetic future overloads."""
    parts = [SELECTORS, TYPES]
    for index, binding in enumerate(bindings):
        label = f'binding_{index}_{binding.name}'
        parts.append(f'\n#line 1 "{label}"\nnamespace case_{index} {{\nusing namespace emscripten;\nusing clock = std::chrono::system_clock;')
        parts.append(f'using expected_signature = {binding.signature};')
        prefix = 'static ' if binding.kind == 'static' else ''
        declarations = ['expected_signature selected;'] if present else []
        if binding.kind == 'static' and present:
            declarations = ['static expected_signature selected;']
        additional = []
        if extras:
            additional += [f'{prefix}other_result selected(added_argument = {{}});',
                           f'template <typename... Args> {prefix}other_result selected(Args&&...);']
            if binding.kind == 'member':
                # A same-argument non-const/const method is a particularly easy
                # way to break an inferred pointer even with an unchanged arity.
                opposite = (binding.signature.removesuffix(' const') if binding.signature.endswith(' const')
                            else binding.signature + ' const')
                additional += [f'using other_cv = {opposite};', 'other_cv selected;']
        declarations = additional + declarations if reverse else declarations + additional
        if binding.kind in ('member', 'static'):
            parts.append('struct receiver { ' + '\n'.join(declarations) + '\n};')
            target = 'receiver::selected'
        else:
            parts += declarations
            target = 'selected'
        # Use the production type arguments with only the callee/class renamed
        # for this isolated fixture. The optional real-header test uses each
        # entire production selector unchanged.
        args = binding.signature + (', receiver' if binding.kind == 'member' and not deduce_class else '')
        expression = f'select_overload<{args}>(&{target})'
        pointer = 'expected_signature receiver::*' if binding.kind == 'member' else 'expected_signature *'
        if bare:
            parts.append(f'void probe() {{ require_unambiguous(&{target}); }}')
        else:
            parts.append(f'static_assert(std::is_same<decltype({expression}), {pointer}>::value, "wrong binding type");')
        parts.append('}')
    return '\n'.join(parts)


class BindingPolicyTests(unittest.TestCase):
    def test_every_direct_named_binding_is_typed(self):
        bindings = named_bindings(BRIDGE.read_text())
        # Regression count for the current exposed surface, not a production
        # registry. New exports intentionally need test review too.
        self.assertEqual(len(bindings), 49)
        self.assertEqual({binding.kind for binding in bindings}, {'free', 'static', 'member'})
        self.assertEqual(len({(binding.target, binding.signature, binding.owner) for binding in bindings}), len(bindings))

    def test_raw_free_static_and_member_registrations_are_rejected(self):
        for code in ['function("call", &native);', 'function("call", native);',
                     '.class_function("call", &Type::native)', '.function("call", &Type::native)',
                     '.function("call", select_const(&Type::native))']:
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, 'requires select_overload'):
                named_bindings(code)

    def test_inferred_signatures_are_not_a_prevention_mechanism(self):
        for signature in ['decltype(native())()', 'auto()', 'decltype(&native)']:
            with self.subTest(signature=signature), self.assertRaisesRegex(ValueError, 'explicitly state'):
                named_bindings(f'function("call", select_overload<{signature}>(&native));')

    def test_constness_reference_types_nested_templates_and_adapters_are_recognized(self):
        code = '''// .function("ignored", &T::bad)
          .class_function("from", select_overload<std::vector<T>(const std::map<int, T> &)>(&T::from))
          /* .function("also_ignored", &T::bad) */
          .function("read", select_overload<const T &() const, T>(&T::read))
          .function("adapter", +[](T & self, int n) { return self.call(n); });
        '''
        bindings = named_bindings(code)
        self.assertEqual([(b.name, b.kind) for b in bindings], [('from', 'static'), ('read', 'member')])
        self.assertEqual(bindings[1].signature, 'const T &() const')

    def test_member_class_deduction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'owning class'):
            named_bindings('.function("read", select_overload<bool() const>(&T::read))')
        with self.assertRaisesRegex(ValueError, 'owning class'):
            named_bindings('.function("read", select_overload<bool() const, Other>(&T::read))')

    def test_removing_any_production_selector_breaks_the_guard(self):
        source = BRIDGE.read_text()
        for binding in named_bindings(source):
            with self.subTest(target=binding.target), self.assertRaisesRegex(ValueError, 'requires select_overload'):
                named_bindings(source.replace(binding.expression, '&' + binding.target, 1))

    def test_empty_scan_cannot_pass_silently(self):
        with self.assertRaisesRegex(ValueError, 'No named bindings'):
            named_bindings('// empty')


class OverloadCompilerTests(unittest.TestCase):
    def setUp(self):
        self.compiler = os.environ.get('LCB_TEST_CXX') or shutil.which('clang++') or shutil.which('g++')
        if not self.compiler:
            self.skipTest('A native C++ compiler is needed for overload selection checks')
        self.tmp = tempfile.TemporaryDirectory(prefix='lcb-embind-overloads-')
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.bindings = named_bindings(BRIDGE.read_text())

    def compile(self, source, *, includes=(), executable=False):
        path = self.directory / 'probe.cpp'
        path.write_text(source)
        command = [self.compiler, '-std=c++17', '-Wall', '-Wextra']
        command += ['-I' + str(path) for path in includes]
        if not includes:
            command.append('-Werror')
        command += ['-o', str(self.directory / 'probe')] if executable else ['-fsyntax-only']
        # Negative tests compile every binding, not just the compiler's default
        # first twenty errors. Each diagnostic must name its fixture below.
        command += ['-ferror-limit=0'] if 'clang' in Path(self.compiler).name else ['-fmax-errors=0']
        return subprocess.run(command + [str(path)], capture_output=True, text=True, timeout=40)

    def test_all_current_signatures_compile_without_future_overloads(self):
        result = self.compile(overload_cases(self.bindings, extras=False))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_selected_signatures_survive_added_overloads_in_either_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                result = self.compile(overload_cases(self.bindings, reverse=reverse))
                self.assertEqual(result.returncode, 0, result.stderr)

    def assert_all_bindings_rejected(self, source, bindings=None):
        result = self.compile(source)
        self.assertNotEqual(result.returncode, 0)
        for index, binding in enumerate(self.bindings if bindings is None else bindings):
            self.assertIn(f'binding_{index}_{binding.name}:', result.stderr, result.stderr)

    def test_bare_addresses_fail_for_every_extended_overload_set(self):
        self.assert_all_bindings_rejected(overload_cases(self.bindings, bare=True))

    def test_member_function_templates_can_break_class_deduction(self):
        members = [binding for binding in self.bindings if binding.kind == 'member']
        # A signature alone leaves Class deduced in select_overload. The added
        # member template makes that deduction ambiguous despite an exact match.
        source = overload_cases(members, deduce_class=True)
        without_templates = re.sub(r'template <typename\.\.\. Args> other_result selected\(Args&&\.\.\.\);', '', source)
        result = self.compile(without_templates)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_all_bindings_rejected(source, members)

    def test_removed_signatures_do_not_silently_bind_added_overloads(self):
        self.assert_all_bindings_rejected(overload_cases(self.bindings, present=False))

    def test_selected_calls_keep_bool_const_reference_and_return_semantics(self):
        # Real native execution of selection, independent of Embind's marshalling.
        source = SELECTORS + r'''
#include <cassert>
namespace fixture {
    const std::string * seen = nullptr;
    std::string convert(const std::string & s, bool enabled) { seen = &s; return enabled ? s : "off"; }
    std::string convert(std::string &, bool) { return "wrong mutable overload"; }
    std::string convert(const std::string &, int) { return "wrong integer overload"; }
    struct Value {
        int value = 7;
        const int & get() const noexcept { return value; }
        int & get() { return value; }
        int get(int = 0) const { return -1; }
        void add(int & n) { value += n; ++n; }
        void add(const int &) { value = -1; }
        static int create() { return 9; }
        static int create(int = 0) { return -1; }
    };
    void run() {
        using emscripten::select_overload;
        auto convert_json = select_overload<std::string(const std::string &, bool)>(&convert);
        std::string input = "payload";
        assert(convert_json(input, true) == input && seen == &input);
        assert(convert_json(input, false) == "off" && input == "payload");
        auto get = select_overload<const int &() const, Value>(&Value::get);
        auto add = select_overload<void(int &), Value>(&Value::add);
        auto create = select_overload<int()>(&Value::create);
        Value value;
        int amount = 2;
        (value.*add)(amount);
        assert(value.value == 9 && amount == 3);
        const Value & view = value;
        assert(&(view.*get)() == &value.value);
        assert(create() == 9);
        using clock = std::chrono::system_clock;
        auto count = select_overload<clock::duration::rep() const, clock::duration>(&clock::duration::count);
        auto epoch = select_overload<clock::duration() const, clock::time_point>(&clock::time_point::time_since_epoch);
        auto now = select_overload<clock::time_point()>(&clock::now);
        clock::duration ticks(17);
        clock::time_point point(ticks);
        assert((ticks.*count)() == 17 && ((point.*epoch)().*count)() == 17);
        (void)now();
    }
}
int main() { fixture::run(); }
'''
        result = self.compile(source, executable=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(self.directory / 'probe')], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_selector_matches_the_checked_out_upstream_headers(self):
        # CI supplies the pinned submodule. An offline checkout may instead use
        # an explicitly supplied header tree without modifying the Git pins.
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (upstream / 'common/chat.h').is_file():
            self.skipTest('Initialize the pinned submodule or set LCB_TEST_LLAMA_SOURCE')
        source = ('#include "common/chat.h"\n#include "common/json-schema-to-grammar.h"\n'
                  '#include "common/reasoning-budget.h"\n' + SELECTORS
                  + '\nvoid probe() { using namespace emscripten; using clock = std::chrono::system_clock;\n'
                  + '\n'.join('(void)' + binding.expression + ';' for binding in self.bindings) + '\n}\n')
        result = self.compile(source, includes=(upstream, upstream / 'include', upstream / 'ggml/include'))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
