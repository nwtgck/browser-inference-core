"""Adversarial selections from actual Clang syntax trees, not parser mocks."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.support.llama_cpp.copy_ast import Declarations


@unittest.skipUnless(shutil.which('clang++'), 'Clang required')
class CurrentCopySelections(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / 'source.cpp'
        self.header = self.root / 'interface.h'
        self.header.write_text('struct ggml_iface { void (*first)(); void (*copy)(); };\n')

    def parse(self, code, names=None, header=None):
        self.source.write_text(code)
        return Declarations(self.source, ['-std=c++20'], names or {
            'ggml_target', 'ggml_other', 'ggml_iface', 'ggml_registration',
            'ggml_backend_sched', 'ggml_backend_sched_new'},
            headers=(header or self.header,))

    def registration(self, entries='ggml_other, ggml_target', prefix=''):
        return ('#include "interface.h"\n' + prefix + '\n'
                'void ggml_target() {}\nvoid ggml_other() {}\n'
                'const ggml_iface ggml_registration = {' + entries + '};\n')

    def target(self, unit):
        return unit.registered_function('ggml_registration', 'ggml_iface', 'copy', self.header)

    def test_semantic_field_order_and_comments_not_fixed_indexes(self):
        code = self.registration(prefix='// .copy = ggml_other, unrelated fake registration\n')
        self.assertEqual(self.target(self.parse(code)), 'ggml_target')
        self.header.write_text('struct ggml_iface { void (*copy)(); void (*first)(); };\n')
        self.assertEqual(self.target(self.parse(code)), 'ggml_other')

    def test_alias_and_address_of_direct_function_are_accepted(self):
        code = self.registration('ggml_other, (&ggml_target)')
        code = code.replace('const ggml_iface ggml_registration',
                            'using iface_alias = ggml_iface; const iface_alias ggml_registration')
        self.assertEqual(self.target(self.parse(code)), 'ggml_target')

    def test_null_conditional_and_macro_registrations_are_not_guessed(self):
        for value in ('nullptr', 'true ? ggml_target : ggml_other', 'COPY_TARGET'):
            with self.subTest(value=value):
                unit = self.parse(self.registration('ggml_other, ' + value,
                                                     '#define COPY_TARGET ggml_target'))
                with self.assertRaises(ValueError):
                    self.target(unit)

    def test_foreign_definition_is_not_treated_as_local(self):
        self.header.write_text(self.header.read_text() + 'inline void ggml_target() {}\n')
        unit = self.parse(self.registration().replace('void ggml_target() {}', ''))
        with self.assertRaisesRegex(ValueError, 'definition'):
            self.target(unit)

    def test_unused_type_error_anywhere_in_unit_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.parse(self.registration() + 'struct unused { missing_type value; };\n')

    def test_overloaded_function_name_is_not_arbitrarily_selected(self):
        unit = self.parse('void ggml_target() {}\nvoid ggml_target(int) {}\n')
        with self.assertRaisesRegex(ValueError, 'got 2'):
            unit.function('ggml_target')

    def test_macro_function_definition_is_not_a_physical_selection(self):
        unit = self.parse('#define DEF(name) void name() {}\nDEF(ggml_target)\n')
        with self.assertRaises(ValueError):
            unit.function('ggml_target')

    def test_utf8_byte_ranges_and_source_mutations(self):
        unit = self.parse('// 日本語 documentation\nvoid ggml_target() { /* ü */ }\n')
        self.assertEqual(unit.function('ggml_target').strip(), 'void ggml_target() { /* ü */ }')
        self.source.write_text(self.source.read_text() + '// changed\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            unit.function('ggml_target')

    def test_header_mutation_is_rejected(self):
        unit = self.parse(self.registration())
        self.header.write_text(self.header.read_text() + '// changed\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.target(unit)

    def scheduler(self, loops):
        return ('struct ggml_backend_sched { int host_input_snapshot[8]; };\n'
                'void ggml_backend_sched_new(int n_backends, ggml_backend_sched* sched) {\n'
                + loops + '\n}\n')

    def test_discovery_uses_parameter_and_field_identity(self):
        loop = 'for (int i = 0; i < n_backends; ++i) { sched->host_input_snapshot[i] = 1; }'
        decoy = '{ int n_backends = 8; ' + loop + ' }'
        unit = self.parse(self.scheduler(decoy + '\n' + loop))
        self.assertEqual(unit.constructor_discovery(), loop)

    def test_multiple_discovery_loops_are_rejected(self):
        loop = 'for (int i = 0; i < n_backends; ++i) { sched->host_input_snapshot[i] = 1; }'
        unit = self.parse(self.scheduler(loop + '\n' + loop))
        with self.assertRaisesRegex(ValueError, 'got 2'):
            unit.constructor_discovery()

    def test_macro_discovery_loop_is_rejected(self):
        loop = 'for (int i = 0; i < n_backends; ++i) { sched->host_input_snapshot[i] = 1; }'
        unit = self.parse('#define LOOP ' + loop + '\n' + self.scheduler('LOOP'))
        with self.assertRaises(ValueError):
            unit.constructor_discovery()


if __name__ == '__main__':
    unittest.main()
