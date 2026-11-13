"""Compiler-selected dispatch boundaries, including misleading text and UTF-8."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
import json
from unittest.mock import patch

from tests.support.llama_cpp import dispatch_ast

from tests.support.llama_cpp.dispatch_ast import extract_dispatch, macro_definition

COMPUTE = 'static inline void compute_2d_workgroups(unsigned n, unsigned limit, unsigned & x, unsigned & y) { x = n; y = limit; }'
BRANCH = '''{
        const char * misleading = "} else {";
        (void)misleading;
        wg_x = 7; wg_y = 11;
    }'''


def unit(branch=BRANCH):
    return ('// 日本語: byte offsets must not become Python string offsets.\n' + COMPUTE
            + '\nvoid ggml_webgpu_mul_mat(bool use_mat_vec) {\n'
            '    unsigned wg_x = 1, wg_y = 1;\n'
            '    if (use_mat_vec) { const char * other = "wg_x wg_y"; (void)other; }\n'
            '    if ((use_mat_vec)) ' + branch + ' else { wg_x = 1; wg_y = 1; }\n}\n')


@unittest.skipUnless(shutil.which(os.environ.get('CLANGXX', 'clang++')), 'Clang is required for semantic source ranges')
class DispatchAst(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dispatch-ast-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'provider.cpp'

    def extract(self, text):
        self.source.write_text(text)
        return extract_dispatch(self.source, ['-std=c++20'])

    def test_comments_unicode_braces_and_unrelated_if_do_not_choose_code(self):
        compute, branch = self.extract(unit())
        self.assertEqual(compute, COMPUTE)
        self.assertEqual(branch, BRANCH)

    def test_ambiguous_dispatch_is_rejected(self):
        text = unit().replace('if ((use_mat_vec)) ', 'if (use_mat_vec) { wg_x = 2; wg_y = 3; }\nif ((use_mat_vec)) ')
        with self.assertRaisesRegex(ValueError, 'one matvec workgroup branch; got 2'):
            self.extract(text)

    def test_missing_dispatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one matvec workgroup branch; got 0'):
            self.extract(unit().replace('if ((use_mat_vec))', 'if (false)'))

    def test_macro_boundary_is_not_silently_sliced(self):
        text = '#define DISPATCH { wg_x = 7; wg_y = 11; }\n' + unit('DISPATCH')
        with self.assertRaisesRegex(ValueError, 'macro-expanded'):
            self.extract(text)

    def test_invalid_unreferenced_code_fails_before_partial_ast_can_be_used(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.extract(unit() + '\nstruct unused { unknown_dispatch_type member; };\n')
        self.assertIn('unknown_dispatch_type', failure.exception.stderr)

    def test_definition_in_a_different_header_is_not_a_current_provider_definition(self):
        (self.root / 'other.hpp').write_text(COMPUTE)
        with self.assertRaisesRegex(ValueError, 'compute_2d_workgroups; got 0'):
            self.extract(unit().replace(COMPUTE, '#include "other.hpp"'))

    def test_shadow_dispatch_state_is_not_accepted_by_name_alone(self):
        text = unit().replace('if ((use_mat_vec)) ', '{ unsigned wg_x = 3; (void)wg_x; }\nif ((use_mat_vec)) ')
        with self.assertRaisesRegex(ValueError, 'one dispatch declaration of wg_x; got 2'):
            self.extract(text)

    def test_diagnostic_template_uses_primary_definition_not_instantiations_or_wrapper(self):
        source = unit().replace('void ggml_webgpu_mul_mat(bool',
                                'template<bool Diagnostic> void ggml_webgpu_mul_mat_impl(bool')
        source += ('\nvoid ggml_webgpu_mul_mat(bool select) { ggml_webgpu_mul_mat_impl<false>(select); }'
                   '\nvoid diagnose() { ggml_webgpu_mul_mat_impl<true>(true); }\n')
        self.source.write_text(source)
        compute, branch = extract_dispatch(self.source, ['-std=c++20'],
                                           matmul_name='ggml_webgpu_mul_mat_impl')
        self.assertEqual(compute, COMPUTE)
        self.assertEqual(branch, BRANCH)

    def test_complete_consumer_retains_primary_template_parameters(self):
        primary = 'template<bool Diagnostic> int selected() { return Diagnostic ? 7 : 3; }'
        self.source.write_text(primary + '\nint instantiate() { return selected<false>() + selected<true>(); }')
        node, data = dispatch_ast.function_ast(self.source, 'selected', ['-std=c++20'],
                                               include_template=True)
        selected = dispatch_ast._source_range(node, self.source, data)
        self.assertEqual(selected, primary)
        consumer = self.root / 'consumer.cpp'
        consumer.write_text(selected + '\nint main() { return selected<true>() == 7 ? 0 : 1; }')
        compiler = shutil.which(os.environ.get('CLANGXX', 'clang++'))
        executable = self.root / 'consumer'
        subprocess.run([compiler, '-std=c++20', str(consumer), '-o', str(executable)],
                       check=True, capture_output=True, text=True)
        subprocess.run([str(executable)], check=True)

    def test_current_macro_is_not_replaced_by_a_handwritten_recorder_formula(self):
        self.source.write_text('#define CEIL_DIV(M, N) ((M) / (N) + 7)\n' + unit())
        observed = macro_definition(self.source, 'CEIL_DIV', ['-std=c++20'])
        self.assertEqual(observed, '#define CEIL_DIV(M,N) ((M) / (N) + 7)')
        consumer = self.source.parent / 'macro.cpp'
        consumer.write_text(observed + '\nstatic_assert(CEIL_DIV(9, 4) == 3);\n')
        compiler = shutil.which(os.environ.get('CLANGXX', 'clang++'))
        result = subprocess.run([compiler, '-std=c++20', '-fsyntax-only', str(consumer)],
                                capture_output=True, text=True, timeout=30)
        # A usual ceiling would be 3, not 9; the deliberate changed formula
        # must remain visible, rather than being silently copied from the mock.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('static assertion failed', result.stderr)
        consumer.write_text(observed + '\nstatic_assert(CEIL_DIV(9, 4) == 9);\n')
        subprocess.run([compiler, '-std=c++20', '-fsyntax-only', str(consumer)],
                       check=True, capture_output=True, text=True, timeout=30)

    def test_macro_lifetime_changes_are_not_guessed(self):
        for tail in ('#undef CEIL_DIV\n', '#undef CEIL_DIV\n#define CEIL_DIV(M,N) 1\n'):
            with self.subTest(tail=tail):
                self.source.write_text('#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))\n' + unit() + tail)
                with self.assertRaisesRegex(ValueError, 'unambiguous current macro'):
                    macro_definition(self.source, 'CEIL_DIV', ['-std=c++20'])

    def test_preprocessor_normalizes_multiline_macro_definitions(self):
        self.source.write_text('#define CEIL_DIV(M,N) \\\n (((M)+(N)-1)/(N))\n' + unit())
        self.assertEqual(macro_definition(self.source, 'CEIL_DIV', ['-std=c++20']),
                         '#define CEIL_DIV(M,N) (((M)+(N)-1)/(N))')


class CurrentDispatchWiring(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dispatch-wiring-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'ggml-webgpu.cpp').write_text('// unchanged current provider')
        (self.root / dispatch_ast.webgpu_plan.RECEIPT).write_text(json.dumps({'gpuProfile': False}))

    def test_observed_macro_replaces_the_mock_formula_in_the_emitted_consumer(self):
        observed = '#define CEIL_DIV(M,N) ((M) / (N) + 7)'
        with patch.object(dispatch_ast, 'dawn_package', return_value=self.root), \
                patch.object(dispatch_ast, 'extract_dispatch', return_value=(COMPUTE, BRANCH)), \
                patch.object(dispatch_ast, 'macro_definition', return_value=observed) as macro:
            result = dispatch_ast.current_dispatch(self.root, self.root)
        self.assertTrue(result.startswith('#undef CEIL_DIV\n' + observed + '\n' + COMPUTE))
        self.assertEqual(macro.call_args.args[:2], (self.root / 'ggml-webgpu.cpp', 'CEIL_DIV'))
        self.assertIn(BRANCH, result)

    def test_source_change_between_ast_and_macro_passes_is_rejected(self):
        def change_source(*_arguments):
            (self.root / 'ggml-webgpu.cpp').write_text('// changed provider')
            return '#define CEIL_DIV(M,N) 1'
        with patch.object(dispatch_ast, 'dawn_package', return_value=self.root), \
                patch.object(dispatch_ast, 'extract_dispatch', return_value=(COMPUTE, BRANCH)), \
                patch.object(dispatch_ast, 'macro_definition', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'between AST and macro passes'):
                dispatch_ast.current_dispatch(self.root, self.root)
