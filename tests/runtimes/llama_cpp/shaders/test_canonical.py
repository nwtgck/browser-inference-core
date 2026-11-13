"""Canonical generation handles chunk boundaries without editing C++ bodies."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT, runtime_root
from build_tools.source import wgsl
from tests.support.shader_embedding import compiled_shaders

ROOT = runtime_root('llama-cpp')
SHADERS = Path('ggml/src/ggml-webgpu/wgsl-shaders')


class CanonicalGeneration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'upstream'
        self.shaders = self.source / SHADERS
        self.shaders.mkdir(parents=True)
        self.original = '// original template\n'
        (self.shaders / 'common_decls.tmpl').write_text(self.original)
        self.shader = '#include "common_decls.tmpl"\n@compute @workgroup_size(1) fn main() {}\n'
        (self.shaders / 'sample.wgsl').write_text(self.shader)
        # Lower only the fixture generator's chunk size to force a function body
        # across C++ raw strings. The production upstream is never changed.
        generator = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')) / SHADERS / 'embed_wgsl.py'
        text = generator.read_text()
        self.assertEqual(text.count('max_chunk_len=60000'), 1)
        (self.shaders / 'embed_wgsl.py').write_text(text.replace('max_chunk_len=60000', 'max_chunk_len=32'))
        self.template = self.root / 'private/common_decls.tmpl'
        self.template.parent.mkdir()
        self.candidate = 'fn helper(value: u32) -> u32 {\n    return value + 1u;\n}\n'
        self.template.write_text(self.candidate)
        self.output = self.root / 'shader.hpp'

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_function_crossing_chunks_and_source_comments_survive_compilation(self):
        wgsl.generate(self.shaders, self.template, self.output)
        self.assertNotIn(self.candidate, self.output.read_text())
        result = compiled_shaders({'generated': self.output}, ['sample'], self.root / 'compile')
        self.assertEqual(result['generated']['sample'], self.shader.replace('#include "common_decls.tmpl"', self.candidate))
        (self.shaders / 'sample.wgsl').write_text(self.shader + '// harmless comment\n')
        wgsl.generate(self.shaders, self.template, self.output)
        result = compiled_shaders({'generated': self.output}, ['sample'], self.root / 'recompile')
        self.assertTrue(result['generated']['sample'].endswith('// harmless comment\n'))
        self.assertEqual((self.shaders / 'common_decls.tmpl').read_text(), self.original)

    def test_generator_failure_is_not_replaced_with_a_stale_success(self):
        (self.shaders / 'embed_wgsl.py').write_text('raise RuntimeError("injected failure")\n')
        with self.assertRaises(subprocess.CalledProcessError):
            wgsl.generate(self.shaders, self.template, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_input_and_linked_templates_are_rejected(self):
        embedder = self.shaders / 'embed_wgsl.py'
        embedder.unlink()
        with self.assertRaises(ValueError):
            wgsl.generate(self.shaders, self.template, self.output)
        embedder.symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError):
            wgsl.generate(self.shaders, self.template, self.output)

    def test_module_entrypoint_runs_from_the_repository_root(self):
        result = subprocess.run([sys.executable, '-m', 'build_tools.source.wgsl',
                                 '--source', str(self.shaders), '--template', str(self.template),
                                 '--output', str(self.output)], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_file())

    def test_partial_generator_output_does_not_replace_previous_header(self):
        self.output.write_text('previous complete header')
        (self.shaders / 'embed_wgsl.py').write_text(
            'import pathlib,sys\npathlib.Path(sys.argv[-1]).write_text("partial")\nraise RuntimeError("injected")\n')
        with self.assertRaises(subprocess.CalledProcessError):
            wgsl.generate(self.shaders, self.template, self.output)
        self.assertEqual(self.output.read_text(), 'previous complete header')
        self.assertEqual(list(self.root.glob('wgsl-embed-*')), [])

    def test_generator_cannot_overwrite_original_template(self):
        original = self.shaders / 'common_decls.tmpl'
        with self.assertRaisesRegex(ValueError, 'outside'):
            wgsl.generate(self.shaders, original, self.output)
        self.assertEqual(original.read_text(), self.original)

    def test_deleted_shader_and_unrelated_private_neighbor_are_not_reused(self):
        extra = self.shaders / 'extra.wgsl'
        extra.write_text('fn extra() {}\n')
        (self.template.parent / 'unrelated.wgsl').write_text('must never be an input')
        wgsl.generate(self.shaders, self.template, self.output)
        self.assertIn('wgsl_extra', self.output.read_text())
        self.assertNotIn('wgsl_unrelated', self.output.read_text())
        extra.unlink()
        wgsl.generate(self.shaders, self.template, self.output)
        self.assertNotIn('wgsl_extra', self.output.read_text())
        self.assertEqual(set(p.name for p in self.template.parent.iterdir()), {'common_decls.tmpl', 'unrelated.wgsl'})


    def test_dedicated_inputs_can_use_an_external_unchanged_generator(self):
        private = self.root / 'dedicated'
        private.mkdir()
        (private / 'common_decls.tmpl').write_text(self.candidate)
        (private / 'sample.wgsl').write_text(self.shader)
        generator = self.shaders / 'embed_wgsl.py'
        wgsl.generate(private, None, self.output, generator=generator)
        self.assertTrue(self.output.is_file())
        with self.assertRaisesRegex(ValueError, 'input or a link'):
            wgsl.generate(private, None, generator.parent / 'bad.hpp', generator=generator)

    def test_stdout_and_stderr_reach_cli_and_source_composition_callers(self):
        generator = self.shaders / 'embed_wgsl.py'
        generator.write_text('import sys\nprint("GENERATOR_STDOUT", flush=True)\nprint("GENERATOR_STDERR", file=sys.stderr, flush=True)\nraise RuntimeError("generator broke")\n')
        commands = [[sys.executable, '-m', 'build_tools.source.wgsl', '--source', str(self.shaders),
                     '--template', str(self.template), '--output', str(self.output)],
                    [sys.executable, '-c',
                     'from pathlib import Path; from tests.support.llama_cpp.webgpu_stages.aligned_word_load import generate_canonical; '
                     f'generate_canonical(Path({str(self.source)!r}), Path({str(self.template)!r}), Path({str(self.output)!r}))']]
        for command in commands:
            with self.subTest(entry=command[1]):
                result = subprocess.run(command, cwd=REPOSITORY_ROOT, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('GENERATOR_STDOUT', result.stdout)
                self.assertIn('GENERATOR_STDERR', result.stderr)
                self.assertIn('generator broke', result.stderr)
                self.assertFalse(self.output.exists())
