"""Type emission preserves C identifiers while producing compilable C++."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_tools.bindings import cpp_type, write_generated_file
from build_tools.runtimes.llama_cpp import bindings as llama
from build_tools.runtimes.stable_diffusion_cpp import bindings as image


class TypeSpelling(unittest.TestCase):
    def test_boolean_type_positions(self):
        for original, expected in [
            ('_Bool', 'bool'), ('const _Bool *', 'const bool *'),
            ('_Bool (*)(_Bool, const _Bool *)', 'bool (*)(bool, const bool *)'),
            ('struct llama_Bool_state *', 'struct llama_Bool_state *'),
            ('sd_Bool_t', 'sd_Bool_t'), ('bool', 'bool'),
        ]:
            with self.subTest(original=original):
                self.assertEqual(cpp_type(original), expected)

    @unittest.skipUnless(shutil.which('clang') and shutil.which('clang++'), 'requires Clang')
    def test_real_ast_and_generated_cpp_preserve_boolean_identifiers(self):
        for generator, prefix, header in [(llama, 'llama', 'llama.h'),
                                           (image, 'sd', 'stable-diffusion.h')]:
            with self.subTest(runtime=prefix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); include = root / 'include'; include.mkdir()
                for name in ('llama.h', 'stable-diffusion.h', 'gguf.h', 'ggml.h',
                             'ggml-alloc.h', 'ggml-backend.h', 'mtmd.h',
                             'mtmd-helper.h', 'mtmd-vision-bf16.h'):
                    (include / name).write_text('')
                declarations = f"""#pragma once
#include <stdbool.h>
#include <stdint.h>
typedef struct {{ uint16_t bits; }} ggml_bf16_t;
typedef struct {prefix}_Bool_state {{ bool _Bool_field; }} {prefix}_Bool_state;
bool {prefix}_Bool_identity(bool value);
bool * {prefix}_Bool_pointer(const bool * value);
bool {prefix}_Bool_callback(bool (*callback)(bool), bool value);
{prefix}_Bool_state {prefix}_Bool_record({prefix}_Bool_state value);
"""
                (include / header).write_text(declarations)
                output = root / 'out'
                # The fixture has no project-specific metadata getters/constants.
                with patch.object(generator, 'EXTRA_CONSTANTS', []), \
                     patch.object(llama, 'CALLBACK_METADATA_GETTERS', {}):
                    schema = generator.generate(root, output, 'clang')
                outputs = ('bindings.cpp', 'exports.json', 'jspi-exports.json',
                           'schema.json', 'schema.mjs', 'functions.d.ts')
                original = {name: ((output / name).stat().st_ino, (output / name).stat().st_mtime_ns,
                                    (output / name).read_bytes()) for name in outputs}
                with patch.object(generator, 'EXTRA_CONSTANTS', []), \
                     patch.object(llama, 'CALLBACK_METADATA_GETTERS', {}):
                    self.assertEqual(schema, generator.generate(root, output, 'clang'))
                self.assertEqual(original, {name: ((output / name).stat().st_ino, (output / name).stat().st_mtime_ns,
                                                   (output / name).read_bytes()) for name in outputs})
                functions = {function['name'] for function in schema['functions']}
                self.assertIn(prefix + '_Bool_identity', functions)
                source = (output / 'bindings.cpp').read_text()
                self.assertIn(prefix + '_Bool_identity', source)
                self.assertIn('_Bool_field', source)
                command = ['clang++', '-std=c++17', '-Werror', '-fsyntax-only',
                           '-I' + str(include), str(output / 'bindings.cpp')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                # A whole-source replacement must break these actual declarations.
                (output / 'bindings.cpp').write_text(source.replace('_Bool', 'bool'))
                mutant = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(mutant.returncode, 0)


class GeneratedFiles(unittest.TestCase):
    def test_unchanged_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'generated.cpp'
            write_generated_file(path, 'first\n')
            original = path.stat()
            write_generated_file(path, 'first\n')
            self.assertEqual((path.stat().st_ino, path.stat().st_mtime_ns),
                             (original.st_ino, original.st_mtime_ns))
            write_generated_file(path, 'changed\n')
            self.assertEqual(path.read_bytes(), b'changed\n')

    def test_linked_output_is_rejected_without_modifying_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'; target.write_text('retain')
            output = root / 'generated'; output.symlink_to(target)
            with self.assertRaises(ValueError):
                write_generated_file(output, 'replace')
            self.assertEqual(target.read_text(), 'retain')
