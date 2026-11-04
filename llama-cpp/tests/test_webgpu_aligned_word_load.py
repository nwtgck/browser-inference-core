"""Shared aligned word reads: bounded byte oracle and actual include composition."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_aligned_word_load as word
from webgpu_prefix_context import without_shared_word_load, without_checkpoint_readback
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


class Words:
    def __init__(self, data, allow_oob=False):
        self.data, self.allow_oob, self.reads = data, allow_oob, []

    def __getitem__(self, index):
        self.reads.append(index)
        if not 0 <= index * 4 <= len(self.data) - 4:
            if self.allow_oob:
                return 0  # Instrument reads only; not an assertion about WGSL errors.
            raise IndexError(index)
        return struct.unpack_from('<I', self.data, index * 4)[0]


def execute_helper(code, data, offset, allow_oob=False):
    """Execute only the actual pinned helper's small statement subset, tracing reads."""
    memory = Words(data, allow_oob)
    values = {'byte_offset': offset, 'src': memory, 'SRC0': memory,
              'select': lambda a, b, condition: b if condition else a}
    def expression(text):
        text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', text).replace(' / ', ' // ')
        return eval(text, {'__builtins__': {}}, values)
    lines = [line.strip() for line in code.splitlines()[1:-1]]
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.fullmatch(r'let (\w+) = (.*);', line)
        if match:
            values[match[1]] = expression(match[2])
        elif line == 'if (shift == 0u) {':
            if lines[i + 1:i + 3] != ['return lo;', '}']:
                raise AssertionError('Unreviewed helper branch')
            if values['shift'] == 0:
                return values['lo'] & 0xffffffff, memory.reads
            i += 2
        elif line.startswith('return '):
            return expression(line[7:-1]) & 0xffffffff, memory.reads
        else:
            raise AssertionError('Unreviewed helper statement: ' + line)
        i += 1
    raise AssertionError('Missing return')


class AlignedWordLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = without_checkpoint_readback(source)
        prefix.__enter__()
        cls.addClassCleanup(prefix.__exit__, None, None, None)
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='aligned-word-load-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_shared_word_load(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent
        cls.original = (cls.upstream / word.SOURCE_PATH).read_text()
        cls.candidate = (cls.final[False] / word.HEADER_NAME).read_text()
        shaders = cls.upstream / word.BASE / 'wgsl-shaders'
        cls.canonical = cls.directory / 'canonical-original.hpp'
        subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                        '--output_file', str(cls.canonical)], check=True, capture_output=True)

    def test_all_consumers_inverse_and_unaffected_bytes(self):
        self.assertEqual(word.INPUTS, source.q4_subgroup.OUTPUTS)
        self.assertEqual(word.OUTPUTS, source.FINAL_OUTPUTS)
        self.assertEqual(sha(self.canonical), word.CANONICAL_INPUT_SHA256)
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in new.iterdir()} - {p.name for p in old.iterdir()}, {word.HEADER_NAME})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {*word.EMBEDDING_OUTPUTS, 'ggml-webgpu-shader-lib.hpp'})
            for name, expected in word.OUTPUTS[profile].items():
                self.assertEqual(sha(new / name), expected)
            for name in [word.HEADER_NAME, *word.EMBEDDING_OUTPUTS]:
                previous = self.canonical.read_text() if name == word.HEADER_NAME else (old / name).read_text()
                current = (new / name).read_text()
                for helper in word.HELPERS:
                    a, b = word.definition(self.original, helper), word.definition(self.candidate, helper)
                    self.assertEqual(current.count(b), 12 if name == word.HEADER_NAME else 1)
                    self.assertNotIn(a, current)
                    current = current.replace(b, a)
                self.assertEqual(current, previous)  # Includes chunk delimiters/reserve values.
            self.assertEqual((new / 'ggml-webgpu.cpp').read_bytes(), (old / 'ggml-webgpu.cpp').read_bytes())
            library = (new / 'ggml-webgpu-shader-lib.hpp').read_text()
            self.assertEqual(library.count('#include "' + word.HEADER_NAME + '"'), 1)
            self.assertNotIn('"ggml-wgsl-shaders.hpp"', library)
            self.assertEqual(library.replace(word.HEADER_NAME, 'ggml-wgsl-shaders.hpp'),
                             (old / 'ggml-webgpu-shader-lib.hpp').read_text())

    def test_complete_spans_all_residues_and_final_q4_word(self):
        for helper in word.HELPERS:
            old, new = word.definition(self.original, helper), word.definition(self.candidate, helper)
            for seed in range(8):
                for offset in range(128):
                    size = (offset + 7) // 4 * 4
                    data = bytes((i * 37 + seed * 29) & 255 for i in range(size))
                    expected = int.from_bytes(data[offset:offset + 4], 'little')
                    value, reads = execute_helper(new, data, offset)
                    previous, old_reads = execute_helper(old, data, offset, True)
                    self.assertEqual(value, expected)
                    self.assertEqual(previous, value)
                    self.assertEqual(reads, [offset // 4, offset // 4 + (offset % 4 != 0)])
                    if offset % 4:
                        self.assertEqual(old_reads, reads)
            for residue in range(4):
                for bit in range(64):
                    data = (1 << bit).to_bytes(8, 'little')
                    expected = int.from_bytes(data[residue:residue + 4], 'little')
                    self.assertEqual(execute_helper(new, data, residue)[0], expected)
            for value in (0, 1, 0x01234567, 0x80000000, 0x89abcdef, 0xffffffff):
                self.assertEqual(execute_helper(new, struct.pack('<I', value), 0), (value, [0, 0]))
                self.assertEqual(execute_helper(old, struct.pack('<I', value), 0, True)[1], [0, 1])
            with self.assertRaises(IndexError):
                execute_helper(new, b'\x12\x34\x56\x78', 2)  # Incomplete span stays out of scope.
            for k, rows in ((64, 1), (128, 3), (1024, 4096), (4096, 4096)):
                end = k // 32 * rows * 18
                qword = end - 18 + 2 + 3 * 4
                self.assertEqual((qword, qword % 4), (end - 4, 0))
                # Translate the final word to a four-byte local view, without allocating the payload.
                self.assertEqual(execute_helper(new, b'\x12\x34\x56\x78', 0)[1], [0, 0])

    def test_fail_closed_and_atomic_idempotent_publication(self):
        for field in ('TEMPLATE_INPUT_SHA256', 'PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'output'
                with patch.object(word, field, '0' * 64), self.assertRaises(ValueError):
                    source.prepare(self.upstream, out)
                self.assertEqual(list(out.iterdir()), [])
        for profile in (False, True):
            final = self.final[profile]
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})
            previous = {p.name: p.read_bytes() for p in final.iterdir()}
            with patch.object(word, 'HEADER_SHA256', '0' * 64), self.assertRaises(ValueError):
                source.prepare(self.upstream, final, gpu_profile=not profile)
            self.assertEqual(previous, {p.name: p.read_bytes() for p in final.iterdir()})
        changed = self.canonical.read_bytes().replace(b'fn load_u32_at_src(', b'fn other(')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            word.replace_embedding(changed, self.original, self.candidate, count=12)

    def test_current_standalone_preparation_uses_final_embeddings_and_canonical(self):
        from webgpu_embedding_test_support import canonical_prefix, matvec_constants
        prepared = self.final[False]
        for directory in ('webgpu-q8-subgroup', 'webgpu-iq3s-subgroup'):
            spec = importlib.util.spec_from_file_location(directory, ROOT / 'tests' / directory / 'emit_factory.py')
            factory = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(factory)
            factory.prepare(self.upstream, prepared)
        prefix = '#include "ggml-wgsl-shaders.hpp"\n'
        constants = matvec_constants((prepared / 'ggml-webgpu-shader-lib.hpp').read_text()) + '\n'
        self.assertEqual(canonical_prefix(prefix, prepared), constants + '#include "' + word.HEADER_NAME + '"\n')
        self.assertEqual(canonical_prefix(prefix, self.before[False]), constants + prefix)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)
            (bad / 'ggml-webgpu-shader-lib.hpp').write_bytes((prepared / 'ggml-webgpu-shader-lib.hpp').read_bytes())
            with self.assertRaisesRegex(ValueError, 'embedding is missing'):
                canonical_prefix(prefix, bad)
            (bad / word.HEADER_NAME).write_text('unreviewed canonical')
            with self.assertRaisesRegex(ValueError, 'canonical word-loader embedding'):
                canonical_prefix(prefix, bad)
        with self.assertRaisesRegex(ValueError, 'include changed'):
            canonical_prefix(prefix + prefix, prepared)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_normal_profile_translation_units_exclude_old_canonical(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((ROOT / 'tests/webgpu-scale-unary/inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(sha(dawn / name), expected)
        poison = self.directory / 'ggml-wgsl-shaders.hpp'
        poison.write_text('#error old canonical embedding must not be included\n')
        for profile in (False, True):
            final = self.final[profile]
            includes = [final, self.directory, self.upstream / word.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
            command = ['c++', '-std=c++20', '-fsyntax-only', '-H', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4',
                       '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] + ['-I' + str(path) for path in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            if profile:
                command += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
            result = subprocess.run(command + [str(final / 'ggml-webgpu.cpp')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            traced = [line.lstrip('. ').strip() for line in result.stderr.splitlines() if line.startswith('.')]
            self.assertEqual(traced.count(str(final / word.HEADER_NAME)), 1)
            self.assertFalse(any(Path(name).name == 'ggml-wgsl-shaders.hpp' for name in traced))


if __name__ == '__main__':
    unittest.main()
