"""Historical aligned-word patch comparison with frozen, explicitly owned inputs.

Current complete providers and standalone preparation live in source/ contracts.
"""
import hashlib
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path[:0] = [str(ROOT.parents[1]), str(runtime_test_root('llama-cpp'))]
# These tests isolate an accepted historical patch prefix.
from tests.support import frozen_webgpu_source as source
from tests.support.llama_cpp.frozen_webgpu_stages import aligned_word_load as word
from tests.support.shader_embedding import compiled_shaders
from tests.support.historical_webgpu import historical_source
from tests.support.llama_cpp.frozen_prefix_context import without_shared_word_load, without_checkpoint_readback
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


from tests.support import frozen_webgpu_source as legacy_source

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


class HistoricalAlignedWordLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = without_checkpoint_readback(source)
        prefix.__enter__()
        cls.addClassCleanup(prefix.__exit__, None, None, None)
        frozen = historical_source()
        cls.upstream = frozen.__enter__()
        cls.addClassCleanup(frozen.__exit__, None, None, None)
        cls.patch_root = cls.upstream / 'patches'
        cls.temp = tempfile.TemporaryDirectory(prefix='aligned-word-load-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_shared_word_load(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile, patch_root=cls.patch_root).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile, patch_root=cls.patch_root).parent
        cls.original = (cls.upstream / word.SOURCE_PATH).read_text()
        template_root = cls.directory / 'template-oracle'
        template = template_root / word.SOURCE_PATH
        template.parent.mkdir(parents=True)
        template.write_text(cls.original)
        subprocess.run(['git', 'apply', '--no-index',
                        str(cls.patch_root / word.PATCH_NAME)],
                       cwd=template_root, env=source.patch_environment(template_root), check=True, capture_output=True)
        cls.candidate = template.read_text()
        shaders = cls.upstream / word.BASE / 'wgsl-shaders'
        cls.canonical = cls.directory / 'canonical-original.hpp'
        subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                        '--output_file', str(cls.canonical)], check=True, capture_output=True)

    def test_historical_recording_prefix_keeps_its_original_include(self):
        from tests.support.llama_cpp.frozen_embedding import canonical_prefix, matvec_constants
        before = self.before[False]
        prefix = '#include "ggml-wgsl-shaders.hpp"\n'
        constants = matvec_constants((before / 'ggml-webgpu-shader-lib.hpp').read_text()) + '\n'
        self.assertEqual(canonical_prefix(prefix, before), constants + prefix)

    def test_all_consumers_inverse_and_unaffected_bytes(self):
        self.assertEqual(word.INPUTS, source.q4_subgroup.OUTPUTS)
        self.assertEqual(word.OUTPUTS, source.FINAL_OUTPUTS)
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in new.iterdir()} - {p.name for p in old.iterdir()}, {word.HEADER_NAME})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {*word.DEDICATED_SHADERS, 'ggml-webgpu-shader-lib.hpp'})
            for name, expected in word.OUTPUTS[profile].items():
                self.assertEqual(sha(new / name), expected)
            for name, shader in word.DEDICATED_SHADERS.items():
                strings = compiled_shaders({'before': old / name, 'after': new / name}, [shader.symbol],
                                           self.directory / f'dedicated-{profile}' / Path(name).stem)
                current = strings['after'][shader.symbol]
                for helper in word.HELPERS:
                    a, b = word.definition(self.original, helper), word.definition(self.candidate, helper)
                    self.assertEqual(current.count(b), 1)
                    self.assertNotIn(a, current)
                    current = current.replace(b, a)
                self.assertEqual(current, strings['before'][shader.symbol])
            self.assertEqual((new / 'ggml-webgpu.cpp').read_bytes(), (old / 'ggml-webgpu.cpp').read_bytes())
            library = (new / 'ggml-webgpu-shader-lib.hpp').read_text()
            self.assertEqual(library.count('#include "' + word.HEADER_NAME + '"'), 1)
            self.assertNotIn('"ggml-wgsl-shaders.hpp"', library)
            self.assertEqual(library.replace(word.HEADER_NAME, 'ggml-wgsl-shaders.hpp'),
                             (old / 'ggml-webgpu-shader-lib.hpp').read_text())

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_compiled_canonical_strings_preserve_all_shader_bodies(self):
        symbols = sorted(path.stem for path in (self.upstream / word.BASE / 'wgsl-shaders').glob('*.wgsl'))
        headers = {'reference': self.canonical,
                   'normal': self.final[False] / word.HEADER_NAME,
                   'profile': self.final[True] / word.HEADER_NAME}
        actual = compiled_shaders(headers, symbols, self.directory / 'compiled-canonical')
        for symbol in symbols:
            expected = actual['reference'][symbol]
            for helper in word.HELPERS:
                expected = expected.replace(word.definition(self.original, helper),
                                            word.definition(self.candidate, helper))
            self.assertEqual(actual['normal'][symbol], expected, symbol)
            self.assertEqual(actual['profile'][symbol], expected, symbol)
        # This deliberately compares decoded strings, not raw-string chunks or
        # reserve capacities chosen by the upstream embedding generator.

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_bad_generated_cpp_is_rejected_by_the_real_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bad = work / 'bad.hpp'
            bad.write_text('not a C++ declaration\n')
            with self.assertRaises(subprocess.CalledProcessError):
                compiled_shaders({'bad': bad}, ['arange'], work / 'compile')

    def test_include_redirection_is_an_explicit_scoped_patch(self):
        integration = self.patch_root / word.INTEGRATION_PATCH_NAME
        result = subprocess.check_output(['git', 'apply', '--no-index', '--numstat', '-z', str(integration)],
                                         cwd=self.directory, env=source.patch_environment(self.directory))
        self.assertEqual(result, b'1\t1\tggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp\0')
        with tempfile.TemporaryDirectory() as tmp:
            patches = Path(tmp) / 'patches'
            shutil.copytree(self.patch_root, patches)
            (patches / word.INTEGRATION_PATCH_NAME).write_text('not a patch\n')
            output = Path(tmp) / 'out'
            with self.assertRaises(subprocess.CalledProcessError):
                source.prepare(self.upstream, output, patch_root=patches)
            self.assertFalse(output.exists())

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
        for field in ('TEMPLATE_INPUT_SHA256', 'PATCH_SHA256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'output'
                with patch.object(word, field, '0' * 64), self.assertRaises(ValueError):
                    legacy_source.prepare(self.upstream, out, patch_root=self.patch_root)
                self.assertFalse(out.exists())
        for profile in (False, True):
            final = self.final[profile]
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            legacy_source.prepare(self.upstream, final, gpu_profile=profile, patch_root=self.patch_root)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})
            previous = {p.name: p.read_bytes() for p in final.iterdir()}
            with patch.object(word, 'generate_canonical', side_effect=RuntimeError('generator failed')), self.assertRaisesRegex(RuntimeError, 'generator failed'):
                legacy_source.prepare(self.upstream, final, gpu_profile=not profile, patch_root=self.patch_root)
            self.assertEqual(previous, {p.name: p.read_bytes() for p in final.iterdir()})


if __name__ == '__main__':
    unittest.main()
