"""Check exception discoverability and the upstream-only binding/build boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_mtmd import PATCH_DIRECTORY, prepare
from generate_bindings import generate


class UpstreamPatchPolicy(unittest.TestCase):
    def test_policy_is_discoverable_from_the_root_and_the_patch_directory(self):
        directory = ROOT / PATCH_DIRECTORY
        root_rules = (ROOT / 'AGENTS.md').read_text()
        rules = (directory / 'AGENTS.md').read_text()
        register = (directory / 'README.md').read_text()
        self.assertIn(PATCH_DIRECTORY + '/AGENTS.md', root_rules)
        self.assertIn('explicit user approval', root_rules)
        self.assertIn('omitting or deferring the feature', rules)
        self.assertIn('independent', rules.lower())
        self.assertIn('Removal condition', register)
        self.assertFalse((ROOT / 'patches').exists(), 'Do not leave a second generic patch directory')
        self.assertEqual({p.name for p in directory.glob('*.patch')}, {
            'mtmd-webgpu-bf16.patch', 'mtmd-audio-single-thread.patch',
            'ggml-webgpu-moe-direct-slot.patch', 'ggml-webgpu-same-device-tensor-copy.patch',
            'ggml-webgpu-batch-param-uploads.patch'})
        for patch in directory.glob('*.patch'):
            self.assertIn(patch.name, register)
        # Source-rewriting and toolchain exceptions are covered too, not only .patch.
        self.assertIn('scripts/', root_rules)
        self.assertIn('toolchain', rules)
        self.assertIn('scripts/patch_emscripten.py', register)

    def test_cli_resolves_the_renamed_patches_for_both_components(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (source / 'tools/mtmd/clip.cpp').is_file():
            self.skipTest('Initialize the pinned submodule or set LCB_TEST_LLAMA_SOURCE')
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            for component, filename, patch in (
                ('vision', 'clip.cpp', 'mtmd-webgpu-bf16.patch'),
                ('audio', 'mtmd-audio.cpp', 'mtmd-audio-single-thread.patch'),
            ):
                with self.subTest(component=component):
                    before = (source / 'tools/mtmd' / filename).read_bytes()
                    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_mtmd.py'),
                                    '--source', str(source), '--output', str(work / component),
                                    '--component', component], check=True, capture_output=True)
                    reference = prepare(source, work / (component + '-direct'),
                                        ROOT / PATCH_DIRECTORY / patch, filename=filename)
                    self.assertEqual((work / component / filename).read_bytes(), reference.read_bytes())
                    self.assertEqual((source / 'tools/mtmd' / filename).read_bytes(), before)

    def test_binding_generator_uses_the_unmodified_upstream_audio_header(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        header = source / 'tools/mtmd/mtmd-helper.h'
        compiler = shutil.which('clang')
        if not compiler or not header.is_file():
            self.skipTest('A Clang compiler and upstream checkout are required')
        before = header.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            schema = generate(source, output, compiler)
            functions = {f['name'] for f in schema['functions']}
            self.assertIn('mtmd_helper_gen_audio_get_output', functions)
            # Do not forbid a future upstream implementation; expose a query only
            # when it actually exists in the upstream header, not in a local copy.
            name = 'mtmd_helper_gen_audio_supports_language_auto'
            self.assertEqual(name in functions, name in header.read_text())
            self.assertNotIn('const const ', (output / 'bindings.cpp').read_text())
            self.assertEqual(json.loads((output / 'schema.json').read_text()), schema)
        self.assertEqual(header.read_bytes(), before)

    def test_cmake_replaces_only_the_two_accepted_translation_units(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not shutil.which('cmake') or not (source / 'CMakeLists.txt').is_file():
            self.skipTest('CMake and an upstream checkout are required')
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / 'build'
            subprocess.run(['cmake', '-S', str(ROOT / 'tests/mtmd-overlay'), '-B', str(build),
                            '-DLCB_LLAMA_SOURCE=' + str(source)], check=True, capture_output=True, text=True)
            before = (build / 'upstream-mtmd-sources.txt').read_text().strip().split(';')
            after = (build / 'mtmd-sources.txt').read_text().strip().split(';')
            self.assertEqual(len(after), len(before))
            replaced = {'clip.cpp', 'mtmd-audio.cpp'}
            self.assertEqual(set(before) - set(after), replaced)
            copies = set(after) - set(before)
            self.assertEqual({Path(p).name for p in copies}, replaced)
            self.assertTrue(all(Path(p).is_file() for p in copies))
            for p in copies:
                self.assertEqual(after.count(p), 1)
            self.assertEqual((build / 'upstream-mtmd-headers.txt').read_bytes(),
                             (build / 'mtmd-headers.txt').read_bytes())
            self.assertFalse((build / 'mtmd-tts-overlay').exists())
            # A previously configured build may still have the retired copy on
            # disk. A reconfigure must not consume it or its private header.
            stale = build / 'mtmd-tts-overlay/tools/mtmd'
            (stale / 'models').mkdir(parents=True)
            poison = '#error retired overlay must not be included\n'
            (stale / 'models/models.h').write_text(poison)
            (stale / 'mtmd-helper.h').write_text(poison)
            subprocess.run(['cmake', '-S', str(ROOT / 'tests/mtmd-overlay'), '-B', str(build),
                            '-DLCB_LLAMA_SOURCE=' + str(source)], check=True, capture_output=True, text=True)
            commands = (build / 'compile_commands.json').read_text()
            self.assertNotIn(str(build / 'mtmd-tts-overlay'), commands)
            self.assertEqual((stale / 'mtmd-helper.h').read_text(), poison)


if __name__ == '__main__':
    unittest.main()
