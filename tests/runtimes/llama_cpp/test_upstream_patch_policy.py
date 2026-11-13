"""Check exception discoverability and the upstream-only binding/build boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp.sources.mtmd import PATCH_DIRECTORY, prepare
from build_tools.runtimes.llama_cpp.bindings import generate


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
        self.assertIn('Removal', register)
        self.assertFalse((ROOT / 'patches').exists(), 'Do not leave a second generic patch directory')
        self.assertEqual({p.name for p in directory.glob('*.patch')}, {
            'mtmd-webgpu-bf16.patch', 'mtmd-vision-bf16-runtime.patch', 'mtmd-non-pthread-threadpool.patch', 'ggml-webgpu-emscripten-scalar-tensor-writes.patch',
            'ggml-webgpu-same-device-tensor-copy.patch', 'ggml-webgpu-buffer-size-limit.patch',
            'ggml-webgpu-ordered-readback.patch',
            'ggml-webgpu-bind-group-cache.patch', 'ggml-webgpu-host-encode-assembly.patch',
            'ggml-webgpu-scale-unary.patch', 'ggml-webgpu-q8-subgroup-matvec.patch',
            'ggml-webgpu-rms-scale.patch', 'ggml-webgpu-async-tensor-copy.patch',
            'ggml-webgpu-host-input-snapshot.patch',
            'ggml-webgpu-ssm-conv-silu.patch', 'ggml-webgpu-moe-group-skip.patch',
            'ggml-webgpu-q8-diagnostic.patch', 'ggml-webgpu-profile-query-budget.patch', 'ggml-webgpu-f32-small-m.patch', 'ggml-webgpu-moe-gather-logical32.patch',
            'ggml-webgpu-router-tail.patch',
            'ggml-webgpu-iq3s-id-subgroup.patch',
            'ggml-webgpu-iq3s-field-loads.patch', 'ggml-webgpu-iq3s-row-split.patch', 'ggml-webgpu-moe-tight-tile-bound.patch',
            'ggml-webgpu-q6-subgroup-matvec.patch',
            'ggml-webgpu-q4-subgroup-matvec.patch', 'ggml-webgpu-aligned-word-load.patch',
            'ggml-webgpu-subgroup-size-control.patch', 'ggml-webgpu-subgroup-size-control-profile.patch',
            'ggml-webgpu-subgroup-size-control-shared.patch',
            'ggml-webgpu-checkpoint-batch-readback.patch',
            'ggml-webgpu-quant-wide-m.patch',
            'ggml-webgpu-direct-mapped-read.patch',
            'ggml-webgpu-quant-half-pairs.patch',
            'ggml-webgpu-q6-short-workgroup.patch',
            'ggml-webgpu-f32-matvec-epilogue.patch', 'ggml-webgpu-f32-matvec-epilogue-profile.patch',
            'ggml-webgpu-matmul-vec4-layout.patch', 'ggml-webgpu-matmul-vec4-layout-profile.patch',
            'ggml-webgpu-batch-param-uploads.patch', 'ggml-webgpu-moe-expert-sum.patch', 'ggml-webgpu-hc-pre-post.patch',
            'ggml-webgpu-q8-short-workgroup.patch', 'ggml-webgpu-expert-short-workgroup.patch', 'ggml-webgpu-q4-short-workgroup.patch',
            'llama-model-loader-webgpu-chunked-upload.patch',
            'llama-sampler-single-sync.patch', 'common-chat-independent-vocabulary.patch',
            'ggml-webgpu-ssm-conv-single-token.patch', 'qwen4exp-file-backed-rows.patch',
            'qwen4exp-all-pools-selection.patch', 'ggml-webgpu-aligned-word-load-integration.patch',
            'ggml-webgpu-subgroup-size-control-wgsl.patch'})
        for patch in directory.glob('*.patch'):
            self.assertIn(patch.name, register)
        # Source-rewriting and toolchain exceptions are covered too, not only .patch.
        self.assertIn('build_tools/', root_rules)
        self.assertIn('toolchain', rules)
        self.assertIn('build_tools/toolchain/patch_emscripten.py', register)

    def test_cli_resolves_the_renamed_patches_for_both_components(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (source / 'tools/mtmd/mtmd-model.cpp').is_file():
            self.skipTest('Initialize the pinned submodule or set LCB_TEST_LLAMA_SOURCE')
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            before = {p.relative_to(source / 'tools/mtmd'): p.read_bytes()
                      for p in (source / 'tools/mtmd').rglob('*') if p.is_file()}
            for mode in ('cpu', 'webgpu'):
                with self.subTest(mode=mode):
                    subprocess.run([sys.executable, str(ROOT.parents[1] / 'build_tools/runtimes/llama_cpp/sources/mtmd.py'),
                                    '--source', str(source), '--output', str(work / mode),
                                    '--mode', mode], check=True, capture_output=True)
                    reference = prepare(source, work / (mode + '-direct'), mode=mode)
                    files = [*before, Path('mtmd-source-manifest.json')]
                    for name in files:
                        self.assertEqual((work / mode / name).read_bytes(), (reference / name).read_bytes())
            self.assertEqual(before, {p.relative_to(source / 'tools/mtmd'): p.read_bytes()
                                     for p in (source / 'tools/mtmd').rglob('*') if p.is_file()})

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

    def test_cpu_runtime_replaces_only_mtmd_and_the_minimal_clip_plumbing(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not shutil.which('cmake') or not (source / 'CMakeLists.txt').is_file():
            self.skipTest('CMake and an upstream checkout are required')
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / 'cpu-build'
            subprocess.run(['cmake', '-S', str(runtime_test_root('llama-cpp') / 'mtmd-overlay'), '-B', str(build),
                            '-DLCB_WEBGPU=OFF', '-DLCB_LLAMA_SOURCE=' + str(source)],
                           check=True, capture_output=True, text=True)
            sources = (build / 'mtmd-sources.txt').read_text().strip().split(';')
            tree = build / 'mtmd-source-overlay'
            self.assertTrue(all(Path(path).is_relative_to(tree) for path in sources))
            self.assertIn(str(tree / 'mtmd-model.cpp'), sources)
            self.assertIn(str(tree / 'mtmd-subcontext.cpp'), sources)
            self.assertIn(str(tree / 'mtmd.cpp'), sources)
            self.assertNotIn('mtmd-retained-graph.h', (tree / 'mtmd-subcontext.cpp').read_text())
            self.assertFalse((tree / 'clip.h').exists())

    def test_cmake_replaces_only_the_three_accepted_translation_units(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not shutil.which('cmake') or not (source / 'CMakeLists.txt').is_file():
            self.skipTest('CMake and an upstream checkout are required')
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / 'build'
            subprocess.run(['cmake', '-S', str(runtime_test_root('llama-cpp') / 'mtmd-overlay'), '-B', str(build),
                            '-DLCB_LLAMA_SOURCE=' + str(source)], check=True, capture_output=True, text=True)
            before = (build / 'upstream-mtmd-sources.txt').read_text().strip().split(';')
            after = (build / 'mtmd-sources.txt').read_text().strip().split(';')
            self.assertEqual(len(after), len(before))
            tree = build / 'mtmd-source-overlay'
            self.assertEqual([Path(p).relative_to(tree).as_posix() for p in after], before)
            self.assertTrue(all(Path(p).is_file() for p in after))
            self.assertEqual(len(set(after)), len(after))
            manifest = json.loads((tree / 'mtmd-source-manifest.json').read_text())
            self.assertTrue(set(before) <= manifest['preparedFiles'].keys())
            for name, identity in manifest['preparedFiles'].items():
                self.assertEqual(identity, __import__('prepare_mtmd').identity(tree / name))
                if name.startswith('models/'):
                    self.assertEqual(identity, manifest['sourceInputs'][name])
            props = (build / 'mtmd-source-properties.txt').read_text().splitlines()
            unity = {Path(path).relative_to(tree).as_posix(): flag for path, flag in
                     (line.rsplit('|', 1) for line in props)}
            for name in manifest['coreSources']:
                self.assertEqual(unity[name], 'ON', name)
            for name in before:
                if name.startswith('models/') and name.endswith('.cpp'):
                    self.assertNotEqual(unity[name], 'ON', name)
            self.assertEqual((build / 'mtmd-unity-properties.txt').read_text().strip(), 'ON|16')
            commands = (build / 'compile_commands.json').read_text()
            self.assertNotIn(str(source / 'tools/mtmd') + '/mtmd-model.cpp', commands)
            self.assertNotIn(str(source / 'tools/mtmd') + '/mtmd-subcontext.cpp', commands)
            self.assertFalse((tree / 'clip.h').exists())
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
            subprocess.run(['cmake', '-S', str(runtime_test_root('llama-cpp') / 'mtmd-overlay'), '-B', str(build),
                            '-DLCB_LLAMA_SOURCE=' + str(source)], check=True, capture_output=True, text=True)
            commands = (build / 'compile_commands.json').read_text()
            self.assertNotIn(str(build / 'mtmd-tts-overlay'), commands)
            self.assertEqual((stale / 'mtmd-helper.h').read_text(), poison)


if __name__ == '__main__':
    unittest.main()
