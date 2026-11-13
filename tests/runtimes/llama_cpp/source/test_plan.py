"""Exercise the production source plan, not historical intermediate byte pins.

Real upstream sources are required. These are preparation/host compilation tests,
not evidence of WebGPU execution or full-model inference.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root
from build_tools.runtimes.llama_cpp import source, webgpu_plan as plan
from build_tools.source.patches import read_series

ROOT = runtime_root('llama-cpp')


def contents(root):
    return {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}


class SourcePlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (cls.upstream / 'include/llama.h').is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        temporary = tempfile.TemporaryDirectory(prefix='webgpu-plan-tests-')
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name)
        cls.normal = cls.work / 'normal'
        cls.profile = cls.work / 'profile'
        source.prepare(cls.upstream, cls.normal)
        source.prepare(cls.upstream, cls.profile, gpu_profile=True)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='webgpu-plan-case-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def private_source(self):
        # Only paths actually captured by production are needed for preparation.
        receipt = json.loads((self.normal / plan.RECEIPT).read_text())
        root = self.directory / 'upstream'
        for name in receipt['inputs']:
            if name.startswith('runtime/'):
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.upstream / name, target)
        return root

    def test_production_entrypoint_does_not_export_historical_stage_identities(self):
        self.assertIs(source.prepare, plan.prepare)
        for name in ('FINAL_OUTPUTS', 'OUTPUT_SHA256', 'SSM_HEADER_OUTPUT_SHA256',
                     'q4_short', 'q8_subgroup', 'aligned_word_load'):
            self.assertFalse(hasattr(source, name), name)

    def test_selected_patch_order_scope_and_every_output_are_observed(self):
        for profiled, output in ((False, self.normal), (True, self.profile)):
            with self.subTest(profiled=profiled):
                receipt = json.loads((output / plan.RECEIPT).read_text())
                self.assertEqual(receipt['gpuProfile'], profiled)
                records = receipt['patches']
                self.assertEqual([p['file'] for p in records if p['target'] == 'llama-webgpu'],
                                 list(plan.patch_names(profiled)))
                self.assertEqual([p['file'] for p in records if p['target'] == 'iq3s'],
                                 ['ggml-webgpu-iq3s-field-loads.patch', 'ggml-webgpu-iq3s-row-split.patch'])
                self.assertEqual(set(receipt['outputs']), set(contents(output)) - {plan.RECEIPT})
                for name, value in receipt['outputs'].items():
                    raw = (output / name).read_bytes()
                    self.assertEqual(value, {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
                for record in records:
                    original = ROOT / plan.PATCH_DIRECTORY / record['file']
                    self.assertEqual(record['sha256'], hashlib.sha256(original.read_bytes()).hexdigest())
                self.assertEqual(len(receipt['shaders']), 13)
                self.assertNotIn('ggml-wgsl-shaders.hpp', receipt['outputs'])

    def test_profile_only_diagnostic_is_absent_from_normal_outputs(self):
        for name in ('ggml-webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic-sha256.hpp'):
            self.assertFalse((self.normal / name).exists())
            self.assertTrue((self.profile / name).is_file())
        for header in self.normal.glob('*-shaders.hpp'):
            self.assertEqual(header.read_bytes(), (self.profile / header.name).read_bytes())
        self.assertNotIn('ggml/src/ggml-opencl/cl-program-cache.cpp',
                         json.loads((self.profile / plan.RECEIPT).read_text())['inputs'])

    def test_harmless_source_comments_do_not_require_global_hash_refresh(self):
        upstream = self.private_source()
        backend = upstream / plan.BASE / 'ggml-webgpu.cpp'
        backend.write_bytes(backend.read_bytes() + b'\n// unrelated upstream documentation\n')
        output = self.directory / 'out'
        source.prepare(upstream, output)
        self.assertEqual((output / 'ggml-webgpu.cpp').read_bytes(),
                         (self.normal / 'ggml-webgpu.cpp').read_bytes() + b'\n// unrelated upstream documentation\n')
        for name in contents(self.normal):
            if name not in ('ggml-webgpu.cpp', plan.RECEIPT):
                self.assertEqual((output / name).read_bytes(), (self.normal / name).read_bytes(), name)

    def test_each_shader_is_generated_once_and_from_one_captured_source(self):
        upstream = self.private_source()
        generator = upstream / plan.WGSL / 'embed_wgsl.py'
        original_generate = plan.generate
        calls = []
        def changing_producer(*args, **kwargs):
            calls.append(Path(args[2]).name)
            if len(calls) == 1:
                generator.write_text('raise RuntimeError("producer changed after capture")\n')
            return original_generate(*args, **kwargs)
        output = self.directory / 'out'
        with patch.object(plan, 'generate', side_effect=changing_producer):
            source.prepare(upstream, output)
        self.assertEqual(len(calls), 13)
        self.assertEqual(len(set(calls)), 13)
        self.assertEqual(contents(output), contents(self.normal))

    def test_patch_outside_its_declared_scope_fails_without_publishing(self):
        patches = self.directory / 'patches'
        shutil.copytree(ROOT / plan.PATCH_DIRECTORY, patches)
        first = patches / plan.PREFIX[0]
        first.write_text(first.read_text() + '\n--- /dev/null\n+++ b/undeclared.txt\n@@ -0,0 +1 @@\n+unexpected\n')
        output = self.directory / 'out'
        shutil.copytree(self.normal, output)
        before = contents(output)
        with self.assertRaisesRegex(ValueError, 'outside its inventory'):
            plan.prepare(self.upstream, output, patch_root=patches)
        self.assertEqual(contents(output), before)

    def test_changed_patch_hunk_and_inventory_are_not_ignored(self):
        patches = self.directory / 'patches'
        shutil.copytree(ROOT / plan.PATCH_DIRECTORY, patches)
        first = patches / plan.PREFIX[0]
        first.write_text('--- a/ggml/src/ggml-webgpu/ggml-webgpu.cpp\n'
                         '+++ b/ggml/src/ggml-webgpu/ggml-webgpu.cpp\n'
                         '@@ -1 +1 @@\n-impossible source line\n+unexpected\n')
        with self.assertRaisesRegex(ValueError, 'Patch application failed'):
            plan.prepare(self.upstream, self.directory / 'bad', patch_root=patches)
        self.assertFalse((self.directory / 'bad').exists())
        inventory = json.loads(plan.INVENTORY.read_text())
        inventory['patches'].pop()
        path = self.directory / 'inventory.json'
        path.write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, 'differs from the explicit'):
            plan.prepare(self.upstream, self.directory / 'bad', inventory=path)

    def test_late_generation_failure_retains_previous_profile_and_mtimes(self):
        output = self.directory / 'out'
        shutil.copytree(self.profile, output)
        before = contents(output)
        stamps = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}
        real = plan.generate
        count = 0
        def fail_last(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 13:
                raise RuntimeError('final canonical generation failed')
            return real(*args, **kwargs)
        with patch.object(plan, 'generate', side_effect=fail_last):
            with self.assertRaisesRegex(RuntimeError, 'final canonical'):
                source.prepare(self.upstream, output)
        self.assertEqual(contents(output), before)
        self.assertEqual({p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}, stamps)
        source.prepare(self.upstream, output)
        self.assertEqual(contents(output), contents(self.normal))
        stamps = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
        source.prepare(self.upstream, output)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in output.iterdir()}, stamps)

    def test_overlap_symlink_and_unexpected_canonical_headers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            source.prepare(self.upstream, self.upstream / 'private-build')
        source_link = self.directory / 'linked'
        source_link.symlink_to(self.upstream, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            source.prepare(source_link, self.directory / 'out')
        output = self.directory / 'out'
        output.mkdir()
        (output / 'ggml-wgsl-shaders.hpp').write_text('stale duplicate canonical symbols')
        with self.assertRaisesRegex(ValueError, 'shadow upstream'):
            source.prepare(self.upstream, output)


class PlanInventory(unittest.TestCase):
    def test_accepted_patches_have_no_generated_shader_body_edits(self):
        series = read_series(plan.INVENTORY, targets=('llama-webgpu', 'iq3s', 'word-template'))
        self.assertEqual(len(series.patches), len(set(p.file for p in series.patches)))
        for definition in series.patches:
            for name in definition.files:
                self.assertFalse(name.endswith('-shaders.hpp'), (definition.file, name))
