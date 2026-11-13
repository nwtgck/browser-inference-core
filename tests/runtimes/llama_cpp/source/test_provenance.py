"""Observed WebGPU preparation metadata, with real source generation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root
from build_tools.runtimes.llama_cpp import webgpu_plan as plan, webgpu_provenance as reporting

ROOT = runtime_root('llama-cpp')


class FeatureCatalog(unittest.TestCase):
    def test_every_accepted_patch_has_exactly_one_documented_feature(self):
        features = reporting.features(ROOT)
        self.assertEqual(len(features), 42)
        names = [name for feature in features for name in feature['patches'].values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), reporting.patch_names(ROOT))
        for feature in features:
            self.assertTrue(feature['behavior'])
            for name in feature['documentation']:
                self.assertTrue((ROOT / name).is_file())

    def test_missing_duplicate_unknown_and_escaping_catalog_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'config', root / 'config')
            path = root / 'config/webgpu-features.json'
            original = json.loads(path.read_text())
            cases = []
            missing = copy.deepcopy(original); missing['features'].pop(); cases.append(missing)
            duplicate = copy.deepcopy(original); duplicate['features'].append(duplicate['features'][0]); cases.append(duplicate)
            unknown = copy.deepcopy(original); unknown['features'][0]['patches']['patch'] = '../unapproved.patch'; cases.append(unknown)
            escaping = copy.deepcopy(original); escaping['features'][0]['documentation'] = ['docs/../../outside.md']; cases.append(escaping)
            bool_format = copy.deepcopy(original); bool_format['formatVersion'] = True; cases.append(bool_format)
            for value in cases:
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    reporting.features(root)

    def test_duplicate_json_key_cannot_hide_a_feature_declaration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'config', root / 'config')
            path = root / 'config/webgpu-features.json'
            original = path.read_text()
            path.write_text(original.replace('"formatVersion": 1',
                                             '"formatVersion": 0, "formatVersion": 1'))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                reporting.features(root)

    def test_buffer_limit_and_scalar_fallback_are_separate_and_platform_guarded(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(reporting.source, 'prepare', side_effect=AssertionError('unexpected preparation')):
            _, items, _ = reporting.describe(ROOT, ROOT / 'vendor/llama.cpp', Path(temporary), [], [])
        active = {item['id']: item for item in items}
        for key, name in (
            ('webgpu-buffer-size-limit', 'ggml-webgpu-buffer-size-limit.patch'),
            ('webgpu-batched-write-scalar-fallback', 'ggml-webgpu-emscripten-scalar-tensor-writes.patch'),
        ):
            self.assertEqual(active[key]['patch']['sha256'], plan.identity(ROOT / plan.PATCH_DIRECTORY / name)['sha256'])
            self.assertIsNone(active[key]['compiledCopies'])
            self.assertEqual(active[key]['application']['enabledProfileVariants'], [])
        self.assertEqual(active['webgpu-batched-write-scalar-fallback']['application']['compileGuard'],
                         'defined(__EMSCRIPTEN__)')


class ActualPreparationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (cls.source / plan.WGSL / 'embed_wgsl.py').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE to a local upstream tree')
        temp = tempfile.TemporaryDirectory(prefix='source-provenance-')
        cls.addClassCleanup(temp.cleanup)
        cls.work = Path(temp.name)
        cls.details, cls.overlays, cls.files = reporting.describe(
            ROOT, cls.source, cls.work, ['webgpu/test', 'profile/test'], ['profile/test'])

    def test_final_composition_and_patch_receipts_match_observed_bytes(self):
        for key, mode in self.details['profileVariants'].items():
            prepared = self.work / ('webgpu-' + mode)
            observed = {path.name: plan.identity(path) for path in prepared.iterdir()}
            self.assertEqual(self.files[key], observed)
            entry = self.details['modes'][mode]
            self.assertEqual(entry['outputs'], {name: value for name, value in observed.items() if name != plan.RECEIPT})
            self.assertEqual(entry['receipt']['sha256'], observed[plan.RECEIPT]['sha256'])
            self.assertEqual(len(entry['shaders']), 13)
            expected_names = (*plan.patch_names(mode == 'profile'),
                'ggml-webgpu-iq3s-field-loads.patch', 'ggml-webgpu-iq3s-row-split.patch',
                'ggml-webgpu-aligned-word-load.patch')
            self.assertEqual([entry['file'] for entry in entry['patches']], list(expected_names))
            for record in entry['patches']:
                self.assertEqual(record['sha256'], plan.identity(ROOT / plan.PATCH_DIRECTORY / record['file'])['sha256'])

    def test_all_features_use_the_final_mode_not_historical_stage_hashes(self):
        catalog = {item['id']: item for item in reporting.features(ROOT)}
        for item in self.overlays:
            self.assertEqual(item['behavior'], catalog[item['id']]['behavior'])
            self.assertNotIn('reviewedInputs', item)
            self.assertNotIn('prefixInputs', item)
            self.assertNotIn('finalOutputs', item)
            self.assertNotIn('intermediateEmbeddedHeaderSha256', item)
            for key, mode in item['preparationModesByProfileVariant'].items():
                self.assertEqual(item['compiledCopiesByProfileVariant'][key], self.details['modes'][mode]['outputs'])
                self.assertEqual(item['compiledCopiesByProfileVariant'][key]['ggml-webgpu.cpp'],
                                 self.files[key]['ggml-webgpu.cpp'])
            if len(item['preparationModesByProfileVariant']) == 2:
                self.assertIsNone(item['compiledCopies'])
        self.assertNotEqual(self.files['webgpu/test']['ggml-webgpu.cpp'], self.files['profile/test']['ggml-webgpu.cpp'])

    def test_profile_instrumentation_and_owned_hash_code_are_not_attributed_to_normal(self):
        for id in ['webgpu-q8-kernel-diagnostic-v1', 'webgpu-profile-query-budget']:
            item = next(item for item in self.overlays if item['id'] == id)
            self.assertEqual(item['application']['enabledProfileVariants'], ['profile/test'])
            self.assertEqual(item['preparationModesByProfileVariant'], {'profile/test': 'profile'})
        name = 'ggml-webgpu-q8-diagnostic-sha256.hpp'
        self.assertNotIn(name, self.files['webgpu/test'])
        self.assertEqual(self.files['profile/test'][name], plan.identity(ROOT / 'native/diagnostics' / name))
        self.assertFalse(any('ggml-opencl' in path for mode in self.details['modes'].values() for path in mode['inputs']))

    def test_supporting_inputs_are_actual_and_no_old_stage_runner_is_claimed(self):
        implementation = self.details['implementation']
        self.assertIn('build_tools/runtimes/llama_cpp/webgpu_plan.py', implementation)
        for path, recorded in implementation.items():
            self.assertEqual(recorded, reporting.supporting_identity(ROOT, path))
            self.assertNotIn('/sources/', path)
        for item in self.overlays:
            for role, filename in next(feature for feature in reporting.features(ROOT) if feature['id'] == item['id'])['patches'].items():
                self.assertEqual(item[role]['sha256'], plan.identity(ROOT / plan.PATCH_DIRECTORY / filename)['sha256'])

    def test_cpu_only_report_does_not_prepare_and_still_inventories_every_feature(self):
        with patch.object(reporting.source, 'prepare', side_effect=AssertionError('unexpected preparation')):
            details, overlays, files = reporting.describe(ROOT, self.source, self.work / 'cpu', [], [])
        self.assertEqual(details['modes'], {})
        self.assertEqual(files, {})
        self.assertEqual(len(overlays), 42)
        for item in overlays:
            self.assertEqual(item['application']['enabledProfileVariants'], [])
            self.assertIsNone(item['compiledCopies'])

    def test_profile_only_build_does_not_prepare_unused_normal_variant(self):
        with patch.object(reporting.source, 'prepare', return_value=self.work / 'webgpu-profile/ggml-webgpu.cpp') as prepare:
            details, _, files = reporting.describe(ROOT, self.source, self.work, ['profile/test'], ['profile/test'])
        prepare.assert_called_once()
        self.assertEqual(set(details['modes']), {'profile'})
        self.assertEqual(set(files), {'profile/test'})

    def test_changed_preparation_output_is_not_reported_under_stale_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / 'prepared'
            shutil.copytree(self.work / 'webgpu-normal', prepared)
            (prepared / 'ggml-webgpu.cpp').write_text('changed after source preparation')
            with patch.object(reporting.source, 'prepare', return_value=prepared / 'ggml-webgpu.cpp'), \
                 self.assertRaisesRegex(ValueError, 'receipt differs'):
                reporting.describe(ROOT, self.source, self.work, ['normal/test'], [])

    def test_buffer_size_limit_is_independent_and_uses_final_composed_bytes(self):
        for key in ('webgpu-buffer-size-limit', 'webgpu-batched-write-scalar-fallback'):
            item = next(row for row in self.overlays if row['id'] == key)
            self.assertEqual(item['application']['enabledProfileVariants'], ['profile/test', 'webgpu/test'])
            for variant, mode in self.details['profileVariants'].items():
                self.assertEqual(item['compiledCopiesByProfileVariant'][variant],
                                 self.details['modes'][mode]['outputs'])
                self.assertEqual(item['compiledCopiesByProfileVariant'][variant]['ggml-webgpu.cpp'],
                                 self.files[variant]['ggml-webgpu.cpp'])
            self.assertNotIn('finalOutputs', item)  # No historical stage hash substituted.


if __name__ == '__main__':
    unittest.main()
