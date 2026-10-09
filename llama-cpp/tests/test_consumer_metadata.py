"""Validate compact reporting data independently of real runtime compilation."""
import copy
import hashlib
import json
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from fixture_toolchain import merged_toolchain, seed_toolchain
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import consumer_metadata as consumer
import upstream_provenance as provenance

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40
REPO = 'example/core'
SPEC = f'github:{REPO}#{C}'
NAME = 'llama-cpp-browser-core'
ENTRY = {'version': '0.1.0', 'resolved': f'git+ssh://git@github.com/{REPO}.git#{C}',
         'integrity': 'sha512-fixture-not-a-real-package-hash', 'license': 'MIT'}


def fixture_lock():
    return {'lockfileVersion': 3, 'packages': {'': {'dependencies': {NAME: SPEC}}, 'node_modules/' + NAME: copy.deepcopy(ENTRY)}}


def fixture_package(root):
    manifest = {'formatVersion': 2, 'sourceCommit': A, 'llamaCommit': B, 'profiles': {}, 'files': []}
    def file(path, content):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest['files'].append({'path': path, **provenance.file_identity(target)})
    profiles = ['cpu-wasm32', 'cpu-wasm64', 'webgpu-wasm32-asyncify', 'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi']
    for name in profiles:
        manifest['profiles'][name] = {'variants': {}}
        for variant in ['browser', 'test']:
            for ext in ['mjs', 'wasm', 'd.ts']:
                file(f'profiles/{name}/{variant}/core.{ext}', f'Fixture: {name}/{variant}/{ext}\n'.encode())
            manifest['profiles'][name]['variants'][variant] = {
                'validation': {'compiled': True, 'browserSmoke': True, 'realModelInference': False},
            }
    file('api/schema.json', b'{}\n')
    file('api/functions.d.ts', b'// Fixture\n')
    file('examples/runtime/index.mjs', b'// Fixture\n')
    file('examples/runtime/README.md', b'Test documentation\n')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class LockMetadata(unittest.TestCase):
    def test_lock_entry_is_preserved_including_unknown_npm_fields(self):
        lock = fixture_lock()
        lock['packages']['node_modules/' + NAME]['someFutureNpmField'] = 'value'
        result = consumer.extract_lock(lock, SPEC, C, '0.1.0')
        self.assertEqual(result['packageEntry'], lock['packages']['node_modules/' + NAME])

    def test_absent_integrity_is_not_invented(self):
        lock = fixture_lock()
        del lock['packages']['node_modules/' + NAME]['integrity']
        self.assertNotIn('integrity', consumer.extract_lock(lock, SPEC, C, '0.1.0')['packageEntry'])

    def test_wrong_commit_version_specifier_or_lock_schema_is_rejected(self):
        changes = [lambda x: x.update(lockfileVersion=2),
                   lambda x: x['packages']['']['dependencies'].update({NAME: 'latest'}),
                   lambda x: x['packages']['node_modules/' + NAME].update(version='0.2.0'),
                   lambda x: x['packages']['node_modules/' + NAME].update(resolved='git+https://example/x#' + A)]
        for change in changes:
            value = fixture_lock(); change(value)
            with self.assertRaises(ValueError):
                consumer.extract_lock(value, SPEC, C, '0.1.0')

    def test_transitive_dependencies_or_install_hooks_are_rejected(self):
        for key, value in [('dependencies', {'another': '1'}), ('optionalDependencies', {'another': '1'}),
                           ('peerDependencies', {'another': '1'}), ('hasInstallScript', True)]:
            lock = fixture_lock(); lock['packages']['node_modules/' + NAME][key] = value
            with self.assertRaises(ValueError):
                consumer.extract_lock(lock, SPEC, C, '0.1.0')
        lock = fixture_lock(); lock['packages']['node_modules/another'] = {}
        with self.assertRaises(ValueError):
            consumer.extract_lock(lock, SPEC, C, '0.1.0')

    def test_ci_resolution_uses_npm_with_no_scripts_and_an_isolated_lock(self):
        def npm(command, *, cwd, env, **kwargs):
            self.assertIn('--package-lock-only', command)
            self.assertIn('--ignore-scripts', command)
            self.assertIn('--lockfile-version=3', command)
            self.assertEqual(json.loads((cwd / 'package.json').read_text())['dependencies'], {NAME: SPEC})
            self.assertTrue(str(env['NPM_CONFIG_CACHE']).startswith(str(cwd)))
            (cwd / 'package-lock.json').write_text(json.dumps(fixture_lock()))
        with patch.object(consumer.subprocess, 'run', side_effect=npm), \
             patch.object(consumer.subprocess, 'check_output', side_effect=['v22.16.0\n', '10.9.2\n']):
            result = consumer.generate_lock(REPO, C, '0.1.0')
        self.assertEqual(result['packageEntry'], ENTRY)
        self.assertEqual(result['npmVersion'], '10.9.2')


class ReportMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = fixture_package(self.root)
        lock = consumer.extract_lock(fixture_lock(), SPEC, C, '0.1.0')
        lock['specifier'] = SPEC
        self.data = consumer.metadata(self.root, REPO, C, lock, {'baseCommit': B, 'sourceOverlays': []})

    def test_actual_browser_profiles_and_files_are_used_without_hash_duplication(self):
        self.assertEqual(len(self.data['browserProfiles']), 5)
        for name, files in self.data['browserProfiles'].items():
            for kind in ['mjs', 'wasm', 'types']:
                entry = files[kind]
                self.assertEqual(entry['sha256'], provenance.file_identity(self.root / entry['path'])['sha256'])
                self.assertIn('/browser/', entry['path'])
                self.assertNotIn('/test/', entry['path'])
        paths = [entry['path'] for entry in self.data['interfaceFiles']]
        self.assertIn('api/schema.json', paths)
        self.assertIn('examples/runtime/index.mjs', paths)
        self.assertNotIn('examples/runtime/README.md', paths)

    def test_manifest_itself_is_pinned_outside_the_runtime_package(self):
        self.assertEqual(self.data['retrieval']['manifest']['sha256'], provenance.file_identity(self.root / 'manifest.json')['sha256'])
        for name in ['artifactArchive', 'artifactRawBase']:
            self.assertIn(C, self.data['retrieval'][name])
        for name in ['sourceArchive', 'sourceRawBase']:
            self.assertIn(A, self.data['retrieval'][name])
        self.assertIn(B, self.data['retrieval']['upstreamRawBase'])

    def test_overlay_base_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            consumer.metadata(self.root, REPO, C, {}, {'baseCommit': A})

    def test_duplicate_manifest_file_is_rejected(self):
        self.manifest['files'].append(self.manifest['files'][0])
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            consumer.metadata(self.root, REPO, C, {}, {'baseCommit': B})

    def test_one_yaml_contains_all_pin_locations_and_search_hints(self):
        text = consumer.render_yaml(self.data)
        for term in ['package.json', 'package-lock.json', 'coreHashes', 'transformBrowserCore',
                     'Reviewed browser variant artifact commit', 'standaloneWasm', 'manifestSchema',
                     'profileSchema', 'upstreamDivergences', 'knownLocations', 'searchHints',
                     'build/llama-cpp-browser-core.test.ts', 'Brotli capability-probe']:
            self.assertIn(term, text)
        self.assertIn('may have changed', text)
        self.assertNotIn('Policy exception', text)
        self.assertNotIn('must not normally', text)
        self.assertNotIn('ChatGPT', text)
        self.assertNotIn('currentValue', text)
        self.assertLess(len(text), 22000)

    def test_report_is_english_and_descriptive_not_an_assistant_prompt(self):
        text = consumer.render_yaml(self.data)
        self.assertTrue(text.isascii())
        for command in ['Do not ', 'Update this', 'Replace this', 'You must', 'Search Naidan']:
            self.assertNotIn(command, text)

    def test_report_and_envelope_match_and_have_one_collapsed_yaml_block(self):
        output = self.root / 'report'
        text = consumer.write_report(output, self.data, '123', '2')
        self.assertEqual(text, (output / 'consumer-update.md').read_text())
        self.assertEqual(text.count('```yaml'), 1)
        self.assertIn('<details>', text)
        envelope = json.loads((output / 'report.json').read_text())
        self.assertEqual(envelope['runId'], 123)
        self.assertEqual(envelope['runAttempt'], 2)
        self.assertEqual(envelope['markdownSha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(set(path.name for path in output.iterdir()), {'consumer-update.md', 'consumer-update.yaml', 'report.json'})

    def test_oversized_report_fails_instead_of_truncating(self):
        self.data['huge'] = 'x' * 55000
        with self.assertRaisesRegex(ValueError, 'too large'):
            consumer.write_report(self.root / 'report', self.data, '1', '1')

    def test_yaml_ambiguous_scalars_are_quoted(self):
        for value in ['on', 'OFF', 'null', 'yes', '2026-09-22', '1234', '123e9', 'a: b', 'x\ny', 'value # comment', 'abc:']:
            self.assertEqual(consumer.scalar(value), json.dumps(value))
        self.assertEqual(consumer.scalar('profileSchema'), 'profileSchema')
        self.assertEqual(consumer.scalar(True), 'true')
        self.assertEqual(consumer.scalar(False), 'false')
        self.assertEqual(consumer.scalar(0), '0')
        self.assertEqual(consumer.scalar(None), 'null')

    def test_yaml_round_trip_when_parser_available(self):
        try:
            import yaml
        except ImportError:
            self.skipTest('Optional local YAML round-trip check; generator has no third-party dependencies')
        self.assertEqual(yaml.safe_load(consumer.render_yaml(self.data)), self.data)


class OverlayProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'llama-cpp'
        self.vendor = self.root / 'vendor/llama.cpp'
        (self.vendor / 'tools/mtmd').mkdir(parents=True)
        (self.vendor / 'tools/mtmd/clip.cpp').write_text('before\noriginal\nafter\n')
        (self.vendor / 'tools/mtmd/mtmd-audio.cpp').write_text('before\noriginal\nafter\n')
        toolchain = {'llamaCommit': B, 'emscriptenRelease': A,
                     'emscriptenAsyncifyBigIntPatch': {'sourceSha256': '1' * 64, 'patchedSha256': '2' * 64}}
        for name in ['config', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval', 'scripts', 'cmake', 'bridge', 'docs']:
            (self.root / name).mkdir()
        seed_toolchain(self.root, toolchain, scripts=True)
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch').write_text('--- a/clip.cpp\n+++ b/clip.cpp\n@@ -1,3 +1,3 @@\n before\n-original\n+patched\n after\n')
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch').write_text((self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch').read_text().replace('clip.cpp', 'mtmd-audio.cpp'))
        shaders = self.vendor / provenance.moe.SHADER_DIRECTORY
        shaders.mkdir(parents=True)
        # Metadata tests use synthetic inputs, independently of vendor availability.
        (self.vendor / provenance.moe.SHADER_PATH).write_text('Provenance shader fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.moe.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.moe.PATCH_NAME)
        (self.vendor / provenance.tensor_copy.SOURCE_PATH).write_text('Copy provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.tensor_copy.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.tensor_copy.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.PARAM_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.PARAM_PATCH_NAME)
        (self.vendor / provenance.webgpu_source.SSM_HEADER_PATH).write_text('SSM header provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.SSM_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.SSM_PATCH_NAME)
        (self.vendor / 'src').mkdir()
        (self.vendor / provenance.webgpu_source.LOADER_PATH).write_text('Loader provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.LOADER_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.LOADER_PATCH_NAME)
        for path in ['scripts/prepare_webgpu_tensor_copy.py', 'cmake/WebgpuSourceOverlay.cmake', 'scripts/prepare_webgpu_source.py', 'docs/webgpu-param-upload-batching.md',
                     'docs/webgpu-tensor-copy.md', 'scripts/prepare_moe_direct_slot.py', 'cmake/MoeDirectSlotOverlay.cmake',
                     'docs/moe-direct-slot.md', 'scripts/prepare_mtmd.py', 'cmake/MtmdOverlay.cmake', 'bridge/mtmd-bf16.h',
                     'docs/webgpu-bf16-projector.md',
                     'cmake/MtmdAudioOverlay.cmake', 'docs/audio-single-thread.md']:
            (self.root / path).write_text('Provenance fixture: ' + path + '\n')
        self.manifest = {'sourceCommit': A, 'llamaCommit': B, 'profiles': {}}
        for name, enabled in [('cpu-wasm32', False), ('webgpu-wasm64-jspi', True)]:
            self.manifest['profiles'][name] = {'variants': {variant: {
                'toolchain': toolchain, 'cmakeCommand': ['cmake', '-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if enabled else 'OFF'),
                    '-DLCB_WEBGPU_MOE_DIRECT_SLOT=OFF', '-DLCB_WEBGPU=OFF'],
            } for variant in ['browser', 'test']}}
        self.prepared = self.root / 'prepared.cpp'
        self.prepared.write_bytes(b'combined source fixture')
        (self.prepared.parent / 'llama-model-loader.cpp').write_bytes(b'loader source fixture')
        (self.prepared.parent / 'ggml-webgpu-shader-lib.hpp').write_bytes(b'SSM header source fixture')
        source_prepare = patch.object(provenance.webgpu_source, 'prepare', return_value=self.prepared)
        source_prepare.start()
        self.addCleanup(source_prepare.stop)
        source_verify = patch.object(provenance.tensor_copy, 'verify_reviewed_source', return_value=B)
        source_verify.start()
        self.addCleanup(source_verify.stop)
        self.mock_git = patch.object(provenance, 'git', side_effect=lambda *args, cwd: subprocess.CompletedProcess(
            args, 0, stdout=(A if cwd == self.root else B) + '\n'))
        self.mock_git.start()
        self.addCleanup(self.mock_git.stop)

    def test_overlay_hashes_describe_pristine_and_actual_patched_copy(self):
        result = provenance.collect(self.root, self.manifest)
        overlay = result['sourceOverlays'][0]
        self.assertFalse(result['vendorCheckoutModified'])
        self.assertEqual(overlay['compiledCopy']['sha256'], hashlib.sha256(b'before\npatched\nafter\n').hexdigest())
        self.assertEqual(overlay['upstreamSource']['sha256'], hashlib.sha256(b'before\noriginal\nafter\n').hexdigest())
        self.assertEqual(overlay['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertEqual((self.vendor / 'tools/mtmd/clip.cpp').read_text(), 'before\noriginal\nafter\n')
        self.assertIn('toolchainDivergences', result)

    def test_audio_overlay_tracks_both_backends_and_actual_compiled_bytes(self):
        result = provenance.collect(self.root, self.manifest)
        audio = next(item for item in result['sourceOverlays'] if item['id'] == 'single-thread-wasm-audio-preprocessing')
        self.assertEqual(audio['compiledCopy']['sha256'], hashlib.sha256(b'before\npatched\nafter\n').hexdigest())
        self.assertEqual(audio['application']['enabledProfileVariants'], [
            'cpu-wasm32/browser', 'cpu-wasm32/test', 'webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertNotIn('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch', result['otherPatchFiles'])
        self.assertEqual((self.vendor / 'tools/mtmd/mtmd-audio.cpp').read_text(), 'before\noriginal\nafter\n')

    def test_audio_overlay_conflict_is_not_hidden_by_a_successful_vision_overlay(self):
        (self.vendor / 'tools/mtmd/mtmd-audio.cpp').write_text('changed upstream\n')
        with self.assertRaises(subprocess.CalledProcessError):
            provenance.collect(self.root, self.manifest)

    def test_unknown_patch_files_are_not_silently_omitted(self):
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch').write_text('Another fixture patch\n')
        report = provenance.collect(self.root, self.manifest)
        self.assertIn('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch', report['otherPatchFiles'])
        self.assertEqual(report['otherPatchFiles']['upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch']['application'], 'not classified by this report')

    def test_retained_overlays_are_independent_and_use_only_the_renamed_directory(self):
        report = provenance.collect(self.root, self.manifest)
        self.assertEqual({item['id'] for item in report['sourceOverlays']}, {
            'webgpu-vision-bf16-projector', 'single-thread-wasm-audio-preprocessing',
            'experimental-webgpu-moe-direct-slot', 'webgpu-same-device-tensor-copy',
            'webgpu-parameter-upload-batching', 'webgpu-chunked-model-upload', 'webgpu-ssm-conv-single-token'})
        self.assertIn(provenance.PATCH_DIRECTORY, report['inventoryScope'])
        for item in report['sourceOverlays']:
            self.assertTrue(item['patch']['path'].startswith(provenance.PATCH_DIRECTORY + '/'))
            self.assertNotIn('inputOverlay', item['application'])
        self.assertFalse((self.root / 'patches').exists())

    def test_moe_disabled_provenance_keeps_patch_identity_without_preparing(self):
        with patch.object(provenance.moe, 'prepare') as prepare:
            report = provenance.collect(self.root, self.manifest)
        prepare.assert_not_called()
        item = next(entry for entry in report['sourceOverlays']
                    if entry['id'] == 'experimental-webgpu-moe-direct-slot')
        self.assertIsNone(item['compiledCopy'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertEqual(item['patch']['sha256'], provenance.moe.PATCH_SHA256)
        self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

    def test_moe_provenance_selects_the_built_upstream_revision(self):
        for revision, inputs in provenance.moe.REVIEWED_REVISIONS.items():
            with self.subTest(revision=revision):
                manifest = copy.deepcopy(self.manifest)
                manifest['llamaCommit'] = revision
                with patch.object(provenance, 'git', side_effect=lambda *args, cwd: subprocess.CompletedProcess(
                        args, 0, stdout=(A if cwd == self.root else revision) + '\n')):
                    report = provenance.collect(self.root, manifest)
                item = next(entry for entry in report['sourceOverlays']
                            if entry['id'] == 'experimental-webgpu-moe-direct-slot')
                self.assertEqual(item['reviewedCommit'], revision)
                self.assertEqual(item['reviewedInputs'], inputs)

    def test_moe_enabled_provenance_records_actual_prepared_bytes(self):
        variant = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        variant['cmakeCommand'] = [arg.replace('MOE_DIRECT_SLOT=OFF', 'MOE_DIRECT_SLOT=ON').replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON')
                                   for arg in variant['cmakeCommand']]
        shader = self.root / 'shader'; shader.write_bytes(b'prepared shader')
        header = self.root / 'header'; header.write_bytes(b'prepared header')
        with patch.object(provenance.moe, 'prepare', return_value={'shader': shader, 'header': header}) as prepare:
            report = provenance.collect(self.root, self.manifest)
        prepare.assert_called_once()
        item = next(entry for entry in report['sourceOverlays']
                    if entry['id'] == 'experimental-webgpu-moe-direct-slot')
        self.assertEqual(item['compiledCopy']['header'], provenance.file_identity(header))
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])

    def test_moe_activation_follows_every_configured_webgpu_profile(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        shader = self.root / 'shader'; shader.write_bytes(b'prepared shader')
        header = self.root / 'header'; header.write_bytes(b'prepared header')
        template = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        for profile, config in profiles.items():
            with self.subTest(profile=profile):
                manifest = copy.deepcopy(self.manifest)
                variants = {name: copy.deepcopy(template) for name in ('browser', 'test')}
                for variant in variants.values():
                    variant['cmakeCommand'] = [
                        'cmake', '-DLCB_WEBGPU_BF16_PROJECTOR=OFF',
                        '-DLCB_WEBGPU=' + ('ON' if config['webgpu'] else 'OFF'),
                        '-DLCB_WEBGPU_MOE_DIRECT_SLOT=' + ('ON' if config['webgpu'] else 'OFF'),
                    ]
                manifest['profiles'] = {profile: {'variants': variants}}
                with patch.object(provenance.moe, 'prepare', return_value={'shader': shader, 'header': header}) as prepare:
                    report = provenance.collect(self.root, manifest)
                item = next(entry for entry in report['sourceOverlays']
                            if entry['id'] == 'experimental-webgpu-moe-direct-slot')
                self.assertEqual(item['application']['enabledProfileVariants'],
                                 [profile + '/browser', profile + '/test'] if config['webgpu'] else [])
                self.assertEqual(prepare.call_count, 1 if config['webgpu'] else 0)

    def test_moe_activation_fails_closed_for_missing_duplicate_and_cpu_flags(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([arg for arg in original if 'MOE_DIRECT_SLOT' not in arg],
                        original + ['-DLCB_WEBGPU_MOE_DIRECT_SLOT=OFF'],
                        [arg.replace('MOE_DIRECT_SLOT=OFF', 'MOE_DIRECT_SLOT=ON') for arg in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'MoE'):
                variant['cmakeCommand'] = command
                provenance.collect(self.root, self.manifest)

    def test_cpu_inventories_patches_without_preparing_source(self):
        with patch.object(provenance.webgpu_source, 'prepare') as prepare:
            report = provenance.collect(self.root, self.manifest)
        prepare.assert_not_called()
        for suffix in ('same-device-tensor-copy', 'parameter-upload-batching', 'chunked-model-upload', 'ssm-conv-single-token'):
            item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-' + suffix)
            self.assertIsNone(item['compiledCopy'])
            self.assertEqual(item['application']['enabledProfileVariants'], [])
            self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

    def test_webgpu_provenance_records_combined_bytes_for_both_patches(self):
        for variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            variant['cmakeCommand'] = [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for a in variant['cmakeCommand']]
        report = provenance.collect(self.root, self.manifest)
        for suffix in ('same-device-tensor-copy', 'parameter-upload-batching'):
            item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-' + suffix)
            self.assertEqual(item['compiledCopy']['sha256'], provenance.file_identity(self.prepared)['sha256'])
            self.assertEqual(item['compiledCopy']['compileDefinitions'], ['GGML_WEBGPU_BATCH_PARAM_UPLOADS'])
            self.assertEqual(item['application']['option'], 'LCB_WEBGPU')
            self.assertEqual(item['application']['enabledProfileVariants'],
                             ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
            self.assertEqual(len(item['compiledCopiesByProfileVariant']), 2)

    def test_webgpu_loader_provenance_records_separate_compiled_source(self):
        for variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            variant['cmakeCommand'] = [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for a in variant['cmakeCommand']]
        report = provenance.collect(self.root, self.manifest)
        loader = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-chunked-model-upload')
        self.assertEqual(loader['compiledCopy']['sha256'],
                         provenance.file_identity(self.prepared.parent / 'llama-model-loader.cpp')['sha256'])
        self.assertEqual(loader['application']['enabledProfileVariants'],
                         ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertNotIn(loader['patch']['path'], report['otherPatchFiles'])

    def test_webgpu_ssm_header_provenance_preserves_independent_cpp_identity(self):
        for variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            variant['cmakeCommand'] = [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for a in variant['cmakeCommand']]
        report = provenance.collect(self.root, self.manifest)
        item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-ssm-conv-single-token')
        self.assertEqual(item['compiledCopy']['sha256'], provenance.file_identity(self.prepared.parent / 'ggml-webgpu-shader-lib.hpp')['sha256'])
        self.assertEqual(item['compiledCopy']['logicalUpstreamPath'], provenance.webgpu_source.SSM_HEADER_PATH)
        self.assertEqual(item['reviewedInputs'][provenance.webgpu_source.SSM_SHADER_PATH], provenance.webgpu_source.SSM_SHADER_SHA256)
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertEqual(len(item['compiledCopiesByProfileVariant']), 2)
        self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])
        original = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-parameter-upload-batching')
        self.assertEqual(original['compiledCopy']['sha256'], provenance.file_identity(self.prepared)['sha256'])

    def test_webgpu_activation_fails_closed(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([a for a in original if not a.startswith('-DLCB_WEBGPU=')],
                        original + ['-DLCB_WEBGPU=OFF'],
                        original + ['-DLCB_WEBGPU:BOOL=ON'],
                        [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=MAYBE') for a in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'WebGPU'):
                variant['cmakeCommand'] = command
                provenance.collect(self.root, self.manifest)

    def test_nested_patch_is_inventoried_with_its_exact_identity(self):
        relative = provenance.PATCH_DIRECTORY + '/nested/example.patch'
        extra = self.root / relative
        extra.parent.mkdir()
        extra.write_bytes(b'Not an applied exception; inventory fixture only.\n')
        item = provenance.collect(self.root, self.manifest)['otherPatchFiles'][relative]
        self.assertEqual(item['sha256'], hashlib.sha256(extra.read_bytes()).hexdigest())
        self.assertEqual(item['bytes'], extra.stat().st_size)
        self.assertEqual(item['application'], 'not classified by this report')

    def test_unknown_build_activation_fails(self):
        self.manifest['profiles']['cpu-wasm32']['variants']['browser']['cmakeCommand'] = ['cmake']
        with self.assertRaisesRegex(ValueError, 'activation'):
            provenance.collect(self.root, self.manifest)

    def test_patch_conflict_and_source_commit_mismatch_fail(self):
        self.manifest['sourceCommit'] = C
        with self.assertRaisesRegex(ValueError, 'source commit'):
            provenance.collect(self.root, self.manifest)
        self.manifest['sourceCommit'] = A
        (self.vendor / 'tools/mtmd/clip.cpp').write_text('changed upstream\n')
        with self.assertRaises(subprocess.CalledProcessError):
            provenance.collect(self.root, self.manifest)

    def test_symlinked_source_files_are_rejected(self):
        source = self.vendor / 'tools/mtmd/clip.cpp'
        copied = self.root / 'elsewhere'
        source.rename(copied)
        source.symlink_to(copied)
        with self.assertRaisesRegex(ValueError, 'linked'):
            provenance.collect(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()
