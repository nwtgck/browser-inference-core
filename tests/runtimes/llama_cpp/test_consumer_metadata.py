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
from tests.support.llama_toolchain import merged_toolchain, seed_toolchain
from unittest.mock import patch
from tests.runtimes.llama_cpp.mtmd_fixture_support import CheckedMtmdFixture

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
from build_tools.runtimes.llama_cpp import reporting as consumer
from build_tools.runtimes.llama_cpp import provenance as provenance
from build_tools.runtimes.llama_cpp import browser_module as browser
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
    profiles = ['cpu-wasm32', 'webgpu-wasm64-jspi-gpu-profile', 'webgpu-wasm32-asyncify', 'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi']
    for name in profiles:
        manifest['profiles'][name] = {'variants': {}}
        for variant in ['browser', 'test']:
            for ext in ['mjs', 'wasm', 'd.ts']:
                file(f'profiles/{name}/{variant}/core.{ext}', f'Fixture: {name}/{variant}/{ext}\n'.encode())
            manifest['profiles'][name]['variants'][variant] = {
                'validation': {'compiled': True, 'browserSmoke': True, 'realModelInference': False},
            }
    file('profiles/cpu-wasm32/browser/model-graph-v1.schema.json', b'{}\n')
    file('profiles/cpu-wasm32/browser/model-graph-v1.functions.d.ts', b'// Fixture\n')
    file('api/schema.json', b'{}\n')
    file('api/functions.d.ts', b'// Fixture\n')
    file('host/index.mjs', b'// Fixture\n')
    file('host/README.md', b'Test documentation\n')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


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
        self.assertIn('profiles/cpu-wasm32/browser/model-graph-v1.schema.json', paths)
        self.assertIn('profiles/cpu-wasm32/browser/model-graph-v1.functions.d.ts', paths)
        self.assertIn('host/index.mjs', paths)
        self.assertNotIn('host/README.md', paths)

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

    def test_browser_annotation_impact_is_visible_in_the_publication_summary(self):
        self.data['upstreamDivergences']['generatedModuleTransformations'] = [{'id': 'browser-wasm-url-annotation'}]
        text = consumer.render_markdown(self.data, consumer.render_yaml(self.data))
        self.assertIn('Vite skips asset discovery at this expression; other bundlers may differ', text)
        self.assertIn('runtime fallback behavior is unchanged', text)

    def test_oversized_report_links_the_complete_yaml_without_truncating(self):
        self.data['huge'] = 'x' * 60000
        self.data['upstreamDivergences']['generatedModuleTransformations'] = [{'id': 'browser-wasm-url-annotation'}]
        output = self.root / 'report'
        text = consumer.write_report(output, self.data, '123', '2')
        raw = (output / 'consumer-update.yaml').read_bytes()
        self.assertEqual(raw, consumer.render_yaml(self.data).encode('utf-8'))
        self.assertGreater(len(raw), 60000)
        self.assertLessEqual(len(text.encode('utf-8')), 55000)
        self.assertNotIn('```yaml', text)
        self.assertIn('Vite skips asset discovery at this expression; other bundlers may differ', text)
        self.assertIn('runtime fallback behavior is unchanged', text)
        self.assertIn(SPEC, text)
        self.assertIn(A, text)
        self.assertIn('consumer-update-2', text)
        self.assertIn(f'https://github.com/{REPO}/actions/runs/123/attempts/2', text)
        self.assertIn(str(len(raw)), text)
        self.assertIn(hashlib.sha256(raw).hexdigest(), text)
        self.assertEqual(text, (output / 'consumer-update.md').read_text())
        envelope = json.loads((output / 'report.json').read_text())
        self.assertEqual(envelope['markdownSha256'], hashlib.sha256(text.encode('utf-8')).hexdigest())
        self.assertEqual(set(path.name for path in output.iterdir()), {'consumer-update.md', 'consumer-update.yaml', 'report.json'})

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
        self.root = Path(self.tmp.name) / 'runtimes/llama-cpp'
        self.root.mkdir(parents=True)
        # Real repository inputs, with only mtmd bodies/patches replaced by a
        # three-line fixture. WebGPU recipe reporting is tested separately with
        # the real upstream source, not fabricated stage hashes.
        for directory in ['config', 'native', 'cmake', 'docs', 'host', provenance.PATCH_DIRECTORY]:
            shutil.copytree(ROOT / directory, self.root / directory)
        for directory in ['file-backed-rows', 'qsa-selection']:
            shutil.copytree(runtime_test_root('llama-cpp') / directory, runtime_test_root('llama-cpp', self.root.parents[1]) / directory)
        toolchain = {'llamaCommit': B, 'emscriptenRelease': A,
                     'emscriptenAsyncifyBigIntPatch': {'sourceSha256': '1' * 64, 'patchedSha256': '2' * 64}}
        seed_toolchain(self.root, toolchain, scripts=True)
        self.vendor = self.root / 'vendor/llama.cpp'
        fixture = CheckedMtmdFixture(self.vendor, self.root / provenance.PATCH_DIRECTORY)
        fixture.install(self)
        (self.vendor / 'src/models').mkdir(parents=True)
        (self.vendor / 'src/models/qwen4exp.cpp').write_bytes(b'original QSA fixture')
        (self.vendor / 'src/models/models.h').write_bytes(b'original model header')
        self.manifest = {'sourceCommit': A, 'llamaCommit': B, 'profiles': {}}
        for name, enabled in [('cpu-wasm32', False), ('webgpu-wasm64-jspi', True)]:
            self.manifest['profiles'][name] = {'variants': {variant: {
                'toolchain': toolchain, 'cmakeCommand': ['cmake', '-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if enabled else 'OFF'),
                    '-DLCB_WEBGPU=OFF', '-DLCB_FILE_BACKED_PLE=OFF', '-DLCB_QSA_ALL_POOLS=OFF'],
            } for variant in ['browser', 'test']}}

    def test_runtime_selection_provenance_keeps_both_modes_and_legacy_default(self):
        result = provenance._describe(self.root, self.manifest)
        overlay = next(item for item in result['sourceOverlays'] if item['id']=='vision-bf16-runtime-selection')
        self.assertEqual(overlay['application']['runtimeStorageSelection'],
                         {'version':1,'modes':['legacy-expand','retain-checked'],'default':'legacy-expand'})
        self.assertEqual(len(overlay['application']['enabledProfileVariants']),4)
        self.assertEqual(overlay['compiledCopy']['sha256'],hashlib.sha256(b'before\npatched\nafter\n').hexdigest())
        self.assertNotIn(provenance.PATCH_DIRECTORY+'/mtmd-vision-bf16-runtime.patch',result['otherPatchFiles'])
        info = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['test']
        info['visionBf16Storage']='retain-checked'
        with self.assertRaisesRegex(ValueError,'BF16 runtime capability'):
            provenance._describe(self.root,self.manifest)

    def test_overlay_hashes_describe_pristine_and_actual_patched_copy(self):
        for info in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            info['cmakeCommand'] = [arg.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for arg in info['cmakeCommand']]
        with patch.object(provenance.webgpu_provenance, 'describe', return_value=({}, [], {})):
            result = provenance._describe(self.root, self.manifest)
        overlay = result['sourceOverlays'][0]
        self.assertFalse(result['vendorCheckoutModified'])
        self.assertEqual(overlay['compiledCopies']['mtmd-model.cpp']['sha256'], hashlib.sha256(b'before\npatched\nafter\n').hexdigest())
        self.assertEqual(overlay['upstreamSource']['sha256'], hashlib.sha256(b'before\noriginal\nafter\n').hexdigest())
        self.assertEqual(overlay['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertEqual((self.vendor / 'tools/mtmd/mtmd-model.cpp').read_text(), 'before\noriginal\nafter\n')
        self.assertIn('toolchainDivergences', result)

    def test_generated_browser_preparation_records_are_bound_to_source_and_published_output(self):
        for name in ('build_tools/runtimes/llama_cpp/browser_module.py', 'build_tools/runtimes/llama_cpp/build.py',
                     'host/emscripten/host-navigator-pre.js', 'docs/browser-host.md'):
            destination = provenance.repository_input(self.root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(provenance.repository_input(ROOT, name), destination)
        module = self.root / 'synthetic-core.mjs'
        module.write_bytes(b'function locate(){return new URL("core.wasm",import.meta.url).href}')
        record = browser.prepare(module)
        self.manifest['profiles']['cpu-wasm32']['variants']['browser']['browserModulePreparation'] = record
        self.manifest['files'] = [{'path': 'profiles/cpu-wasm32/browser/core.mjs', **record['output']}]
        report = provenance._describe(self.root, self.manifest)
        item, = report['generatedModuleTransformations']
        self.assertEqual(item['implementation'], record['implementation'])
        self.assertEqual(item['profileVariants'], [{'profile': 'cpu-wasm32', 'variant': 'browser',
                         'input': record['input'], 'output': record['output'], 'alreadyAnnotated': False}])
        self.assertEqual(item['supportingFiles']['host/emscripten/host-navigator-pre.js'],
                         provenance.file_identity(self.root / 'host/emscripten/host-navigator-pre.js'))
        self.manifest['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Browser module preparation'):
            provenance._describe(self.root, self.manifest)

    def test_audio_overlay_tracks_both_backends_and_actual_compiled_bytes(self):
        result = provenance._describe(self.root, self.manifest)
        audio = next(item for item in result['sourceOverlays'] if item['id'] == 'non-pthread-wasm-mtmd-threadpool')
        self.assertEqual(audio['compiledCopy']['sha256'], hashlib.sha256(b'before\npatched\nafter\n').hexdigest())
        self.assertEqual(audio['application']['enabledProfileVariants'], [
            'cpu-wasm32/browser', 'cpu-wasm32/test', 'webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertNotIn('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-non-pthread-threadpool.patch', result['otherPatchFiles'])
        self.assertEqual((self.vendor / 'tools/mtmd/mtmd-audio.cpp').read_text(), 'before\noriginal\nafter\n')

    def test_audio_overlay_conflict_is_not_hidden_by_a_successful_vision_overlay(self):
        (self.vendor / 'tools/mtmd/mtmd-subcontext.cpp').write_text('changed upstream\n')
        with self.assertRaises(ValueError):
            provenance._describe(self.root, self.manifest)

    def test_unknown_patch_files_are_not_silently_omitted(self):
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch').write_text('Another fixture patch\n')
        report = provenance._describe(self.root, self.manifest)
        self.assertIn('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch', report['otherPatchFiles'])
        self.assertEqual(report['otherPatchFiles']['upstream-patches-only-as-a-last-resort-with-explicit-user-approval/another.patch']['application'], 'not classified by this report')

    def test_retained_overlays_are_independent_and_use_only_the_renamed_directory(self):
        report = provenance._describe(self.root, self.manifest)
        self.assertEqual({item['id'] for item in report['sourceOverlays']}, {
            'webgpu-q6-subgroup-matvec', 'webgpu-checkpoint-batch-readback', 'webgpu-subgroup-size-control-matvec',
            'webgpu-aligned-word-load', 'webgpu-q4-fixed-subgroup-matvec', 'webgpu-quant-wide-m', 'webgpu-direct-mapped-read', 'webgpu-quant-half-pairs', 'webgpu-q6-short-workgroup', 'webgpu-f32-matvec-epilogue', 'webgpu-moe-tight-tile-bound', 'webgpu-matmul-vec4-layout', 'webgpu-q8-kernel-diagnostic-v1', 'webgpu-profile-query-budget', 'webgpu-host-input-snapshot', 'webgpu-iq3s-indexed-subgroup-matvec', 'webgpu-iq3s-indexed-field-loads', 'webgpu-iq3s-indexed-row-split', 'webgpu-rms-scale-fusion', 'webgpu-ssm-conv-silu-fusion', 'webgpu-async-tensor-copy', 'webgpu-scale-unary-fusion', 'webgpu-q8-fixed-subgroup-matvec', 'webgpu-q4-short-workgroup', 'webgpu-expert-short-workgroup', 'webgpu-moe-expert-sum', 'webgpu-q8-short-workgroup', 'webgpu-hc-pre-post', 'webgpu-vision-bf16-projector', 'non-pthread-wasm-mtmd-threadpool',
            'webgpu-same-device-tensor-copy', 'webgpu-moe-matrix-group-skip', 'webgpu-router-tail-fusion', 'webgpu-f32-small-m-prefill', 'webgpu-moe-gather-logical32',
            'webgpu-parameter-upload-batching', 'webgpu-ordered-readback', 'webgpu-bind-group-cache', 'webgpu-host-encode-assembly', 'webgpu-sampler-single-sync', 'webgpu-chunked-model-upload', 'webgpu-ssm-conv-single-token',
            'qwen4exp-file-backed-rows', 'qsa-all-pools-selection', 'vision-bf16-runtime-selection', 'webgpu-buffer-size-limit', 'webgpu-batched-write-scalar-fallback'})
        self.assertIn(provenance.PATCH_DIRECTORY, report['inventoryScope'])
        for item in report['sourceOverlays']:
            self.assertTrue(item['patch']['path'].startswith(provenance.PATCH_DIRECTORY + '/'))
            self.assertNotIn('inputOverlay', item['application'])
        self.assertFalse((self.root / 'patches').exists())

    def test_retired_moe_overlay_is_reported_but_never_claimed_as_compiled(self):
        report = provenance._describe(self.root, self.manifest)
        retired = next(item for item in report['retiredSourceOverlays'] if item['id'] == 'experimental-webgpu-moe-direct-slot')
        self.assertEqual(retired['id'], 'experimental-webgpu-moe-direct-slot')
        self.assertEqual(retired['upstreamedBy'], '620d2225cc436ab0b001e3200e590e3b286ef9ab')
        self.assertNotIn(retired['id'], {item['id'] for item in report['sourceOverlays']})
        self.assertFalse(any('moe-direct-slot' in path for path in report['otherPatchFiles']))

    def test_retired_moe_flags_cannot_describe_a_new_build(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for flag in ('-DLCB_WEBGPU_MOE_DIRECT_SLOT=ON', '-DLCB_WEBGPU_MOE_DIRECT_SLOT=OFF',
                     '-DLCB_WEBGPU_MOE_DIRECT_SLOT:BOOL=ON'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'Retired MoE'):
                variant['cmakeCommand'] = original + [flag]
                provenance._describe(self.root, self.manifest)

    def test_file_rows_enabled_records_coherent_sources_and_supporting_files(self):
        variant = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        variant['cmakeCommand'] = [arg.replace('LCB_FILE_BACKED_PLE=OFF', 'LCB_FILE_BACKED_PLE=ON') for arg in variant['cmakeCommand']]
        prepared = self.root / 'rows-copy/src'
        (prepared / 'models').mkdir(parents=True)
        (prepared.parent / provenance.file_rows.RECEIPT).write_text(json.dumps({'fixtureOnly': True}))
        for relative in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp', 'models/llama.cpp', 'models/models.h', 'models/file-backed-rows.h', 'models/file-backed-identity.h']:
            (prepared / relative).write_text('coherent ' + relative)
        with patch.object(provenance.file_rows, 'prepare', return_value=prepared) as prepare:
            report = provenance._describe(self.root, self.manifest)
        prepare.assert_called_once()
        item = next(x for x in report['sourceOverlays'] if x['id'] == 'qwen4exp-file-backed-rows')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(item['compiledCopies']['models/qwen4exp.cpp'], provenance.file_identity(prepared / 'models/qwen4exp.cpp'))
        self.assertEqual(item['coherentSourceSet']['files'], 7)
        self.assertEqual(len(item['compiledCopies']), 6)
        # This fixture has no model graph extension: report its approved patch
        # as unclassified, rather than falsely claiming it was compiled.
        graph_patch = provenance.PATCH_DIRECTORY + '/common-chat-independent-vocabulary.patch'
        self.assertEqual(set(report['otherPatchFiles']), {graph_patch})
        self.assertEqual(report['otherPatchFiles'][graph_patch]['application'],
                         'not classified by this report')
        # Coherent unchanged model copies are still covered without expanding
        # the compact consumer report to hundreds of per-file hashes.
        identities = {p.relative_to(prepared).as_posix(): provenance.file_identity(p)
                      for p in prepared.rglob('*') if p.is_file()}
        encoded = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(item['coherentSourceSet']['sha256'], hashlib.sha256(encoded).hexdigest())

    def test_qsa_disabled_is_reported_without_preparing(self):
        with patch.object(provenance.qsa, 'prepare') as prepare:
            report = provenance._describe(self.root, self.manifest)
        prepare.assert_not_called()
        item = next(x for x in report['sourceOverlays'] if x['id'] == 'qsa-all-pools-selection')
        self.assertEqual(item['compiledCopiesByProfileVariant'], {})
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

    def test_qsa_provenance_rejects_missing_duplicate_typed_or_invalid_flags(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([a for a in original if 'LCB_QSA_ALL_POOLS' not in a],
                        original + ['-DLCB_QSA_ALL_POOLS=ON'],
                        original + ['-DLCB_QSA_ALL_POOLS:BOOL=ON'],
                        [a.replace('LCB_QSA_ALL_POOLS=OFF', 'LCB_QSA_ALL_POOLS=MAYBE') for a in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'QSA all-pools'):
                variant['cmakeCommand'] = command
                provenance._describe(self.root, self.manifest)

    def test_composed_provenance_follows_final_source_in_each_profile_variant(self):
        cpu = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        cpu['cmakeCommand'] = [arg.replace('LCB_QSA_ALL_POOLS=OFF', 'LCB_QSA_ALL_POOLS=ON')
                               for arg in cpu['cmakeCommand']]
        for name, variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].items():
            variant['cmakeCommand'] = [arg.replace('LCB_FILE_BACKED_PLE=OFF', 'LCB_FILE_BACKED_PLE=ON')
                                       .replace('LCB_QSA_ALL_POOLS=OFF', 'LCB_QSA_ALL_POOLS=' + ('ON' if name == 'browser' else 'OFF'))
                                       for arg in variant['cmakeCommand']]
        prepared = self.root / 'rows-copy/src'
        (prepared / 'models').mkdir(parents=True)
        (prepared.parent / provenance.file_rows.RECEIPT).write_text(json.dumps({'fixtureOnly': True}))
        for relative in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp', 'models/llama.cpp',
                         'models/models.h', 'models/file-backed-rows.h', 'models/file-backed-identity.h']:
            (prepared / relative).write_text('coherent ' + relative)
        original = (prepared / 'models/qwen4exp.cpp').read_bytes()
        def compose(source, model_source, output):
            self.assertEqual(source, self.vendor)
            output.mkdir(parents=True)
            result = output / 'qwen4exp.cpp'
            result.write_bytes(model_source.read_bytes() + b' optimized')
            (output / provenance.qsa.RECEIPT).write_text(json.dumps({'fixtureOnly': True}))
            return result
        with patch.object(provenance.file_rows, 'prepare', return_value=prepared), \
             patch.object(provenance.qsa, 'prepare', side_effect=compose) as prepare:
            report = provenance._describe(self.root, self.manifest)
        self.assertEqual(prepare.call_count, 2)
        qsa = next(x for x in report['sourceOverlays'] if x['id'] == 'qsa-all-pools-selection')
        rows = next(x for x in report['sourceOverlays'] if x['id'] == 'qwen4exp-file-backed-rows')
        copies = qsa['compiledCopiesByProfileVariant']
        self.assertEqual(set(copies), {'cpu-wasm32/browser', 'webgpu-wasm64-jspi/browser'})
        self.assertNotEqual(copies['cpu-wasm32/browser']['sha256'], copies['webgpu-wasm64-jspi/browser']['sha256'])
        composed = copies['webgpu-wasm64-jspi/browser']
        self.assertEqual(composed['sha256'], hashlib.sha256(original + b' optimized').hexdigest())
        self.assertEqual(composed['modelHeader'], provenance.file_identity(prepared / 'models/models.h'))
        self.assertIsNone(rows['compiledCopies'])  # Do not misreport the intermediate copy as compiled.
        self.assertIsNone(rows['coherentSourceSet'])
        self.assertEqual(rows['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/browser']['models/qwen4exp.cpp']['sha256'], composed['sha256'])
        self.assertEqual(rows['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/test']['models/qwen4exp.cpp']['sha256'], hashlib.sha256(original).hexdigest())
        trees = rows['coherentSourceSetsByProfileVariant']
        self.assertNotEqual(trees['webgpu-wasm64-jspi/browser']['sha256'], trees['webgpu-wasm64-jspi/test']['sha256'])
        self.assertEqual((prepared / 'models/qwen4exp.cpp').read_bytes(), original)

    def test_file_rows_provenance_rejects_ambiguous_build_flags(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([a for a in original if 'LCB_FILE_BACKED_PLE' not in a],
                        original + ['-DLCB_FILE_BACKED_PLE=ON'],
                        original + ['-DLCB_FILE_BACKED_PLE:BOOL=ON'],
                        [a.replace('LCB_FILE_BACKED_PLE=OFF', 'LCB_FILE_BACKED_PLE=MAYBE') for a in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'file-backed PLE'):
                variant['cmakeCommand'] = command
                provenance._describe(self.root, self.manifest)

    def test_webgpu_activation_fails_closed(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([a for a in original if not a.startswith('-DLCB_WEBGPU=')],
                        original + ['-DLCB_WEBGPU=OFF'],
                        original + ['-DLCB_WEBGPU:BOOL=ON'],
                        [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=MAYBE') for a in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'WebGPU'):
                variant['cmakeCommand'] = command
                provenance._describe(self.root, self.manifest)

    def test_nested_patch_is_inventoried_with_its_exact_identity(self):
        relative = provenance.PATCH_DIRECTORY + '/nested/example.patch'
        extra = self.root / relative
        extra.parent.mkdir()
        extra.write_bytes(b'Not an applied exception; inventory fixture only.\n')
        item = provenance._describe(self.root, self.manifest)['otherPatchFiles'][relative]
        self.assertEqual(item['sha256'], hashlib.sha256(extra.read_bytes()).hexdigest())
        self.assertEqual(item['bytes'], extra.stat().st_size)
        self.assertEqual(item['application'], 'not classified by this report')

    def test_unknown_build_activation_fails(self):
        self.manifest['profiles']['cpu-wasm32']['variants']['browser']['cmakeCommand'] = ['cmake']
        with self.assertRaisesRegex(ValueError, 'activation'):
            provenance._describe(self.root, self.manifest)

    def test_patch_conflict_and_source_commit_mismatch_fail(self):
        self.manifest['sourceCommit'] = C
        with self.assertRaisesRegex(ValueError, 'captured build-input receipt'):
            provenance._check_receipts(self.manifest, {'formatVersion': 1, 'parentInputMode': 'commit-objects',
                                                      'sourceCommit': A, 'upstream': {'commit': B}})
        self.manifest['sourceCommit'] = A
        (self.vendor / 'tools/mtmd/mtmd-model.cpp').write_text('changed upstream\n')
        with self.assertRaises(ValueError):
            provenance._describe(self.root, self.manifest)

    def test_symlinked_source_files_are_rejected(self):
        source = self.vendor / 'tools/mtmd/mtmd-model.cpp'
        copied = self.root / 'elsewhere'
        source.rename(copied)
        source.symlink_to(copied)
        with self.assertRaisesRegex(ValueError, '[Ll]inked'):
            provenance._describe(self.root, self.manifest)

    def test_mtmd_retirements_and_profile_header_universe_remain_visible(self):
        variant = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        variant['cmakeCommand'] = [arg.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON')
                                  for arg in variant['cmakeCommand']]
        with patch.object(provenance.webgpu_provenance, 'describe', return_value=({}, [], {})):
            report = provenance._describe(self.root, self.manifest)
        retired = {row['id'] for row in report['retiredSourceOverlays']}
        active = {row['id']: row for row in report['sourceOverlays']}
        self.assertTrue({'single-thread-wasm-audio-preprocessing',
                         'vision-bf16-cpu-loader-plumbing'} <= retired)
        audio = next(row for row in report['retiredSourceOverlays']
                     if row['id'] == 'single-thread-wasm-audio-preprocessing')
        self.assertEqual(audio['upstreamedBy'], 'a2654ec43148b8ae827d956276ce2c1eacfd3853')
        self.assertEqual(report['baseCommit'], B)
        self.assertFalse(retired & active.keys())
        trees = active['vision-bf16-runtime-selection']['compiledSourceTreesByProfileVariant']
        cpu, gpu = trees['cpu-wasm32/browser'], trees['webgpu-wasm64-jspi/browser']
        self.assertEqual(cpu['sourceInputs'], gpu['sourceInputs'])
        self.assertNotEqual(cpu['sha256'], gpu['sha256'])
        self.assertEqual(cpu['files'], 7)
        self.assertEqual(cpu['coreSources'], ['mtmd.cpp', 'mtmd-model.cpp',
                                             'mtmd-subcontext.cpp', 'mtmd-audio.cpp'])
        self.assertNotIn('mtmd-audio.cpp', cpu['modifiedFiles'])


if __name__ == '__main__':
    unittest.main()
