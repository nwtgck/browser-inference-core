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
import prepare_browser_module as browser

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
        (self.vendor / provenance.tensor_copy.SOURCE_PATH).parent.mkdir(parents=True)
        (self.vendor / provenance.tensor_copy.SOURCE_PATH).write_text('Copy provenance fixture\n')
        (self.vendor / provenance.host_input_snapshot.BACKEND_SOURCE_PATH).write_text('Backend provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.tensor_copy.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.tensor_copy.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.PARAM_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.PARAM_PATCH_NAME)
        (self.vendor / provenance.webgpu_source.SSM_HEADER_PATH).write_text('SSM header provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.SSM_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.SSM_PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.hc.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.hc.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.q8_short.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.q8_short.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.moe_sum.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.moe_sum.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.expert_short.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.expert_short.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.ordered_readback.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.ordered_readback.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.bind_group_cache.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.bind_group_cache.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.q4_short.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.q4_short.PATCH_NAME)
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.sampler_single_sync.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.sampler_single_sync.PATCH_NAME)
        for module in (provenance.scale_unary, provenance.q8_subgroup, provenance.rms_scale, provenance.async_tensor_copy, provenance.iq3s_subgroup, provenance.host_input_snapshot, provenance.ssm_conv_silu, provenance.moe_group_skip, provenance.router_tail, provenance.q8_diagnostic):
            shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / module.PATCH_NAME,
                         self.root / provenance.PATCH_DIRECTORY / module.PATCH_NAME)
        (self.vendor / 'src').mkdir()
        (self.vendor / provenance.webgpu_source.LOADER_PATH).write_text('Loader provenance fixture\n')
        (self.vendor / provenance.sampler_single_sync.SOURCE_PATH).write_text('Sampler provenance fixture\n')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.webgpu_source.LOADER_PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.webgpu_source.LOADER_PATCH_NAME)
        (self.root / 'tests/file-backed-rows').mkdir(parents=True)
        shutil.copy2(ROOT / 'tests/file-backed-rows/inputs.json', self.root / 'tests/file-backed-rows/inputs.json')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.file_rows.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.file_rows.PATCH_NAME)
        for path in ['scripts/prepare_file_backed_rows.py', 'cmake/FileBackedRowsOverlay.cmake',
                     'bridge/file-backed-rows.h', 'docs/file-backed-rows.md']:
            (self.root / path).write_text('file-backed rows provenance fixture\n')
        for path in ['CMakeLists.txt', 'scripts/prepare_webgpu_q8_diagnostic.py', 'scripts/generate_webgpu_q8_diagnostic.py', 'bridge/webgpu-q8-diagnostic.h', 'bridge/webgpu-q8-diagnostic.cpp', 'scripts/prepare_webgpu_router_tail.py', 'scripts/prepare_webgpu_moe_group_skip.py', 'scripts/prepare_webgpu_ssm_conv_silu.py', 'scripts/prepare_webgpu_host_input_snapshot.py', 'scripts/prepare_webgpu_iq3s_subgroup.py', 'scripts/prepare_webgpu_rms_scale.py', 'scripts/prepare_webgpu_async_tensor_copy.py', 'scripts/prepare_webgpu_scale_unary.py', 'scripts/prepare_webgpu_q8_subgroup.py', 'scripts/prepare_sampler_single_sync.py', 'scripts/prepare_webgpu_bind_group_cache.py', 'scripts/prepare_webgpu_q4_short.py', 'scripts/prepare_webgpu_ordered_readback.py', 'scripts/prepare_webgpu_expert_short.py', 'docs/webgpu-expert-short-workgroup.md', 'scripts/prepare_webgpu_moe_sum.py', 'docs/webgpu-moe-expert-sum.md', 'scripts/prepare_webgpu_q8_short.py', 'docs/webgpu-q8-short-workgroup.md',
                     'scripts/prepare_webgpu_hc.py', 'docs/webgpu-hc-pre-post.md', 'scripts/prepare_webgpu_tensor_copy.py', 'cmake/WebgpuSourceOverlay.cmake', 'scripts/prepare_webgpu_source.py', 'docs/webgpu-param-upload-batching.md',
                     'docs/webgpu-tensor-copy.md', 'docs/moe-direct-slot.md', 'scripts/prepare_mtmd.py', 'cmake/MtmdOverlay.cmake', 'bridge/mtmd-bf16.h',
                     'docs/webgpu-bf16-projector.md',
                     'cmake/MtmdAudioOverlay.cmake', 'docs/audio-single-thread.md']:
            (self.root / path).write_text('Provenance fixture: ' + path + '\n')
        (self.vendor / 'src/models').mkdir(parents=True)
        (self.vendor / 'src/models/qwen4exp.cpp').write_bytes(b'original QSA fixture')
        (self.vendor / 'src/models/models.h').write_bytes(b'original model header')
        (self.root / 'tests/qsa-selection').mkdir(parents=True)
        shutil.copy2(ROOT / 'tests/qsa-selection/inputs.json', self.root / 'tests/qsa-selection/inputs.json')
        shutil.copy2(ROOT / provenance.PATCH_DIRECTORY / provenance.qsa.PATCH_NAME,
                     self.root / provenance.PATCH_DIRECTORY / provenance.qsa.PATCH_NAME)
        for path in ['scripts/prepare_qsa_selection.py', 'cmake/QsaSelectionOverlay.cmake',
                     'docs/qsa-all-pools-selection.md']:
            (self.root / path).write_text('QSA provenance fixture\n')
        self.manifest = {'sourceCommit': A, 'llamaCommit': B, 'profiles': {}}
        for name, enabled in [('cpu-wasm32', False), ('webgpu-wasm64-jspi', True)]:
            self.manifest['profiles'][name] = {'variants': {variant: {
                'toolchain': toolchain, 'cmakeCommand': ['cmake', '-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if enabled else 'OFF'),
                    '-DLCB_WEBGPU=OFF', '-DLCB_FILE_BACKED_PLE=OFF', '-DLCB_QSA_ALL_POOLS=OFF'],
            } for variant in ['browser', 'test']}}
        self.prepared = self.root / 'prepared.cpp'
        self.prepared.write_bytes(b'combined source fixture')
        (self.prepared.parent / 'llama-model-loader.cpp').write_bytes(b'loader source fixture')
        (self.prepared.parent / 'llama-sampler.cpp').write_bytes(b'sampler source fixture')
        (self.prepared.parent / 'ggml-backend.cpp').write_bytes(b'prepared backend source fixture')
        (self.prepared.parent / 'ggml-backend-host-snapshot.h').write_bytes(b'private snapshot header fixture')
        (self.prepared.parent / 'ggml-webgpu-shader-lib.hpp').write_bytes(b'SSM header source fixture')
        (self.prepared.parent / 'ggml-webgpu-hc.hpp').write_bytes(b'HC host fixture')
        (self.prepared.parent / provenance.hc.HEADER_NAME).write_bytes(b'HC shader fixture')
        (self.prepared.parent / 'ggml-webgpu.cpp').write_bytes(b'combined source fixture')
        (self.prepared.parent / 'ggml-webgpu-moe-sum.hpp').write_bytes(b'MoE host fixture')
        (self.prepared.parent / 'ggml-webgpu-scale-unary.hpp').write_bytes(b'scale unary host fixture')
        (self.prepared.parent / provenance.scale_unary.HEADER_NAME).write_bytes(b'scale unary shader fixture')
        (self.prepared.parent / provenance.q8_subgroup.HEADER_NAME).write_bytes(b'Q8 subgroup shader fixture')
        (self.prepared.parent / 'ggml-webgpu-rms-scale.hpp').write_bytes(b'RMS scale host fixture')
        (self.prepared.parent / 'ggml-webgpu-ssm-conv-silu.hpp').write_bytes(b'SSM convolution SiLU host fixture')
        (self.prepared.parent / provenance.ssm_conv_silu.HEADER_NAME).write_bytes(b'SSM convolution SiLU shader fixture')
        (self.prepared.parent / provenance.moe_group_skip.HEADER_NAME).write_bytes(b'MoE matrix group shader fixture')
        (self.prepared.parent / provenance.router_tail.HEADER_NAME).write_bytes(b'router tail shader fixture')
        (self.prepared.parent / 'ggml-webgpu-router-tail.hpp').write_bytes(b'router tail helper fixture')
        (self.vendor / provenance.router_tail.BASE / 'wgsl-shaders').mkdir(parents=True, exist_ok=True)
        (self.vendor / provenance.router_tail.BASE / 'wgsl-shaders/sum_rows.wgsl').write_bytes(b'upstream router sum fixture')
        moe_shader = self.vendor / provenance.moe_group_skip.BASE / 'wgsl-shaders/mul_mat_id.wgsl'
        moe_shader.parent.mkdir(parents=True, exist_ok=True)
        moe_shader.write_bytes(b'Canonical MoE matrix shader fixture')
        (self.prepared.parent / provenance.rms_scale.HEADER_NAME).write_bytes(b'RMS scale shader fixture')
        (self.prepared.parent / provenance.iq3s_subgroup.HEADER_NAME).write_bytes(b'IQ3 subgroup shader fixture')
        (self.prepared.parent / provenance.moe_sum.HEADER_NAME).write_bytes(b'MoE shader fixture')
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

    def test_generated_browser_preparation_records_are_bound_to_source_and_published_output(self):
        for name in ('scripts/prepare_browser_module.py', 'scripts/build.py',
                     'bridge/host-navigator-pre.js', 'docs/browser-host.md'):
            shutil.copy2(ROOT / name, self.root / name)
        module = self.root / 'synthetic-core.mjs'
        module.write_bytes(b'function locate(){return new URL("core.wasm",import.meta.url).href}')
        record = browser.prepare(module)
        self.manifest['profiles']['cpu-wasm32']['variants']['browser']['browserModulePreparation'] = record
        self.manifest['files'] = [{'path': 'profiles/cpu-wasm32/browser/core.mjs', **record['output']}]
        report = provenance.collect(self.root, self.manifest)
        item, = report['generatedModuleTransformations']
        self.assertEqual(item['implementation'], record['implementation'])
        self.assertEqual(item['profileVariants'], [{'profile': 'cpu-wasm32', 'variant': 'browser',
                         'input': record['input'], 'output': record['output'], 'alreadyAnnotated': False}])
        self.assertEqual(item['supportingFiles']['bridge/host-navigator-pre.js'],
                         provenance.file_identity(self.root / 'bridge/host-navigator-pre.js'))
        self.manifest['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Browser module preparation'):
            provenance.collect(self.root, self.manifest)

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
            'webgpu-q8-kernel-diagnostic-v1', 'webgpu-host-input-snapshot', 'webgpu-iq3s-indexed-subgroup-matvec', 'webgpu-rms-scale-fusion', 'webgpu-ssm-conv-silu-fusion', 'webgpu-async-tensor-copy', 'webgpu-scale-unary-fusion', 'webgpu-q8-fixed-subgroup-matvec', 'webgpu-q4-short-workgroup', 'webgpu-expert-short-workgroup', 'webgpu-moe-expert-sum', 'webgpu-q8-short-workgroup', 'webgpu-hc-pre-post', 'webgpu-vision-bf16-projector', 'single-thread-wasm-audio-preprocessing',
            'webgpu-same-device-tensor-copy', 'webgpu-moe-matrix-group-skip', 'webgpu-router-tail-fusion',
            'webgpu-parameter-upload-batching', 'webgpu-ordered-readback', 'webgpu-bind-group-cache', 'webgpu-sampler-single-sync', 'webgpu-chunked-model-upload', 'webgpu-ssm-conv-single-token',
            'qwen4exp-file-backed-rows', 'qsa-all-pools-selection'})
        self.assertIn(provenance.PATCH_DIRECTORY, report['inventoryScope'])
        for item in report['sourceOverlays']:
            self.assertTrue(item['patch']['path'].startswith(provenance.PATCH_DIRECTORY + '/'))
            self.assertNotIn('inputOverlay', item['application'])
        self.assertFalse((self.root / 'patches').exists())

    def test_hc_inventory_and_active_source_identity(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-hc-pre-post')
        self.assertIsNone(item['compiledCopies'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-hc-pre-post')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(set(item['compiledCopies']), {'ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
            'ggml-webgpu-hc.hpp', provenance.hc.HEADER_NAME})
        self.assertEqual(item['compiledCopies']['ggml-webgpu-hc.hpp']['sha256'], hashlib.sha256(b'HC host fixture').hexdigest())

    def test_q8_short_inventory_tracks_only_the_active_final_compiled_header(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-q8-short-workgroup')
        self.assertIsNone(item['compiledCopy'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON')
                                  for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-q8-short-workgroup')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(item['compiledCopy']['sha256'],
                         hashlib.sha256(b'SSM header source fixture').hexdigest())
        self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])
        self.assertIn('scripts/prepare_webgpu_q8_short.py', item['supportingFiles'])

    def test_expert_short_inventory_tracks_final_copies_and_activation(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-expert-short-workgroup')
        self.assertIsNone(item['compiledCopies'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-expert-short-workgroup')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(set(item['compiledCopies']), {'ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp'})
        self.assertEqual(item['compiledCopies']['ggml-webgpu-shader-lib.hpp']['sha256'],
                         hashlib.sha256(b'SSM header source fixture').hexdigest())
        self.assertEqual(item['reviewedInputs'], provenance.expert_short.CONTRACT_INPUTS)
        self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])
        self.assertIn('scripts/prepare_webgpu_expert_short.py', item['supportingFiles'])

    def test_q4_short_inventory_tracks_final_header_and_activation(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-q4-short-workgroup')
        self.assertIsNone(item['compiledCopy'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-q4-short-workgroup')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(item['compiledCopy']['sha256'], hashlib.sha256(b'SSM header source fixture').hexdigest())
        self.assertEqual(item['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/browser'], item['compiledCopy'])
        self.assertEqual(item['reviewedInputs'], provenance.q4_short.CONTRACT_INPUTS)
        self.assertEqual(item['patch']['sha256'], provenance.q4_short.PATCH_SHA256)
        self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])

    def test_q8_diagnostic_and_shared_final_tus_have_profile_specific_identities(self):
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']
        for name, info in active.items():
            info['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in info['cmakeCommand']]
            info['cmakeCommand'].append('-DGGML_WEBGPU_GPU_PROFILE=' + ('ON' if name == 'test' else 'OFF'))
        profiled = self.root / 'profiled'; profiled.mkdir()
        for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic-sha256.hpp'):
            (profiled / name).write_text('profile-only final fixture: ' + name)
        def prepare_variant(*args, **kwargs):
            return profiled / 'ggml-webgpu.cpp' if kwargs.get('gpu_profile') else self.prepared
        with patch.object(provenance.webgpu_source, 'prepare', side_effect=prepare_variant):
            report = provenance.collect(self.root, self.manifest)
        items = {item['id']: item for item in report['sourceOverlays']}
        diagnostic = items['webgpu-q8-kernel-diagnostic-v1']
        self.assertEqual(diagnostic['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/test'])
        self.assertEqual(diagnostic['compiledCopies']['ggml-webgpu.cpp'], provenance.file_identity(profiled / 'ggml-webgpu.cpp'))
        for key in ('webgpu-same-device-tensor-copy', 'webgpu-rms-scale-fusion', 'webgpu-q8-fixed-subgroup-matvec'):
            item = items[key]
            field = 'compiledCopies' if 'compiledCopies' in item else 'compiledCopy'
            self.assertIsNone(item[field])
            copies = item['compiledCopiesByProfileVariant']
            self.assertNotEqual(copies['webgpu-wasm64-jspi/browser'], copies['webgpu-wasm64-jspi/test'])
        single = items['webgpu-same-device-tensor-copy']['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/test']
        self.assertEqual(single['sha256'], provenance.file_identity(profiled / 'ggml-webgpu.cpp')['sha256'])
        self.assertIn('GGML_WEBGPU_GPU_PROFILE=1', single['compileDefinitions'])

    def test_fusion_and_q8_subgroup_inventory_track_compiled_copies(self):
        for module, overlay in ((provenance.router_tail, 'webgpu-router-tail-fusion'),
                                (provenance.moe_group_skip, 'webgpu-moe-matrix-group-skip'),
                                (provenance.ssm_conv_silu, 'webgpu-ssm-conv-silu-fusion'),
                                (provenance.iq3s_subgroup, 'webgpu-iq3s-indexed-subgroup-matvec'),
                                (provenance.rms_scale, 'webgpu-rms-scale-fusion'),
                                (provenance.scale_unary, 'webgpu-scale-unary-fusion'),
                                (provenance.q8_subgroup, 'webgpu-q8-fixed-subgroup-matvec')):
            off = provenance.collect(self.root, self.manifest)
            item = next(x for x in off['sourceOverlays'] if x['id'] == overlay)
            self.assertIsNone(item['compiledCopies'])
            self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
            active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
            active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
            on = provenance.collect(self.root, self.manifest)
            item = next(x for x in on['sourceOverlays'] if x['id'] == overlay)
            self.assertEqual(item['patch']['sha256'], module.PATCH_SHA256)
            self.assertEqual(item['reviewedInputs'], module.CONTRACT_INPUTS)
            self.assertEqual(item['compiledCopies'][module.HEADER_NAME]['sha256'], provenance.file_identity(self.prepared.parent / module.HEADER_NAME)['sha256'])
            self.assertEqual(item['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/browser'], item['compiledCopies'])
            self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])
            active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=ON', '-DLCB_WEBGPU=OFF') for arg in active['cmakeCommand']]

    def test_host_snapshot_inventory_tracks_both_translation_units_and_private_header(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-host-input-snapshot')
        self.assertIsNone(item['compiledCopies'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-host-input-snapshot')
        self.assertEqual(item['patch']['sha256'], provenance.host_input_snapshot.PATCH_SHA256)
        self.assertEqual(item['reviewedInputs'], provenance.host_input_snapshot.CONTRACT_INPUTS)
        self.assertEqual(set(item['compiledCopies']), {'ggml-webgpu.cpp', 'ggml-backend.cpp', 'ggml-backend-host-snapshot.h'})
        for name, identity in item['compiledCopies'].items():
            self.assertEqual(identity, provenance.file_identity(self.prepared.parent / name))
        self.assertEqual(item['compiledCopiesByProfileVariant']['webgpu-wasm64-jspi/browser'], item['compiledCopies'])
        self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])

    def test_moe_sum_inventory_tracks_active_compiled_copies_and_not_unclassified(self):
        off = provenance.collect(self.root, self.manifest)
        item = next(x for x in off['sourceOverlays'] if x['id'] == 'webgpu-moe-expert-sum')
        self.assertIsNone(item['compiledCopies'])
        self.assertEqual(item['application']['enabledProfileVariants'], [])
        self.assertNotIn(item['patch']['path'], off['otherPatchFiles'])
        active = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        active['cmakeCommand'] = [arg.replace('-DLCB_WEBGPU=OFF', '-DLCB_WEBGPU=ON') for arg in active['cmakeCommand']]
        on = provenance.collect(self.root, self.manifest)
        item = next(x for x in on['sourceOverlays'] if x['id'] == 'webgpu-moe-expert-sum')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(set(item['compiledCopies']), {'ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                         'ggml-webgpu-moe-sum.hpp', provenance.moe_sum.HEADER_NAME})
        self.assertEqual(item['compiledCopies']['ggml-webgpu-moe-sum.hpp']['sha256'], hashlib.sha256(b'MoE host fixture').hexdigest())
        self.assertNotIn(item['patch']['path'], on['otherPatchFiles'])

    def test_retired_moe_overlay_is_reported_but_never_claimed_as_compiled(self):
        report = provenance.collect(self.root, self.manifest)
        retired, = report['retiredSourceOverlays']
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
                provenance.collect(self.root, self.manifest)

    def test_file_rows_enabled_records_coherent_sources_and_supporting_files(self):
        variant = self.manifest['profiles']['webgpu-wasm64-jspi']['variants']['browser']
        variant['cmakeCommand'] = [arg.replace('LCB_FILE_BACKED_PLE=OFF', 'LCB_FILE_BACKED_PLE=ON') for arg in variant['cmakeCommand']]
        prepared = self.root / 'rows-copy/src'
        (prepared / 'models').mkdir(parents=True)
        for relative in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp', 'models/llama.cpp', 'models/models.h', 'models/file-backed-rows.h']:
            (prepared / relative).write_text('coherent ' + relative)
        with patch.object(provenance.file_rows, 'prepare', return_value=prepared) as prepare:
            report = provenance.collect(self.root, self.manifest)
        prepare.assert_called_once()
        item = next(x for x in report['sourceOverlays'] if x['id'] == 'qwen4exp-file-backed-rows')
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser'])
        self.assertEqual(item['compiledCopies']['models/qwen4exp.cpp'], provenance.file_identity(prepared / 'models/qwen4exp.cpp'))
        self.assertEqual(item['coherentSourceSet']['files'], 6)
        self.assertEqual(len(item['compiledCopies']), 5)
        self.assertEqual(report['otherPatchFiles'], {})
        # Coherent unchanged model copies are still covered without expanding
        # the compact consumer report to hundreds of per-file hashes.
        identities = {p.relative_to(prepared).as_posix(): provenance.file_identity(p)
                      for p in prepared.rglob('*') if p.is_file()}
        encoded = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(item['coherentSourceSet']['sha256'], hashlib.sha256(encoded).hexdigest())

    def test_qsa_disabled_is_reported_without_preparing(self):
        with patch.object(provenance.qsa, 'prepare') as prepare:
            report = provenance.collect(self.root, self.manifest)
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
                provenance.collect(self.root, self.manifest)

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
        for relative in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp', 'models/llama.cpp',
                         'models/models.h', 'models/file-backed-rows.h']:
            (prepared / relative).write_text('coherent ' + relative)
        original = (prepared / 'models/qwen4exp.cpp').read_bytes()
        def compose(source, model_source, output):
            self.assertEqual(source, self.vendor)
            output.mkdir(parents=True)
            result = output / 'qwen4exp.cpp'
            result.write_bytes(model_source.read_bytes() + b' optimized')
            return result
        with patch.object(provenance.file_rows, 'prepare', return_value=prepared), \
             patch.object(provenance.qsa, 'prepare', side_effect=compose) as prepare:
            report = provenance.collect(self.root, self.manifest)
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

    def test_all_profile_compositions_fit_the_publication_report_limit(self):
        template = copy.deepcopy(self.manifest['profiles']['cpu-wasm32'])
        profiles = ['cpu-wasm32', 'webgpu-wasm64-jspi-gpu-profile', 'webgpu-wasm32-asyncify',
                    'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi']
        self.manifest['profiles'] = {name: copy.deepcopy(template) for name in profiles}
        for name, info in self.manifest['profiles'].items():
            for variant in info['variants'].values():
                variant['cmakeCommand'] = [
                    'cmake', '-DLCB_QSA_ALL_POOLS=ON',
                    '-DLCB_FILE_BACKED_PLE=' + ('ON' if 'wasm64' in name else 'OFF'),
                    '-DLCB_WEBGPU=' + ('ON' if name.startswith('webgpu') else 'OFF'),
                    '-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if name.startswith('webgpu') else 'OFF'),
                ]
        prepared = self.root / 'rows-copy/src'
        (prepared / 'models').mkdir(parents=True)
        for relative in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp', 'models/llama.cpp',
                         'models/models.h', 'models/file-backed-rows.h']:
            (prepared / relative).write_text('coherent fixture ' + relative)
        def compose(source, model_source, output):
            output.mkdir(parents=True)
            result = output / 'qwen4exp.cpp'
            result.write_bytes(model_source.read_bytes() + b' optimized')
            return result
        with patch.object(provenance.file_rows, 'prepare', return_value=prepared), \
             patch.object(provenance.qsa, 'prepare', side_effect=compose):
            divergence = provenance.collect(self.root, self.manifest)
        package = self.root / 'runtime-package'
        fixture_package(package)
        lock = consumer.extract_lock(fixture_lock(), SPEC, C, '0.1.0')
        lock['specifier'] = SPEC
        data = consumer.metadata(package, REPO, C, lock, divergence)
        text = consumer.write_report(self.root / 'consumer-report', data, '123', '1')
        self.assertLess(len(text.encode()), 55000)
        qsa = next(item for item in divergence['sourceOverlays'] if item['id'] == 'qsa-all-pools-selection')
        self.assertEqual(len(qsa['compiledCopiesByProfileVariant']), 10)

    def test_file_rows_provenance_rejects_ambiguous_build_flags(self):
        variant = self.manifest['profiles']['cpu-wasm32']['variants']['browser']
        original = variant['cmakeCommand']
        for command in ([a for a in original if 'LCB_FILE_BACKED_PLE' not in a],
                        original + ['-DLCB_FILE_BACKED_PLE=ON'],
                        original + ['-DLCB_FILE_BACKED_PLE:BOOL=ON'],
                        [a.replace('LCB_FILE_BACKED_PLE=OFF', 'LCB_FILE_BACKED_PLE=MAYBE') for a in original]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'file-backed PLE'):
                variant['cmakeCommand'] = command
                provenance.collect(self.root, self.manifest)

    def test_cpu_inventories_patches_without_preparing_source(self):
        with patch.object(provenance.webgpu_source, 'prepare') as prepare:
            report = provenance.collect(self.root, self.manifest)
        prepare.assert_not_called()
        for suffix in ('same-device-tensor-copy', 'async-tensor-copy', 'parameter-upload-batching', 'ordered-readback', 'bind-group-cache', 'sampler-single-sync', 'chunked-model-upload', 'ssm-conv-single-token'):
            item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-' + suffix)
            self.assertIsNone(item['compiledCopy'])
            self.assertEqual(item['application']['enabledProfileVariants'], [])
            self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

    def test_webgpu_provenance_records_combined_bytes_for_both_patches(self):
        for variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            variant['cmakeCommand'] = [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for a in variant['cmakeCommand']]
        report = provenance.collect(self.root, self.manifest)
        for suffix in ('same-device-tensor-copy', 'async-tensor-copy', 'parameter-upload-batching', 'ordered-readback', 'bind-group-cache'):
            item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-' + suffix)
            self.assertEqual(item['compiledCopy']['sha256'], provenance.file_identity(self.prepared)['sha256'])
            self.assertEqual(item['compiledCopy']['compileDefinitions'], ['GGML_WEBGPU_BATCH_PARAM_UPLOADS'])
            self.assertEqual(item['application']['option'], 'LCB_WEBGPU')
            self.assertEqual(item['application']['enabledProfileVariants'],
                             ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
            self.assertEqual(len(item['compiledCopiesByProfileVariant']), 2)
            if suffix == 'ordered-readback':
                self.assertEqual(item['patch']['sha256'], provenance.ordered_readback.PATCH_SHA256)
                self.assertEqual(item['reviewedInputs'], provenance.ordered_readback.CONTRACT_INPUTS)
                self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])
            if suffix == 'bind-group-cache':
                self.assertEqual(item['patch']['sha256'], provenance.bind_group_cache.PATCH_SHA256)
                self.assertEqual(item['reviewedInputs'], provenance.bind_group_cache.CONTRACT_INPUTS)
                self.assertTrue(item['compiledCopy']['bindGroupCache'])
                self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])
            if suffix == 'async-tensor-copy':
                self.assertEqual(item['patch']['sha256'], provenance.async_tensor_copy.PATCH_SHA256)
                self.assertEqual(item['reviewedInputs'], provenance.async_tensor_copy.CONTRACT_INPUTS)
                self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

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

    def test_webgpu_sampler_provenance_records_separate_compiled_source(self):
        for variant in self.manifest['profiles']['webgpu-wasm64-jspi']['variants'].values():
            variant['cmakeCommand'] = [a.replace('LCB_WEBGPU=OFF', 'LCB_WEBGPU=ON') for a in variant['cmakeCommand']]
        report = provenance.collect(self.root, self.manifest)
        item = next(e for e in report['sourceOverlays'] if e['id'] == 'webgpu-sampler-single-sync')
        self.assertEqual(item['compiledCopy']['sha256'], provenance.file_identity(self.prepared.parent / 'llama-sampler.cpp')['sha256'])
        self.assertEqual(item['patch']['sha256'], provenance.sampler_single_sync.PATCH_SHA256)
        self.assertEqual(item['reviewedInputs'], provenance.sampler_single_sync.CONTRACT_INPUTS)
        self.assertEqual(item['application']['enabledProfileVariants'], ['webgpu-wasm64-jspi/browser', 'webgpu-wasm64-jspi/test'])
        self.assertEqual(len(item['compiledCopiesByProfileVariant']), 2)
        self.assertNotIn(item['patch']['path'], report['otherPatchFiles'])

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
