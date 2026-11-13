"""Actual llama preparation/reporting against explicitly local synthetic history.

This exercises source generation and receipts, not CMake or runtime compilation.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT, runtime_root
from tests.support.local_source import source_repository
from build_tools.runtimes.llama_cpp import inputs, provenance
from build_tools.source.checkout import Checkout, git, git_environment


class CapturedReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', runtime_root('llama-cpp') / 'vendor/llama.cpp'))
        if not (source / 'src/llama-sampler.cpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE to the pinned source or its supplied archive')
        temporary = tempfile.TemporaryDirectory(prefix='captured-report-')
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name)
        cls.repository = cls.work / 'repository'
        inputs._copy_local(REPOSITORY_ROOT, cls.repository)
        cls.runtime = cls.repository / inputs.RUNTIME
        vendor = cls.repository / inputs.UPSTREAM
        vendor.parent.mkdir(parents=True, exist_ok=True)
        source_repository(source, vendor)
        upstream_commit = git(vendor, 'rev-parse', 'HEAD').strip()
        local = cls.runtime / 'config/toolchain.json'
        configuration = json.loads(local.read_text())
        configuration['llamaCommit'] = upstream_commit
        local.write_text(json.dumps(configuration))
        (cls.repository / '.gitmodules').write_text('[submodule "llama-fixture"]\n path = ' + inputs.UPSTREAM + '\n url = https://example.invalid/local-fixture.git\n')
        git(cls.repository, 'init', '--quiet')
        git(cls.repository, 'config', 'user.name', 'Local synthetic source fixture')
        git(cls.repository, 'config', 'user.email', 'fixture@example.invalid')
        git(cls.repository, 'config', 'advice.addEmbeddedRepo', 'false')
        git(cls.repository, 'add', '-A')
        git(cls.repository, 'commit', '--quiet', '-m', 'test: local synthetic reporting fixture, not public source history')
        revision = git(cls.repository, 'rev-parse', 'HEAD').strip()
        captured = inputs.capture(cls.repository, revision, Checkout(vendor, upstream_commit), cls.runtime / 'build/source-inputs')
        script = cls.work / 'prepare.py'
        # These are the same preparer entry points used by CMake. Their outputs
        # are observed as source artifacts; no compiler success is fabricated.
        script.write_text('''from pathlib import Path
import json,sys
from build_tools.runtimes.llama_cpp import source,inputs
from build_tools.runtimes.llama_cpp.sources import file_backed_rows,qsa_selection,mtmd
from build_tools.runtimes.llama_cpp.bf16 import runtime_metadata
from build_tools.toolchain.config import runtime_toolchain
from build_tools.project import runtime_root
root=runtime_root('llama-cpp'); upstream=root/'vendor/llama.cpp'
receipt=json.loads((root.parents[2]/'inputs.json').read_text())
settings=json.loads((root.parents[2]/'build-info.json').read_text())
manifest={'sourceCommit':receipt['sourceCommit'],'llamaCommit':receipt['upstream']['commit'],'profiles':{},'files':[]}
for profile, webgpu, diagnostic, rows in [('cpu',False,False,False),('webgpu',True,False,True),('profile',True,True,True)]:
    build=Path(sys.argv[1])/profile
    build.mkdir(parents=True)
    if webgpu:
        source.prepare(upstream,build/'webgpu-source-overlay',gpu_profile=diagnostic)
    if rows:
        models=file_backed_rows.prepare(upstream,build/'file-backed-rows-overlay')
    else:
        models=upstream/'src'
    qsa_selection.prepare(upstream,models/'models/qwen4exp.cpp',build/'qsa-selection-overlay')
    mtmd.prepare(upstream,build/'mtmd-source-overlay',mode='webgpu' if webgpu else 'cpu',
                 patch_root=root/mtmd.PATCH_DIRECTORY)
    command=['cmake','-DLLAMA_BUILD_COMMIT='+settings['commit'],'-DLLAMA_BUILD_NUMBER='+str(settings['number']),'-DGIT_EXE:FILEPATH=','-DLCB_WEBGPU='+('ON' if webgpu else 'OFF'),'-DLCB_WEBGPU_BF16_PROJECTOR='+('ON' if webgpu else 'OFF'),'-DLCB_FILE_BACKED_PLE='+('ON' if rows else 'OFF'),'-DLCB_QSA_ALL_POOLS=ON','-DGGML_WEBGPU_GPU_PROFILE='+('ON' if diagnostic else 'OFF')]
    manifest['profiles'][profile]={'variants':{'test':{'toolchain':runtime_toolchain(root),'visionBf16Storage':runtime_metadata(),'cmakeCommand':command,'sourceInputs':receipt,'upstreamBuildInfo':settings,'preparedSources':inputs.prepared_sources(build,webgpu=webgpu,file_rows=rows)}}}
Path(sys.argv[2]).write_text(json.dumps(manifest))
''')
        manifest_path = cls.work / 'manifest.json'
        environment = git_environment()
        environment.update({'PYTHONPATH': str(captured.root / 'repository'), 'PYTHONDONTWRITEBYTECODE': '1',
                            'GIT_CEILING_DIRECTORIES': str(captured.root)})
        subprocess.run([sys.executable, str(script), str(cls.work / 'prepared'), str(manifest_path)],
                       cwd=captured.root / 'repository', env=environment, check=True)
        cls.manifest = json.loads(manifest_path.read_text())
        inputs.verify(captured)

    def test_report_uses_committed_patches_and_matches_all_observed_prepared_sources(self):
        patch_file = self.runtime / inputs.PATCH_DIRECTORY / 'ggml-webgpu-aligned-word-load.patch'
        original = patch_file.read_bytes()
        patch_file.write_bytes(b'changed original worktree after source generation\n')
        try:
            report = provenance.collect(self.runtime, self.manifest)
        finally:
            patch_file.write_bytes(original)
        self.assertEqual(report['capturedInputs'], self.manifest['profiles']['cpu']['variants']['test']['sourceInputs'])
        overlays = {item['id']: item for item in report['sourceOverlays']}
        self.assertEqual(overlays['webgpu-aligned-word-load']['patch']['sha256'],
                         provenance.file_identity(patch_file)['sha256'])
        self.assertIsNone(overlays['webgpu-aligned-word-load']['compiledCopies'])
        self.assertIn('profile/test', overlays['webgpu-aligned-word-load']['compiledCopiesByProfileVariant'])
        for item in report['sourceOverlays']:
            for path, record in item.get('supportingFiles', {}).items():
                actual = provenance.repository_input(self.runtime, path)
                self.assertTrue(actual.is_file(), path)
                for key, value in provenance.file_identity(actual).items():
                    self.assertEqual(record[key], value, path)

    def test_changed_generated_source_receipt_is_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed['profiles']['webgpu']['variants']['test']['preparedSources']['webgpu-source-overlay']['ggml-webgpu.cpp']['sha256'] = '0' * 64
        with self.assertRaises(subprocess.CalledProcessError):
            provenance.collect(self.runtime, changed)

    def test_missing_or_dirty_capture_receipt_is_rejected_before_description(self):
        for change in ['missing', 'dirty', 'different']:
            manifest = copy.deepcopy(self.manifest)
            entry = manifest['profiles']['cpu']['variants']['test']
            if change == 'missing':
                del entry['sourceInputs']
            elif change == 'dirty':
                entry['sourceInputs']['parentInputMode'] = 'dirty-local-worktree'
            else:
                entry['sourceInputs']['repositoryTree']['sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'receipt differs'):
                provenance.collect(self.runtime, manifest)


    def test_recorded_build_counter_is_independent_of_reporting_checkout_depth(self):
        # A complete local clone and a depth-one CI clone have identical source
        # objects but different upstream build counters. Bind the recorded
        # setting to its compile argument instead of inventing another counter.
        manifest = copy.deepcopy(self.manifest)
        receipt = manifest['profiles']['cpu']['variants']['test']['sourceInputs']
        for info in manifest['profiles'].values():
            entry = info['variants']['test']
            original = entry['upstreamBuildInfo']['number']
            entry['upstreamBuildInfo']['number'] = original + 10
            entry['cmakeCommand'] = [arg.replace('-DLLAMA_BUILD_NUMBER=' + str(original),
                                                '-DLLAMA_BUILD_NUMBER=' + str(original + 10))
                                     for arg in entry['cmakeCommand']]
        provenance._check_receipts(manifest, receipt)
        manifest['profiles']['cpu']['variants']['test']['upstreamBuildInfo']['number'] += 1
        with self.assertRaisesRegex(ValueError, 'differ from CMake arguments'):
            provenance._check_receipts(manifest, receipt)
