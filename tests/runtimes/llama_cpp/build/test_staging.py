"""Test matrix artifact staging with synthetic runtimes, never compiling Wasm."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from tests.support.llama_toolchain import merged_toolchain, seed_toolchain
from unittest.mock import patch

from build_tools.project import runtime_root
ROOT = runtime_root('llama-cpp')
from build_tools.runtimes.llama_cpp import package as package_runtime
from build_tools.runtimes.llama_cpp import staging
from build_tools.runtimes.llama_cpp.package import VARIANTS, build_package, copy_license_notices, validate
from build_tools.runtimes.llama_cpp.staging import API_FILES, stage_profile, stage_toolchain_notices
from build_tools.packaging.files import tree_identity
from tests.support.packaging_cli import run_cli


class CiBuildArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lcb-ci-artifacts-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'runtimes/llama-cpp'
        self.root.mkdir(parents=True)
        self.build=self.root/'build'
        self.output=self.root/'upload'
        self.source_commit='a'*40
        self.toolchain=merged_toolchain()
        self.profiles=json.loads((ROOT/'config/profiles.json').read_text())
        for name,cfg in self.profiles.items():
            for variant,settings in VARIANTS.items():
                folder=self.build/name/variant
                (folder/'runtime/auxiliary').mkdir(parents=True)
                (folder/'generated').mkdir()
                marker=(name+'/'+variant).encode()
                # A valid empty module with a custom section distinguishes each
                # pair without compiling or pretending to exercise inference.
                (folder/'runtime/core.wasm').write_bytes(b'\0asm\1\0\0\0\0'+bytes([len(marker)+1,len(marker)])+marker)
                (folder/'runtime/core.mjs').write_text(f'// CI transfer fixture: {name}/{variant}, not an inference runtime.\n')
                (folder/'runtime/core.d.ts').write_text('export default function create(): Promise<unknown>;\n')
                (folder/'runtime/auxiliary/worker.js').write_text('// Complete runtime tree is preserved.\n')
                for file in API_FILES:
                    (folder/'generated'/file).write_text('{}\n' if file.endswith('.json') else '// Generated API fixture\n')
                (folder/'generated/bindings.cpp').write_text('// Build-only intermediate.\n')
                (folder/'CMakeCache.txt').write_text('PRIVATE_BUILD_PATH=/example\n')
                (folder/'object.o').write_bytes(b'not a runtime asset')
                (folder/'libcore.a').write_bytes(b'not a runtime asset')
                data={'profile':name,'variant':variant,'variantConfiguration':settings,'sourceCommit':self.source_commit,'sourceDirty':False,
                      'sourceStatusBeforeBuild':[],'sourceStatusAfterBuild':[],
                      'llamaCommit':self.toolchain['llamaCommit'],'toolchain':self.toolchain,
                      'configuration':cfg,'builtAtUnix':123456,
                      'validation':{'compiled':True,'browserSmoke':False,'realModelInference':False}}
                (folder/'provenance.json').write_text(json.dumps(data,indent=2)+'\n')
        self.sdk=self.root/'tools/emscripten'
        self.dawn=self.root/'tools/emdawnwebgpu_pkg'
        for tool in (self.sdk,self.dawn):
            tool.mkdir(parents=True)
            (tool/'LICENSE').write_text('Synthetic toolchain license\n'+tool.name+'\n')
            (tool/'vendor/component').mkdir(parents=True)
            (tool/'vendor/component/COPYRIGHT.txt').write_text('Synthetic embedded dependency notice\n')
            (tool/'compiler-binary').write_bytes(b'sdk executables must not be transferred')
            (tool/'cache').mkdir()
            (tool/'cache/libc.a').write_bytes(b'sdk build products must not be transferred')
            (tool/'.git').mkdir()
            (tool/'.git/LICENSE').write_text('Never collect Git metadata.\n')
        # A normal source tree has these notices in the pinned submodule. For the
        # package-identity test, copy reference code but use synthetic notice bytes.
        self.source=self.root/'source'
        shutil.copytree(ROOT/'host',self.source/'host')
        shutil.copytree(ROOT/'packaging',self.source/'packaging')
        (self.source/'docs').mkdir()
        for file in package_runtime.GUIDE_FILES:
            shutil.copy2(ROOT/'docs'/file,self.source/'docs'/file)
        shutil.copy2(ROOT/'LICENSE',self.source/'LICENSE')
        self.upstream=self.source/'vendor/llama.cpp'
        self.upstream.mkdir(parents=True)
        (self.upstream/'LICENSE').write_text('Synthetic upstream license\n')
        for relative in package_runtime.EMBEDDED_NOTICE_FILES:
            file=self.upstream/relative
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text('Synthetic embedded notice fixture\n'+relative+'\n')

    def stage(self,name='cpu-wasm32',output=None,source_commit=None,variant='browser'):
        return stage_profile(self.build,output or self.output,name,
                             variant=variant,source_commit=source_commit or self.source_commit,
                             toolchain=self.toolchain,configuration=self.profiles[name])

    def set_provenance(self,name,build_variant='browser',**fields):
        file=self.build/name/build_variant/'provenance.json';data=json.loads(file.read_text())
        data.update(fields);file.write_text(json.dumps(data,indent=2)+'\n')

    @staticmethod
    def contents(directory):
        return {file.relative_to(directory).as_posix():file.read_bytes()
                for file in directory.rglob('*') if file.is_file()}

    def test_stage_contains_complete_runtime_but_no_compiler_intermediates(self):
        folder=self.stage()
        expected={f'runtime/{p}' for p in ('core.mjs','core.wasm','core.d.ts','auxiliary/worker.js')}
        expected|={'generated/'+name for name in API_FILES}|{'provenance.json'}
        self.assertEqual(set(self.contents(folder)),expected)
        for name in expected:
            self.assertEqual((folder/name).read_bytes(),(self.build/'cpu-wasm32/browser'/name).read_bytes())
        self.assertFalse((folder/'CMakeCache.txt').exists())
        self.assertFalse((folder/'generated/bindings.cpp').exists())

    def test_variants_keep_their_matching_javascript_and_wasm(self):
        for variant in VARIANTS:
            staged=self.stage(variant=variant)
            original=self.build/'cpu-wasm32'/variant/'runtime'
            for file in ('core.mjs','core.wasm'):
                self.assertEqual((staged/'runtime'/file).read_bytes(),(original/file).read_bytes())
        for file in ('core.mjs','core.wasm'):
            self.assertNotEqual((self.output/'cpu-wasm32/browser/runtime'/file).read_bytes(),
                                (self.output/'cpu-wasm32/test/runtime'/file).read_bytes())

    def test_missing_runtime_or_api_prevents_staging(self):
        for file in ['runtime/core.wasm','runtime/core.d.ts','generated/schema.mjs','provenance.json']:
            with self.subTest(file=file):
                source=self.build/'cpu-wasm32/browser'/file;original=source.read_bytes();source.unlink()
                with self.assertRaisesRegex(ValueError,'Missing or linked'):
                    self.stage()
                self.assertFalse((self.output/'cpu-wasm32').exists())
                source.write_bytes(original)

    def test_dirty_or_uncompiled_profile_is_not_transferred(self):
        for fields,expected in [({'sourceDirty':True},'dirty'),
                                ({'validation':{'compiled':False}},'uncompiled')]:
            with self.subTest(fields=fields):
                self.set_provenance('cpu-wasm32',**fields)
                with self.assertRaisesRegex(ValueError,expected):self.stage()
                self.assertFalse((self.output/'cpu-wasm32').exists())
                self.set_provenance('cpu-wasm32',sourceDirty=False,validation={'compiled':True})

    def test_stale_source_profile_toolchain_or_config_is_rejected(self):
        file=self.build/'cpu-wasm32/browser/provenance.json';original=file.read_bytes()
        changes=[{'sourceCommit':'c'*40},{'profile':'webgpu-wasm64-jspi-gpu-profile'},
                 {'llamaCommit':'d'*40},{'toolchain':{}},{'configuration':{}},
                 {'variant':'test'},{'variantConfiguration':VARIANTS['test']}]
        for change in changes:
            with self.subTest(change=change):
                self.set_provenance('cpu-wasm32',**change)
                with self.assertRaises(ValueError):self.stage()
                self.assertFalse((self.output/'cpu-wasm32').exists())
                file.write_bytes(original)

    def test_symlinked_payload_is_rejected(self):
        external=self.root/'external';external.write_text('Do not follow this link.')
        link=self.build/'cpu-wasm32/browser/runtime/extra.js';link.symlink_to(external)
        with self.assertRaisesRegex(ValueError,'symlink'):self.stage()
        link.unlink()
        schema=self.build/'cpu-wasm32/browser/generated/schema.json';schema.unlink();schema.symlink_to(external)
        with self.assertRaisesRegex(ValueError,'linked'):self.stage()

    def test_symlinked_profile_directory_is_rejected(self):
        original=self.build/'cpu-wasm32'
        elsewhere=self.root/'relocated-profile'
        original.rename(elsewhere)
        original.symlink_to(elsewhere,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink'):self.stage()

    def test_profile_paths_and_repeated_staging_are_rejected(self):
        for name in ['../escape','/absolute','a/b','CPU']:
            with self.assertRaisesRegex(ValueError,'Invalid profile'):
                stage_profile(self.build,self.output,name,source_commit=self.source_commit,
                              variant='browser',toolchain=self.toolchain,configuration={})
        self.stage()
        before=self.contents(self.output)
        with self.assertRaisesRegex(ValueError,'already staged'):self.stage()
        self.assertEqual(self.contents(self.output),before)

    def test_only_standalone_toolchain_notices_are_transferred(self):
        stage_toolchain_notices([self.sdk,self.dawn],self.output)
        expected={f'toolchain-licenses/{root.name}/{name}'
                  for root in (self.sdk,self.dawn) for name in ['LICENSE','vendor/component/COPYRIGHT.txt']}
        self.assertEqual(set(self.contents(self.output)),expected)
        for root in (self.sdk,self.dawn):
            self.assertEqual((self.output/'toolchain-licenses'/root.name/'LICENSE').read_bytes(),
                             (root/'LICENSE').read_bytes())
        with self.assertRaisesRegex(ValueError,'already staged'):
            stage_toolchain_notices([self.sdk,self.dawn],self.output)

    def test_missing_or_duplicate_toolchain_notice_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing license'):
            stage_toolchain_notices([self.root/'missing'],self.output)
        empty=self.root/'empty';empty.mkdir()
        with self.assertRaisesRegex(ValueError,'No standalone'):
            stage_toolchain_notices([empty],self.output)
        with self.assertRaisesRegex(ValueError,'distinct names'):
            stage_toolchain_notices([self.sdk,self.sdk],self.output)

    def test_requested_notice_failure_cannot_leave_a_partly_staged_group(self):
        empty = self.root / 'empty-notices'; empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No standalone notices'):
            stage_profile(self.build, self.output, 'cpu-wasm32', variant='browser',
                          source_commit=self.source_commit, toolchain=self.toolchain,
                          configuration=self.profiles['cpu-wasm32'], notice_roots=(empty,))
        self.assertFalse((self.output / 'cpu-wasm32/browser').exists())
        self.assertFalse((self.output / 'toolchain-licenses').exists())

    def package_cli(self, destination, *, fail_pack=None, defer=False):
        arguments = ['--build-root', str(self.build), '--output', str(destination), '--profiles', 'cpu-wasm32']
        for root in (self.upstream, self.sdk, self.dawn):
            arguments += ['--license-root', str(root)]
        if defer:
            arguments.append('--defer-npm-pack')
        return run_cli(self.root, 'build_tools.runtimes.llama_cpp.package', arguments,
                       source_root=self.source, fail_pack=fail_pack)

    def test_package_cli_npm_failure_keeps_previous_package(self):
        destination = self.root / 'package'
        result, _ = self.package_cli(destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        before, inode = tree_identity(destination), destination.stat().st_ino
        result, calls = self.package_cli(destination, fail_pack=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('injected npm pack failure', result.stderr)
        self.assertEqual(tree_identity(destination), before)
        self.assertEqual(destination.stat().st_ino, inode)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(Path(calls[0]['cwd']), destination)
        self.assertEqual(list(destination.parent.glob('bicore-output-*')), [])

    def test_package_cli_packs_once_in_candidate_or_explicitly_defers(self):
        destination = self.root / 'package'
        result, calls = self.package_cli(destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0]['cwd']).name, 'prepared')
        result, calls = self.package_cli(destination, defer=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_package_rejects_profile_paths_before_reading_or_copying_any_inputs(self):
        destination = self.root / 'package'
        for profiles in ([], ['cpu-wasm32', 'cpu-wasm32'], ['../escape'], ['/absolute'], ['a/b'], ['CPU']):
            with self.subTest(profiles=profiles), patch.object(package_runtime, 'read_regular') as read:
                with self.assertRaisesRegex(ValueError, 'Invalid package profile selection'):
                    build_package(self.build, destination, profiles, license_roots=[])
            read.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob('bicore-output-*')), [])
        result, calls = run_cli(self.root, 'build_tools.runtimes.llama_cpp.package',
                                ['--build-root', str(self.build), '--output', str(destination),
                                 '--profiles', '../escape'], source_root=self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid package profile selection', result.stderr)
        self.assertEqual(calls, [])
        self.assertFalse(destination.exists())

    def test_parallel_shards_reconstruct_identical_package_without_sdk_reinstall(self):
        uploads=self.root/'uploads'
        def build_shard(name):
            folder=uploads/name
            for variant in VARIANTS:self.stage(name,folder,variant=variant)
            if name=='webgpu-wasm64-jspi':stage_toolchain_notices([self.sdk,self.dawn],folder)
            return folder
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            shards=list(executor.map(build_shard,self.profiles))
        # Mimic download-artifact's merge: distinct directories, never overwrite.
        downloaded=self.root/'downloaded'
        seen=set()
        for shard in shards:
            names=set(self.contents(shard))
            self.assertFalse(seen&names)
            seen.update(names)
            shutil.copytree(shard,downloaded,dirs_exist_ok=True)
        serial_package=self.root/'serial-package'
        merged_package=self.root/'merged-package'
        with patch.object(package_runtime,'ROOT',self.source):
            build_package(self.build,serial_package,list(self.profiles),license_roots=[self.upstream,self.sdk,self.dawn])
            # Remove the original build tree and the SDK. Assembly must not depend on them.
            shutil.rmtree(self.build);shutil.rmtree(self.sdk);shutil.rmtree(self.dawn)
            build_package(downloaded,merged_package,list(self.profiles),license_roots=[self.upstream,
                downloaded/'toolchain-licenses/emscripten',downloaded/'toolchain-licenses/emdawnwebgpu_pkg'])
        self.assertEqual(self.contents(merged_package),self.contents(serial_package))
        self.assertEqual(validate(merged_package)['profiles'],list(self.profiles))
        manifest=json.loads((merged_package/'manifest.json').read_text())
        for profile in manifest['profiles'].values():
            self.assertEqual(set(profile['variants']),set(VARIANTS))
            for provenance in profile['variants'].values():
                self.assertFalse(provenance['validation']['browserSmoke'])
                self.assertFalse(provenance['validation']['realModelInference'])

    def test_reassembly_still_rejects_mixed_schemas_and_source_commits(self):
        for name in self.profiles:
            for variant in VARIANTS:self.stage(name,variant=variant)
        stage_toolchain_notices([self.sdk,self.dawn],self.output)
        roots=[self.upstream,self.output/'toolchain-licenses/emscripten',
               self.output/'toolchain-licenses/emdawnwebgpu_pkg']
        file=self.output/'webgpu-wasm64-jspi-gpu-profile/browser/generated/schema.json';original=file.read_bytes()
        file.write_text('{"different":true}')
        with patch.object(package_runtime,'ROOT',self.source):
            with self.assertRaisesRegex(ValueError,'Mixed binding schemas'):
                build_package(self.output,self.root/'package',list(self.profiles),license_roots=roots)
            file.write_bytes(original)
            provenance=self.output/'webgpu-wasm64-jspi-gpu-profile/browser/provenance.json'
            data=json.loads(provenance.read_text());data['sourceCommit']='c'*40;provenance.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Mixed source commits'):
                build_package(self.output,self.root/'package',list(self.profiles),license_roots=roots)

    def test_reassembly_requires_every_profile(self):
        self.stage('cpu-wasm32');self.stage('webgpu-wasm64-jspi')
        stage_toolchain_notices([self.sdk,self.dawn],self.output)
        with patch.object(package_runtime,'ROOT',self.source):
            with self.assertRaises(FileNotFoundError):
                build_package(self.output,self.root/'package',list(self.profiles),license_roots=[self.upstream,
                    self.output/'toolchain-licenses/emscripten',self.output/'toolchain-licenses/emdawnwebgpu_pkg'])

    def test_cli_preserves_download_paths_and_source_provenance(self):
        repository = self.seed_staging_repository()
        shutil.copytree(self.sdk,self.root.parents[1]/'.tools/emsdk/upstream/emscripten')
        shutil.copytree(self.dawn,self.root.parents[1]/'.tools/emdawnwebgpu_pkg')
        argv=['staging.py','--profile','webgpu-wasm64-jspi','--variant','browser','--include-toolchain-notices']
        original_git = staging.git
        with patch.object(staging,'ROOT',self.root), patch.object(sys,'argv',argv), \
             patch.object(staging, 'git', wraps=original_git) as git, \
             patch('builtins.print'):
            staging.main()
        self.assertEqual(git.call_count, 2)
        git.assert_called_with(repository, 'rev-parse', 'HEAD')
        stage=self.root/'build/ci-upload'
        self.assertEqual({p.name for p in stage.iterdir()},{'webgpu-wasm64-jspi','toolchain-licenses'})
        original=self.build/'webgpu-wasm64-jspi/browser/provenance.json'
        self.assertEqual((stage/'webgpu-wasm64-jspi/browser/provenance.json').read_bytes(),original.read_bytes())
        self.assertTrue((stage/'toolchain-licenses/emscripten/LICENSE').is_file())
        self.assertTrue((stage/'toolchain-licenses/emdawnwebgpu_pkg/LICENSE').is_file())

    def test_cli_refuses_staging_outside_build_or_inside_a_profile(self):
        self.seed_staging_repository()
        for output in (self.root,self.build,self.build/'cpu-wasm32/browser/nested'):
            with self.subTest(output=output):
                argv=['staging.py','--profile','cpu-wasm32','--variant','browser','--output',str(output)]
                with patch.object(staging,'ROOT',self.root), patch.object(sys,'argv',argv), \
                     patch.object(staging, 'stage_profile') as stage, \
                     patch('sys.stderr'):
                    with self.assertRaises(SystemExit):staging.main()
                stage.assert_not_called()

    def seed_staging_repository(self):
        """Commit only selection inputs; the prebuilt payloads remain local fixtures."""
        repository = self.root.parents[1]
        config = self.root / 'config'; config.mkdir()
        (config / 'profiles.json').write_text(json.dumps(self.profiles))
        (config / 'variants.json').write_text(json.dumps(VARIANTS))
        seed_toolchain(self.root, self.toolchain)
        (repository / '.gitignore').write_text('*\n')
        for args in [('init', '-q'), ('config', 'user.name', 'fixture'),
                     ('config', 'user.email', 'fixture@example.invalid')]:
            subprocess.run(['git', '-C', str(repository), *args], check=True)
        subprocess.run(['git', '-C', str(repository), 'add', '-f', '.gitignore', 'toolchain/config.json',
                        'runtimes/llama-cpp/config/profiles.json', 'runtimes/llama-cpp/config/variants.json',
                        'runtimes/llama-cpp/config/toolchain.json'], check=True)
        subprocess.run(['git', '-C', str(repository), 'commit', '-qm', 'captured transfer configuration'], check=True)
        self.source_commit = staging.git(repository, 'rev-parse', 'HEAD').strip()
        for name in self.profiles:
            for variant in VARIANTS:
                self.set_provenance(name, variant, sourceCommit=self.source_commit,
                                    sourceInputs={'sourceCommit': self.source_commit, 'parentInputMode': 'commit-objects'})
        return repository

    def test_hidden_worktree_config_edits_do_not_replace_committed_transfer_checks(self):
        repository = self.seed_staging_repository()
        names = ['runtimes/llama-cpp/config/profiles.json', 'runtimes/llama-cpp/config/variants.json',
                 'runtimes/llama-cpp/config/toolchain.json', 'toolchain/config.json']
        subprocess.run(['git', '-C', str(repository), 'update-index', '--assume-unchanged', *names], check=True)
        for name in names:
            (repository / name).write_text('{"hidden":"must not select build metadata"}')
        self.assertEqual(staging.git(repository, 'status', '--porcelain').strip(), '')
        with patch.object(staging, 'ROOT', self.root), patch('builtins.print'):
            staging.main(['--profile', 'cpu-wasm32', '--variant', 'browser'])
        result = self.build / 'ci-upload/cpu-wasm32/browser/provenance.json'
        data = json.loads(result.read_text())
        self.assertEqual(data['configuration'], self.profiles['cpu-wasm32'])
        self.assertEqual(data['variantConfiguration'], VARIANTS['browser'])
        self.assertEqual(data['toolchain'], self.toolchain)
        self.assertEqual(data['sourceCommit'], self.source_commit)

    def test_head_switch_during_selection_cannot_publish_partial_output_or_touch_prior_stage(self):
        repository = self.seed_staging_repository()
        selected = self.source_commit
        config = self.root / 'config/variants.json'
        config.write_text(json.dumps({**VARIANTS, 'browser': {'changed': True}}))
        subprocess.run(['git', '-C', str(repository), 'add', '-f', str(config)], check=True)
        subprocess.run(['git', '-C', str(repository), 'commit', '-qm', 'later configuration'], check=True)
        later = staging.git(repository, 'rev-parse', 'HEAD').strip()
        subprocess.run(['git', '-C', str(repository), 'checkout', '--detach', '-q', selected], check=True)
        previous = self.build / 'ci-upload/cpu-wasm32/browser'
        previous.mkdir(parents=True)
        (previous / 'keep').write_text('previous staged output')
        before, inode = tree_identity(previous), previous.stat().st_ino
        for source, relative in ((self.sdk, '.tools/emsdk/upstream/emscripten'),
                                 (self.dawn, '.tools/emdawnwebgpu_pkg')):
            shutil.copytree(source, repository / relative)
        original_git = staging.git
        calls = []
        def changing_git(repo, *args):
            value = original_git(repo, *args)
            calls.append(args)
            if len(calls) == 1:
                subprocess.run(['git', '-C', str(repository), 'update-ref', 'HEAD', later], check=True)
            return value
        with patch.object(staging, 'ROOT', self.root), patch.object(staging, 'git', side_effect=changing_git):
            with self.assertRaisesRegex(ValueError, 'Source commit changed while staging'):
                staging.main(['--profile', 'webgpu-wasm64-jspi', '--variant', 'browser', '--include-toolchain-notices'])
        self.assertEqual(calls, [('rev-parse', 'HEAD'), ('rev-parse', 'HEAD')])
        self.assertEqual(tree_identity(previous), before)
        self.assertEqual(previous.stat().st_ino, inode)
        self.assertFalse((self.build / 'ci-upload/webgpu-wasm64-jspi/browser').exists())
        self.assertFalse((self.build / 'ci-upload/toolchain-licenses').exists())
        self.assertEqual(list((self.build / 'ci-upload').rglob('bicore-output-*')), [])

    def test_mismatched_or_dirty_capture_receipt_cannot_be_transferred(self):
        for capture in ({'sourceCommit': 'd' * 40, 'parentInputMode': 'commit-objects'},
                        {'sourceCommit': self.source_commit, 'parentInputMode': 'dirty-local-worktree'}):
            with self.subTest(capture=capture):
                self.set_provenance('cpu-wasm32', sourceInputs=capture)
                with self.assertRaisesRegex(ValueError, 'Captured build inputs'):
                    self.stage()
                self.assertFalse((self.output / 'cpu-wasm32/browser').exists())

    def test_license_selection_matches_original_rules(self):
        (self.sdk/'symlink').symlink_to(self.sdk/'LICENSE')
        (self.sdk/'LICENSE.link').symlink_to(self.sdk/'LICENSE')
        destination=self.root/'notices'
        self.assertEqual(copy_license_notices(self.sdk,destination),2)
        self.assertEqual(set(self.contents(destination)),{'LICENSE','vendor/component/COPYRIGHT.txt'})
        with self.assertRaisesRegex(ValueError,'Missing license'):
            copy_license_notices(self.root/'not-found',destination)

    def test_copy_failure_does_not_publish_a_partial_profile(self):
        with patch.object(staging.shutil, 'copy2', side_effect=OSError('injected copy failure')):
            with self.assertRaisesRegex(OSError, 'injected copy failure'):
                self.stage()
        self.assertFalse((self.output / 'cpu-wasm32/browser').exists())
        self.assertEqual(list(self.output.rglob('bicore-output-*')), [])

    def test_notice_failure_does_not_publish_partial_licenses(self):
        original = staging.copy_license_notices
        def copy(source, destination):
            if source == self.dawn:
                raise OSError('injected notice failure')
            return original(source, destination)
        with patch.object(staging, 'copy_license_notices', side_effect=copy):
            with self.assertRaisesRegex(OSError, 'notice failure'):
                stage_toolchain_notices([self.sdk, self.dawn], self.output)
        self.assertFalse((self.output / 'toolchain-licenses').exists())
        self.assertEqual(list(self.output.rglob('bicore-output-*')), [])

    def test_relocation_entrypoint_uses_an_ordinary_python_package(self):
        import subprocess
        from build_tools.project import REPOSITORY_ROOT
        result = subprocess.run([sys.executable, '-m', staging.__name__, '--help'],
                                cwd=REPOSITORY_ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--profile', result.stdout)

if __name__=='__main__':unittest.main()
