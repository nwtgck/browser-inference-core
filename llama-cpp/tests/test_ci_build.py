"""Test matrix artifact staging with synthetic runtimes, never compiling Wasm."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from fixture_sources import seed_sources
from fixture_toolchain import merged_toolchain, seed_toolchain
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import package_runtime
import stage_ci_build
from package_runtime import VARIANTS, build_package, copy_license_notices, validate
from stage_ci_build import API_FILES, stage_profile, stage_toolchain_notices


class CiBuildArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lcb-ci-artifacts-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'llama-cpp'
        self.root.mkdir()
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
                data={'sourceId':'upstream-stable','profile':name,'variant':variant,'variantConfiguration':settings,'sourceCommit':self.source_commit,'sourceDirty':False,
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
        shutil.copytree(ROOT/'examples',self.source/'examples')
        shutil.copytree(ROOT/'packaging',self.source/'packaging')
        (self.source/'docs').mkdir()
        shutil.copy2(ROOT/'docs/chat-and-multimodal.md',self.source/'docs/chat-and-multimodal.md')
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

    def test_source_track_cannot_be_substituted_at_staging(self):
        with self.assertRaisesRegex(ValueError,'source track'):
            stage_profile(self.build,self.output,'cpu-wasm32',variant='browser',source_commit=self.source_commit,
                          toolchain=self.toolchain,configuration=self.profiles['cpu-wasm32'],source_id='upstream-nightly')
        self.assertFalse(self.output.exists())

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
        changes=[{'sourceCommit':'c'*40},{'profile':'cpu-wasm64'},
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
        file=self.output/'cpu-wasm64/browser/generated/schema.json';original=file.read_bytes()
        file.write_text('{"different":true}')
        with patch.object(package_runtime,'ROOT',self.source):
            with self.assertRaisesRegex(ValueError,'Mixed binding schemas'):
                build_package(self.output,self.root/'package',list(self.profiles),license_roots=roots)
            file.write_bytes(original)
            provenance=self.output/'cpu-wasm64/browser/provenance.json'
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
        config=self.root/'config'; config.mkdir()
        (config/'profiles.json').write_text(json.dumps(self.profiles))
        seed_toolchain(self.root, self.toolchain)
        seed_sources(self.root)
        shutil.copytree(self.sdk,self.root.parent/'.tools/emsdk/upstream/emscripten')
        shutil.copytree(self.dawn,self.root.parent/'.tools/emdawnwebgpu_pkg')
        argv=['stage_ci_build.py','--profile','webgpu-wasm64-jspi','--variant','browser','--include-toolchain-notices']
        with patch.object(stage_ci_build,'ROOT',self.root), patch.object(sys,'argv',argv), \
             patch.object(stage_ci_build.subprocess,'check_output',return_value=self.source_commit+'\n') as git, \
             patch('builtins.print'):
            stage_ci_build.main()
        git.assert_called_once_with(['git','rev-parse','HEAD'],cwd=self.root,text=True)
        stage=self.root/'build/ci-upload'
        self.assertEqual({p.name for p in stage.iterdir()},{'webgpu-wasm64-jspi','toolchain-licenses'})
        original=self.build/'webgpu-wasm64-jspi/browser/provenance.json'
        self.assertEqual((stage/'webgpu-wasm64-jspi/browser/provenance.json').read_bytes(),original.read_bytes())
        self.assertTrue((stage/'toolchain-licenses/emscripten/LICENSE').is_file())
        self.assertTrue((stage/'toolchain-licenses/emdawnwebgpu_pkg/LICENSE').is_file())

    def test_cli_refuses_staging_outside_build_or_inside_a_profile(self):
        config=self.root/'config'; config.mkdir()
        (config/'profiles.json').write_text(json.dumps(self.profiles))
        for output in (self.root,self.build,self.build/'cpu-wasm32/browser/nested'):
            with self.subTest(output=output):
                argv=['stage_ci_build.py','--profile','cpu-wasm32','--variant','browser','--output',str(output)]
                with patch.object(stage_ci_build,'ROOT',self.root), patch.object(sys,'argv',argv), \
                     patch.object(stage_ci_build.subprocess,'check_output') as git, \
                     patch('sys.stderr'):
                    with self.assertRaises(SystemExit):stage_ci_build.main()
                git.assert_not_called()

    def test_license_selection_matches_original_rules(self):
        (self.sdk/'symlink').symlink_to(self.sdk/'LICENSE')
        (self.sdk/'LICENSE.link').symlink_to(self.sdk/'LICENSE')
        destination=self.root/'notices'
        self.assertEqual(copy_license_notices(self.sdk,destination),2)
        self.assertEqual(set(self.contents(destination)),{'LICENSE','vendor/component/COPYRIGHT.txt'})
        with self.assertRaisesRegex(ValueError,'Missing license'):
            copy_license_notices(self.root/'not-found',destination)

if __name__=='__main__':unittest.main()
