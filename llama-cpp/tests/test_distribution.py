"""Use empty Wasm fixtures to test distribution, not inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from package_runtime import VARIANTS, EMBEDDED_NOTICE_FILES, build_package, validate
from publish_artifacts import publish
import publish_artifacts


def git(*args,cwd=None):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()

class Distribution(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lcb-test-')
        self.root=Path(self.tmp.name)
        self.provenance={'profile':'cpu-wasm32','sourceCommit':'a'*40,'llamaCommit':'b'*40,'sourceDirty':False,
                         'validation':{'fixtureOnly':True}}
        for variant,settings in VARIANTS.items():
            build=self.root/'build/cpu-wasm32'/variant
            (build/'runtime').mkdir(parents=True); (build/'generated').mkdir()
            (build/'runtime/core.wasm').write_bytes(b'\0asm\1\0\0\0')
            (build/'runtime/core.mjs').write_text(f'export default async () => ({{ fixture: true, variant: {json.dumps(variant)} }});\n')
            (build/'runtime/core.d.ts').write_text('export default function create(): Promise<{ fixture: boolean }>;\n')
            for name,content in {'schema.json':'{}','schema.mjs':'export default {};',
                                 'functions.d.ts':'export interface LowLevelFunctions {}', 'exports.json':'[]'}.items():
                (build/'generated'/name).write_text(content)
            data={**self.provenance,'variant':variant,'variantConfiguration':settings}
            (build/'provenance.json').write_text(json.dumps(data))
        licenses=self.root/'notices'; licenses.mkdir(); (licenses/'LICENSE').write_text('Test-only notice')
        self.licenses=licenses; self.package=self.root/'package'
        self.upstream=self.root/'upstream-notice-fixture'
        for relative in EMBEDDED_NOTICE_FILES:
            path=self.upstream/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('Synthetic embedded notice fixture, not upstream source.\n')
        self.assemble()
    def tearDown(self): self.tmp.cleanup()
    def assemble(self):
        return build_package(self.root/'build',self.package,['cpu-wasm32'],license_roots=[self.licenses],upstream_source=self.upstream)
    def test_runtime_has_no_build_hooks_or_submodule(self):
        result=validate(self.package)
        self.assertEqual(result['profiles'],['cpu-wasm32'])
        package=json.loads((self.package/'package.json').read_text())
        self.assertNotIn('scripts',package)
        self.assertNotIn('devDependencies',package)
        self.assertFalse((self.package/'.gitmodules').exists())
    def test_payload_tampering_is_rejected(self):
        (self.package/'profiles/cpu-wasm32/browser/core.wasm').write_bytes(b'bad')
        with self.assertRaises(ValueError): validate(self.package)
    def test_both_variant_exports_resolve_their_own_runtime_pair(self):
        # Package self-references exercise the same wildcard exports as consumers,
        # without installing dependencies or running generated Wasm.
        code='''\
import assert from 'node:assert/strict';
for (const variant of ['browser', 'test']) {
  const entry = `llama-cpp-browser-core/profiles/cpu-wasm32/${variant}/core.mjs`;
  const { default: create } = await import(entry);
  assert.equal((await create()).fixture, true);
  assert.equal((await create()).variant, variant);
  const wasm = import.meta.resolve(`llama-cpp-browser-core/profiles/cpu-wasm32/${variant}/core.wasm`);
  assert.equal(new URL('core.wasm', import.meta.resolve(entry)).href, wasm);
}
'''
        subprocess.run(['node','--input-type=module','-e',code],cwd=self.package,check=True)
    def test_missing_variant_or_variant_provenance_is_rejected(self):
        path=self.package/'manifest.json'; original=path.read_text()
        for change in ('missing','mislabelled','configuration'):
            manifest=json.loads(original)
            variants=manifest['profiles']['cpu-wasm32']['variants']
            if change=='missing': del variants['test']
            elif change=='mislabelled': variants['browser']['variant']='test'
            else: variants['browser']['variantConfiguration']=VARIANTS['test']
            path.write_text(json.dumps(manifest))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,'[Vv]ariant'):
                validate(self.package)
        path.write_text(original)
    def test_assembly_requires_the_complete_test_variant_pair(self):
        (self.root/'build/cpu-wasm32/test/runtime/core.wasm').unlink()
        with self.assertRaisesRegex(ValueError,'Missing cpu-wasm32/test runtime'):
            self.assemble()
    def test_embedded_upstream_notices_are_preserved_verbatim(self):
        # These libraries carry notices inside source, not standalone LICENSE files.
        for relative in ('vendor/miniaudio/miniaudio.h', 'vendor/stb/stb_image.h',
                         'vendor/nlohmann/json.hpp', 'vendor/nlohmann/json_fwd.hpp',
                         'vendor/sheredom/subprocess.h', 'vendor/hash/sha1/sha1.c'):
            with self.subTest(source=relative):
                original=self.upstream/relative
                packaged=self.package/'licenses/embedded'/(relative+'.txt')
                self.assertEqual(packaged.read_bytes(), original.read_bytes())
    def test_removed_notice_is_rejected_even_if_manifest_is_updated(self):
        relative='licenses/embedded/vendor/miniaudio/miniaudio.h.txt'
        (self.package/relative).unlink()
        path=self.package/'manifest.json'
        manifest=json.loads(path.read_text())
        manifest['files']=[entry for entry in manifest['files'] if entry['path'] != relative]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Missing embedded third-party license notices'):
            validate(self.package)
    def test_example_runtime_is_shipped_with_resolvable_relative_imports(self):
        for name in ('index.mjs', 'index.d.ts', 'bindings.mjs', 'read-only-file.mjs', 'README.md'):
            self.assertEqual((self.package/'examples/runtime'/name).read_bytes(),
                             (ROOT/'examples/runtime'/name).read_bytes())
        package=json.loads((self.package/'package.json').read_text())
        self.assertEqual(package['exports']['./examples/runtime']['import'], './examples/runtime/index.mjs')
        code="import('./examples/runtime/index.mjs').then(m => { if (typeof m.createCore !== 'function') throw Error('missing example loader'); })"
        subprocess.run(['node', '--input-type=module', '-e', code], cwd=self.package, check=True)
    def test_missing_example_runtime_is_rejected_even_if_manifest_is_updated(self):
        relative='examples/runtime/bindings.mjs'
        (self.package/relative).unlink()
        path=self.package/'manifest.json'
        manifest=json.loads(path.read_text())
        manifest['files']=[entry for entry in manifest['files'] if entry['path'] != relative]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Missing example runtime files'):
            validate(self.package)
    def test_extra_file_is_rejected(self):
        (self.package/'unlisted').write_text('stale asset')
        with self.assertRaises(ValueError): validate(self.package)
    def test_dirty_source_cannot_be_published(self):
        path=self.root/'build/cpu-wasm32/browser/provenance.json'
        value=json.loads(path.read_text()); value['sourceDirty']=True; path.write_text(json.dumps(value)); self.assemble()
        with self.assertRaises(ValueError): validate(self.package)
    def test_dirty_source_error_identifies_profile_and_changed_paths(self):
        path=self.root/'build/cpu-wasm32/browser/provenance.json'
        value=json.loads(path.read_text())
        value.update({'sourceDirty':True,'sourceStatusBeforeBuild':[],
                      'sourceStatusAfterBuild':['?? configure-probe.tmp']})
        path.write_text(json.dumps(value)); self.assemble()
        with self.assertRaises(ValueError) as error:
            validate(self.package)
        self.assertIn('cpu-wasm32',str(error.exception))
        self.assertIn('configure-probe.tmp',str(error.exception))
        self.assertIn('sourceStatusAfterBuild',str(error.exception))

    def test_browser_validation_does_not_clear_dirty_provenance(self):
        script=self.root/'scripts/record_browser_validation.py'
        script.parent.mkdir()
        shutil.copy2(ROOT/'scripts/record_browser_validation.py',script)
        results=[]
        for profile in ('cpu-wasm32','cpu-wasm64','webgpu-wasm32-jspi','webgpu-wasm64-jspi','webgpu-wasm32-asyncify'):
            for variant,settings in VARIANTS.items():
                path=self.root/'build'/profile/variant/'provenance.json'
                path.parent.mkdir(parents=True,exist_ok=True)
                value={**self.provenance,'profile':profile,'variant':variant,'variantConfiguration':settings,
                       'sourceDirty':True,'sourceStatusBeforeBuild':[' M README.md'],'sourceStatusAfterBuild':[]}
                path.write_text(json.dumps(value))
                results.append({'profile':profile,'variant':variant,'passed':True,
                                'syntheticModel':profile.startswith('cpu-')})
        (self.root/'build/browser-results.json').write_text(json.dumps(results))
        subprocess.run([sys.executable,str(script)],check=True,capture_output=True,text=True)
        for result in results:
            value=json.loads((self.root/'build'/result['profile']/result['variant']/'provenance.json').read_text())
            self.assertTrue(value['sourceDirty'])
            self.assertTrue(value['validation']['browserSmoke'])
            self.assertEqual(value['sourceStatusBeforeBuild'],[' M README.md'])
            scope='syntheticUntrainedModel' if result['syntheticModel'] else 'mockedAdapterSuspension'
            self.assertEqual(value['validation'][scope],result)
        self.assemble()
        with self.assertRaises(ValueError): validate(self.package)

    def test_browser_validation_rejects_results_without_the_test_variant(self):
        script=self.root/'scripts/record_browser_validation.py'
        script.parent.mkdir()
        shutil.copy2(ROOT/'scripts/record_browser_validation.py',script)
        results=[{'profile':profile,'variant':'browser','passed':True}
                 for profile in ('cpu-wasm32','cpu-wasm64','webgpu-wasm32-jspi',
                                 'webgpu-wasm64-jspi','webgpu-wasm32-asyncify')]
        (self.root/'build/browser-results.json').write_text(json.dumps(results))
        result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing or failed profile/variant browser smoke tests',result.stderr)
        for variant in VARIANTS:
            value=json.loads((self.root/'build/cpu-wasm32'/variant/'provenance.json').read_text())
            self.assertNotIn('browserSmoke',value['validation'])

    def test_append_only_publication_preserves_old_sha_and_npm_install(self):
        remote=self.root/'remote.git'; git('init','--bare','--quiet',str(remote))
        first=publish(self.package,str(remote))
        # Change a payload and remove an old auxiliary asset by repackaging from scratch.
        (self.root/'build/cpu-wasm32/browser/runtime/core.mjs').write_text('export default async () => ({ fixture: 2 });\n')
        self.assemble()
        second=publish(self.package,str(remote))
        self.assertNotEqual(first,second)
        for commit in (first,second):
            self.assertEqual(git('--git-dir='+str(remote),'show','-s','--format=%an <%ae>',commit),
                             'github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>')
            self.assertEqual(git('--git-dir='+str(remote),'show','-s','--format=%cn <%ce>',commit),
                             'github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>')
            self.assertEqual(git('--git-dir='+str(remote),'show','-s','--format=%s',commit),
                             'build(artifacts): publish runtime from '+self.provenance['sourceCommit'][:12])
            body=git('--git-dir='+str(remote),'show','-s','--format=%B',commit)
            self.assertIn('Source-commit: '+self.provenance['sourceCommit'],body)
            self.assertNotIn('Co-authored-by:',body)
        self.assertEqual(git('--git-dir='+str(remote),'rev-parse',second+'^'),first)
        self.assertEqual(git('--git-dir='+str(remote),'rev-list','--max-parents=0',second),first)
        self.assertNotIn('160000',git('--git-dir='+str(remote),'ls-tree','-r',second))
        install=self.root/'consumer'; install.mkdir()
        (install/'package.json').write_text('{"private":true}')
        command=['npm','install','--no-audit','--no-fund',f'git+file://{remote}#{first}']
        subprocess.run(command,cwd=install,check=True,capture_output=True,text=True)
        copied=install/'node_modules/llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs'
        self.assertIn('fixture: true',copied.read_text())
        shutil.rmtree(install/'node_modules')
        subprocess.run(['npm','ci','--no-audit','--no-fund','--cache',str(self.root/'empty-cache')],
                       cwd=install,check=True,capture_output=True,text=True)
        self.assertIn('fixture: true',copied.read_text())
    def test_publication_identity_does_not_inherit_caller_identity(self):
        remote=self.root/'identity.git'; git('init','--bare','--quiet',str(remote))
        inherited={'GIT_AUTHOR_NAME':'Local Developer','GIT_AUTHOR_EMAIL':'developer@example.invalid',
                   'GIT_COMMITTER_NAME':'Local Committer','GIT_COMMITTER_EMAIL':'committer@example.invalid'}
        with patch.dict(os.environ,inherited):
            commit=publish(self.package,str(remote))
            for key,value in inherited.items():
                self.assertEqual(os.environ[key],value)
        bot='github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>'
        for identity in ('%an <%ae>','%cn <%ce>'):
            self.assertEqual(git('--git-dir='+str(remote),'show','-s','--format='+identity,commit),bot)
        body=git('--git-dir='+str(remote),'show','-s','--format=%B',commit)
        self.assertIn('Source-commit: '+self.provenance['sourceCommit'],body)
        self.assertNotIn('Co-authored-by:',body)

    def test_publication_summary_uses_project_or_workflow_repository(self):
        for workflow_repo,expected_repo in ((None,'nwtgck/browser-inference-core'),
                                            ('example-owner/core-fork','example-owner/core-fork')):
            with self.subTest(workflow_repo=workflow_repo):
                summary=self.root/'summary.md'; output=self.root/'github-output'
                summary.unlink(missing_ok=True); output.unlink(missing_ok=True)
                env={key:value for key,value in os.environ.items()
                     if key not in ('GH_TOKEN','GITHUB_REPOSITORY','GITHUB_STEP_SUMMARY','GITHUB_OUTPUT')}
                env.update({'GITHUB_STEP_SUMMARY':str(summary),'GITHUB_OUTPUT':str(output)})
                if workflow_repo is not None:
                    env['GITHUB_REPOSITORY']=workflow_repo
                commit='c'*40
                argv=['publish_artifacts.py','--package',str(self.package),
                      '--remote','https://github.com/'+expected_repo+'.git']
                with patch.dict(os.environ,env,clear=True), patch.object(sys,'argv',argv), \
                     patch.object(publish_artifacts,'publish',return_value=commit) as publish_mock, \
                     patch('builtins.print'):
                    publish_artifacts.main()
                publish_mock.assert_called_once_with(self.package.resolve(),argv[-1],'artifacts')
                self.assertIn('npm install github:'+expected_repo+'#'+commit,summary.read_text())
                self.assertNotIn('OWNER/',summary.read_text())
                self.assertEqual(output.read_text(),'commit='+commit+'\n')

    def test_source_branch_publication_is_rejected(self):
        with self.assertRaises(ValueError): publish(self.package,str(self.root/'unused.git'),'main')

if __name__=='__main__': unittest.main()
