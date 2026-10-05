import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import source_config as sources
import update_llama_cpp as update

class SourceConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.repo=Path(self.tmp.name);self.root=self.repo/'llama-cpp'
        self.root.mkdir()
        for n in ['config','sources','bridge','cmake','scripts','upstream-patches-only-as-a-last-resort-with-explicit-user-approval']:
            shutil.copytree(ROOT/n,self.root/n,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'CMakeLists.txt',self.root/'CMakeLists.txt')
        shutil.copytree(ROOT.parent/'toolchain',self.repo/'toolchain')
        shutil.copytree(ROOT.parent/'scripts',self.repo/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    def test_nightly_two_profiles_are_configuration_not_code(self):
        matrix=sources.matrix(self.root,['upstream-nightly'])
        self.assertEqual(len(matrix),4)
        self.assertEqual({v['profile'] for v in matrix},{'cpu-wasm32','webgpu-wasm64-jspi'})
        p=self.root/'config/sources.json';c=json.loads(p.read_text());c['sources']['upstream-nightly']['profiles'].append('webgpu-wasm32-jspi');p.write_text(json.dumps(c))
        self.assertEqual(len(sources.matrix(self.root,['upstream-nightly'])),6)
    def test_pins_and_paths_are_independent(self):
        a=sources.get_source(self.root,'upstream-stable');b=sources.get_source(self.root,'upstream-nightly')
        self.assertNotEqual(a['vendorPath'],b['vendorPath']);self.assertNotEqual(a['pinFile'],b['pinFile'])
        self.assertNotEqual(a['commit'],b['commit'])
    def test_nightly_pin_change_does_not_invalidate_stable(self):
        def key(source):return sources.fingerprint(self.root,source,'cpu-wasm32','browser')['sha256']
        a=key('upstream-stable');b=key('upstream-nightly')
        p=self.root/'sources/upstream-nightly/pin.json';c=json.loads(p.read_text());c['llamaCommit']='b'*40;p.write_text(json.dumps(c))
        self.assertEqual(a,key('upstream-stable'));self.assertNotEqual(b,key('upstream-nightly'))
    def test_shared_bridge_change_propagates_to_both(self):
        def keys():return [sources.fingerprint(self.root,s,'cpu-wasm32','browser')['sha256'] for s in ['upstream-stable','upstream-nightly']]
        before=keys();p=self.root/'bridge/new-support.h';p.write_text('// an actual shared input\n');after=keys()
        self.assertTrue(all(a!=b for a,b in zip(before,after)))
    def test_reducing_profile_set_does_not_change_remaining_binary_inputs(self):
        before=sources.fingerprint(self.root,'upstream-nightly','cpu-wasm32','browser')['sha256']
        path=self.root/'config/sources.json';data=json.loads(path.read_text())
        data['sources']['upstream-nightly']['profiles']=['cpu-wasm32'];path.write_text(json.dumps(data))
        self.assertEqual(before,sources.fingerprint(self.root,'upstream-nightly','cpu-wasm32','browser')['sha256'])
        self.assertEqual(len(sources.matrix(self.root,['upstream-nightly'])),2)
        with self.assertRaises(ValueError):
            sources.fingerprint(self.root,'upstream-nightly','webgpu-wasm64-jspi','browser')

    def test_build_input_symlink_is_not_silently_excluded(self):
        (self.root/'bridge/linked.h').symlink_to(self.root/'CMakeLists.txt')
        with self.assertRaisesRegex(ValueError,'Linked'):
            sources.fingerprint(self.root,'upstream-stable','cpu-wasm32','browser')

    def test_workflow_change_propagates_to_sources(self):
        def keys():return [sources.fingerprint(self.root,s,'cpu-wasm32','browser')['sha256'] for s in ['upstream-stable','upstream-nightly']]
        before=keys();path=self.repo/'.github/workflows/build.yml';path.parent.mkdir(parents=True)
        path.write_text('a build or validation recipe change')
        self.assertTrue(all(a!=b for a,b in zip(before,keys())))

    def test_fingerprint_namespaces_do_not_hide_runtime_validator_changes(self):
        def keys():
            return [sources.fingerprint(self.root, name, 'cpu-wasm32', 'browser')['sha256']
                    for name in ('upstream-stable', 'upstream-nightly')]
        for relative in ('llama-cpp/scripts/package_runtime.py',
                         'scripts/package_runtime.py',
                         'llama-cpp/scripts/publish_artifacts.py',
                         'scripts/publish_artifacts.py'):
            with self.subTest(relative=relative):
                before = keys()
                path = self.repo / relative
                path.write_text(path.read_text() + '\n# changed validation input\n')
                self.assertTrue(all(a != b for a, b in zip(before, keys())))
        inputs = sources.fingerprint(self.root, 'upstream-stable', 'cpu-wasm32', 'browser')['inputs']
        self.assertIn('llama-cpp/scripts/package_runtime.py', inputs['inputs'])
        self.assertIn('scripts/package_runtime.py', inputs['inputs'])

    def test_patch_identity_must_not_disappear_or_change_silently(self):
        entry=sources.get_source(self.root,'upstream-nightly');plan=sources.patch_series(self.root,entry)
        self.assertEqual({p['component'] for p in plan},{'vision','audio'})
        p=self.root/plan[0]['file'];p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):sources.patch_series(self.root,entry)
    def test_unknown_source_and_invalid_matrix_fail(self):
        for names in [[],['upstream-nightly','upstream-nightly'],['fork-does-not-exist']]:
            with self.assertRaises(ValueError):sources.matrix(self.root,names)
    def test_nightly_preflight_uses_selected_vendor_and_series(self):
        entry=sources.get_source(self.root,'upstream-nightly')
        with patch.object(update,'prepare') as prepare:
            result=update.overlay_preflight(self.root,entry)
        self.assertEqual(result['status'],'passed');self.assertEqual(prepare.call_count,2)
        self.assertTrue(all(c.args[0]==self.root/entry['vendorPath'] for c in prepare.call_args_list))
    def test_tracks_do_not_collide_on_same_upstream_target(self):
        self.assertNotEqual(update.candidate_branch('main','a'*40,'b'*40,'upstream-stable'),update.candidate_branch('main','a'*40,'b'*40,'upstream-nightly'))
    def test_shared_pin_file_or_vendor_is_rejected(self):
        p=self.root/'config/sources.json'
        for field in ['vendorPath','pinFile']:
            config=json.loads(p.read_text());original=config['sources']['upstream-nightly'][field]
            config['sources']['upstream-nightly'][field]=config['sources']['upstream-stable'][field];p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):sources.load_sources(self.root)
            config['sources']['upstream-nightly'][field]=original;p.write_text(json.dumps(config))

if __name__=='__main__':unittest.main()
