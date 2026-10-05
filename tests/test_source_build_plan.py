"""Synthetic metadata tests of reuse admission; no actual compilation is claimed."""
import json
import shutil
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch
import test_source_package as fixtures
from test_multi_runtime import write_manifest
import plan_source_build as planner

class BuildPlan(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SourcePackage(methodName='test_missing_source_not_silently_replaced');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        for name,path in self.fixture.inputs.items():
            m=json.loads((path/'manifest.json').read_text())
            for profile,p in m['profiles'].items():
                for variant,data in p['variants'].items():
                    data['buildInputFingerprint']=planner.fingerprint(planner.ROOT/'llama-cpp',name,profile,variant)['sha256']
                    scope='syntheticUntrainedModel' if profile.startswith('cpu-') else 'mockedAdapterSuspension'
                    data['validation']={'compiled':True,'browserSmoke':True,scope:{'profile':profile,'variant':variant,'passed':True,'syntheticModel':profile.startswith('cpu-'),'mockedAdapter':profile.startswith('webgpu-'),'suspension':profile.startswith('webgpu-')}}
            write_manifest(path,m)
        self.fixture.build();self.out=self.fixture.out;self.digest=planner.identity(self.out/'manifest.json')['sha256']
    def test_no_prior_artifact_builds_configured_sources(self):
        result=planner.plan();self.assertEqual(len(result['compile']['include']),14);self.assertFalse(result['reuse'])
    def test_exact_inputs_can_be_reused_without_faking_build_commit(self):
        result=planner.plan(self.out,self.digest);self.assertFalse(result['compile']['include'])
        self.assertEqual(set(result['reuse']),{'upstream-stable','upstream-nightly'})
        self.assertEqual(result['reuse']['upstream-nightly']['buildSourceCommit'],'a'*40)
    def test_only_changed_nightly_build_inputs_recompile(self):
        original=planner.fingerprint
        def changed(root,name,profile,variant):
            return {'sha256':'0'*64} if name=='upstream-nightly' else original(root,name,profile,variant)
        with patch.object(planner,'fingerprint',side_effect=changed): result=planner.plan(self.out,self.digest)
        self.assertEqual(len(result['compile']['include']),4)
        self.assertEqual(set(result['reuse']),{'upstream-stable'})
    def test_all_shared_input_changes_recompile(self):
        with patch.object(planner,'fingerprint',return_value={'sha256':'f'*64}):result=planner.plan(self.out,self.digest)
        self.assertEqual(len(result['compile']['include']),14);self.assertFalse(result['reuse'])
    def test_hash_mismatch_and_tampering_never_admit_reuse(self):
        with self.assertRaises(ValueError):planner.plan(self.out,'0'*64)
        p=next(self.out.glob('runtimes/llama-cpp/sources/*/profiles/*/browser/core.wasm'));p.write_bytes(b'wrong')
        with self.assertRaises(ValueError):planner.plan(self.out,self.digest)
    def test_missing_smoke_or_fingerprint_requires_rebuild(self):
        m=json.loads((self.out/'manifest.json').read_text());entry=m['runtimes']['llama-cpp']['sources']['upstream-nightly']
        path=self.out/entry['manifest'];inner=json.loads(path.read_text())
        entry_config=planner.get_source(planner.ROOT/'llama-cpp','upstream-nightly')
        for mode in ['fingerprint','smoke','scope']:
            copy=json.loads(json.dumps(inner));rec=next(iter(next(iter(copy['profiles'].values()))['variants'].values()))
            if mode=='fingerprint':rec.pop('buildInputFingerprint')
            elif mode=='smoke':rec['validation']['browserSmoke']=False
            else:rec['validation']={}
            with patch.object(planner,'read_regular',return_value=json.dumps(copy).encode()):
                ok,reason=planner.can_reuse(self.out,entry,entry_config,planner.ROOT/'llama-cpp')
            self.assertFalse(ok);self.assertTrue(reason)

    def test_runtime_validator_change_rebuilds_without_changing_upstream_pins(self):
        # Copy actual tracked inputs, not a mocked fingerprint. A runtime path and
        # root path with the same basename must remain two independent inputs.
        repo = self.fixture.root / 'changed-repo'
        inventory = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=planner.ROOT)
        for row in inventory.split(b'\0'):
            if not row:
                continue
            metadata, name = row.split(b'\t', 1)
            if metadata.split()[0] not in (b'100644', b'100755'):
                continue
            relative = name.decode()
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(planner.ROOT / relative, destination)
        before = planner.plan(self.out, self.digest, root=repo / 'llama-cpp')
        self.assertEqual(before['compile']['include'], [])
        validator = repo / 'llama-cpp/scripts/package_runtime.py'
        validator.write_text(validator.read_text() + '\n# stricter runtime validation\n')
        after = planner.plan(self.out, self.digest, root=repo / 'llama-cpp')
        self.assertEqual(len(after['compile']['include']), 14)
        self.assertEqual(after['reuse'], {})
        self.assertTrue(all('Build inputs changed' in reason for reason in after['reasons'].values()))
