"""Local synthetic packages and Git remotes; not real-model/Actions evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_source_package as fixtures
from test_multi_runtime import write_manifest
import pipeline_plan as pipeline
import plan_source_build as llama
import restore_source_package as restore
import assemble_source_plan as assemble

ROOT=Path(__file__).resolve().parents[1]

def git(*args, cwd):
    return subprocess.check_output(['git',*map(str,args)],cwd=cwd,text=True).strip()

class PipelinePlan(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SourcePackage(methodName='test_missing_source_not_silently_replaced')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        for name,path in self.fixture.inputs.items():
            manifest=json.loads((path/'manifest.json').read_text())
            for profile,p in manifest['profiles'].items():
                for variant,data in p['variants'].items():
                    data['buildInputFingerprint']=llama.fingerprint(ROOT/'llama-cpp',name,profile,variant)['sha256']
                    scope='syntheticUntrainedModel' if profile.startswith('cpu-') else 'mockedAdapterSuspension'
                    data['validation']={'compiled':True,'browserSmoke':True,scope:{'profile':profile,'variant':variant,'passed':True,'syntheticModel':profile.startswith('cpu-'),'mockedAdapter':profile.startswith('webgpu-'),'suspension':profile.startswith('webgpu-')}}
            write_manifest(path,manifest)
        image=self.fixture.image;manifest=json.loads((image/'manifest.json').read_text())
        # Fixture records attest only simulated smoke; never presented as actual inference.
        pins=json.loads((ROOT/'stable-diffusion-cpp/config/upstreams.json').read_text())
        manifest['upstreams']=pins
        for profile,p in manifest['profiles'].items():
            for variant,data in p['variants'].items():
                data['upstreams']=pins
                data['buildInputFingerprint']=pipeline.image_fingerprint(ROOT,profile,variant)
                data['validation']={'compiled':True,'browserSmoke':True,'browserSmokeScope':pipeline.image_scope(False)}
        write_manifest(image,manifest)
        self.fixture.build();self.previous=self.fixture.out
        self.receipt={'artifactCommit':'b'*40,'manifestIdentity':pipeline.identity(self.previous/'manifest.json')}
        self.reuse=self.root/'reuse'

    def make_plan(self):
        return pipeline.create_plan(self.previous,self.receipt,self.reuse)

    def test_unchanged_packages_keep_original_bytes_and_need_no_compilation(self):
        result=self.make_plan()
        self.assertEqual(result['compile']['include'],[])
        self.assertIsNotNone(result['imageReuse'])
        self.assertTrue(all(s['reuse'] for s in result['sources']['include']))
        for name,path in self.fixture.inputs.items():
            self.assertEqual((path/'manifest.json').read_bytes(),(self.reuse/'llama-cpp'/name/'manifest.json').read_bytes())
        self.assertEqual(result['reuse']['upstream-stable']['buildSourceCommit'],'a'*40)

    def test_nightly_change_only_schedules_four_compile_jobs_and_reuses_image(self):
        original=llama.fingerprint
        def changed(root,name,profile,variant):
            return {'sha256':'0'*64} if name=='upstream-nightly' else original(root,name,profile,variant)
        with patch.object(llama,'fingerprint',side_effect=changed):result=self.make_plan()
        self.assertEqual(len(result['compile']['include']),4)
        self.assertEqual({i['source'] for i in result['compile']['include']},{'upstream-nightly'})
        self.assertEqual(set(result['reuse']),{'upstream-stable'})
        self.assertIsNotNone(result['imageReuse'])
        self.assertFalse((self.reuse/'llama-cpp/upstream-nightly').exists())

    def test_image_build_input_change_does_not_recompile_llama(self):
        with patch.object(pipeline,'image_fingerprint',return_value='f'*64):result=self.make_plan()
        self.assertFalse(result['compile']['include']);self.assertIsNone(result['imageReuse'])

    def test_missing_previous_builds_all_configured_sources(self):
        result=pipeline.create_plan(None,None,self.reuse)
        self.assertEqual(len(result['compile']['include']),14);self.assertIsNone(result['imageReuse'])
        self.assertEqual(len(result['sources']['include']),2)

    def test_legacy_artifact_is_cold_not_an_implicit_stable_package(self):
        path=self.root/'legacy';path.mkdir();(path/'manifest.json').write_text('{"formatVersion":3}')
        result=pipeline.create_plan(path,{'artifactCommit':'c'*40,'manifestIdentity':pipeline.identity(path/'manifest.json')},self.reuse)
        self.assertEqual(len(result['compile']['include']),14);self.assertIsNone(result['imageReuse'])

    def test_bad_receipt_stops_before_reuse_output(self):
        self.receipt['manifestIdentity']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'receipt'):self.make_plan()
        self.assertFalse(self.reuse.exists())

    def test_corrupt_payload_cannot_be_reused_or_silently_rebuilt(self):
        p=next(self.previous.glob('runtimes/llama-cpp/sources/*/profiles/*/*/core.wasm'));p.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'hash/size'):self.make_plan()

    def test_duplicate_root_json_key_rejected(self):
        path=self.previous/'manifest.json';data=path.read_text().rstrip();path.write_text(data[:-1]+',"formatVersion":4}')
        self.receipt['manifestIdentity']=pipeline.identity(path)
        with self.assertRaisesRegex(ValueError,'Duplicate JSON'):self.make_plan()

    def test_old_codec_entry_not_executed_for_raw_reuse(self):
        path=self.previous/'manifest.json';data=json.loads(path.read_text())
        data['packedWasm']={'catalog':'a-future-format.json','decoderApiVersion':12345}
        path.write_text(json.dumps(data));self.receipt['manifestIdentity']=pipeline.identity(path)
        with patch.object(pipeline,'validate',side_effect=AssertionError('Old codec execution forbidden')):
            result=self.make_plan()
        self.assertFalse(result['compile']['include'])

    def test_source_namespace_swap_rejected_even_with_valid_payload_hashes(self):
        path=self.previous/'manifest.json';data=json.loads(path.read_text())
        data['runtimes']['llama-cpp']['sources']['upstream-stable']['manifest']='runtimes/llama-cpp/sources/upstream-nightly/manifest.json'
        path.write_text(json.dumps(data));self.receipt['manifestIdentity']=pipeline.identity(path)
        with self.assertRaisesRegex(ValueError,'Unbound'):self.make_plan()

    def test_absent_smoke_rebuilds_only_that_source(self):
        path=self.fixture.inputs['upstream-nightly'];data=json.loads((path/'manifest.json').read_text())
        for p in data['profiles'].values():
            for v in p['variants'].values():v['validation']['browserSmoke']=False
        write_manifest(path,data);shutil.rmtree(self.previous);self.fixture.build()
        self.receipt['manifestIdentity']=pipeline.identity(self.previous/'manifest.json')
        result=self.make_plan();self.assertEqual(len(result['compile']['include']),4)

    def test_restore_rechecks_manifest_and_preserves_original_commit(self):
        plan=self.make_plan();plan['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT)
        (self.reuse/'plan.json').write_text(json.dumps(plan))
        out=self.root/'restored'
        with patch.dict(os.environ,{'LCB_SOURCE_COMMIT':plan['sourceCommit']}):
            restore.restore(self.reuse,out,'llama-cpp','upstream-nightly')
        self.assertEqual(json.loads((out/'manifest.json').read_text())['sourceCommit'],'a'*40)
        raw=self.reuse/'llama-cpp/upstream-nightly/manifest.json';raw.write_bytes(raw.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'manifest'):
            restore.restore(self.reuse,self.root/'bad','llama-cpp','upstream-nightly')

    def test_restore_image_and_wrong_run_refusal(self):
        plan=self.make_plan();plan['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT)
        (self.reuse/'plan.json').write_text(json.dumps(plan))
        restore.restore(self.reuse,self.root/'image-restored','stable-diffusion-cpp',None)
        plan['sourceCommit']='0'*40;(self.reuse/'plan.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'revision'):
            restore.restore(self.reuse,self.root/'wrong-run','stable-diffusion-cpp',None)

    def test_full_plan_assembly_with_brotli_gzip_and_original_raw_packages(self):
        plan=self.make_plan();plan['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT)
        path=self.root/'plan.json';path.write_text(json.dumps(plan))
        output=self.root/'assembled'
        result=assemble.assemble_plan(path,self.fixture.inputs,self.fixture.image,output,
                                     packing={'codecs':['gzip','brotli'],'pairs':[]})
        self.assertEqual(result['sourceCommit'],plan['sourceCommit'])
        self.assertEqual(result['runtimes']['llama-cpp']['sources']['upstream-nightly']['buildSourceCommit'],'a'*40)
        self.assertIn('packedWasm',result)
        self.assertEqual(result['reuseInputs']['artifactCommit'], self.receipt['artifactCommit'])
        self.assertEqual(len(result['reuseInputs']['packages']), 3)
        self.assertEqual(result['reuseInputs']['manifestIdentity'], self.receipt['manifestIdentity'])

    def test_reuse_without_origin_receipt_is_not_published(self):
        plan=self.make_plan();plan['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT);plan['previous']=None
        path=self.root/'plan.json';path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'previous artifact receipt'):
            assemble.assemble_plan(path,self.fixture.inputs,self.fixture.image,self.root/'bad',packing=None)

    def test_v4_local_publication_report_and_next_run_reuse(self):
        import io
        import zipfile
        import publish_artifacts as publisher
        import consumer_metadata as reporter
        import runtime_comments
        remote=self.root/'published.git'
        git('init','--bare','--quiet',str(remote),cwd=self.root)
        digest=pipeline.identity(self.previous/'manifest.json')['sha256']
        published=publisher.publish(self.previous,str(remote),expected_manifest_sha256=digest)
        snapshot=self.root/'fetched';receipt=pipeline.fetch_snapshot(str(remote),snapshot)
        self.assertEqual(receipt['artifactCommit'],published)
        self.assertEqual(receipt['manifestIdentity']['sha256'],digest)
        next_plan=pipeline.create_plan(snapshot,receipt,self.root/'next-reuse')
        self.assertFalse(next_plan['compile']['include'])
        self.assertIsNotNone(next_plan['imageReuse'])
        data=reporter.metadata(snapshot,'example/core',published,
            {'specifier':'github:example/core#'+published},{},published_manifest_sha256=digest)
        report=self.root/'report';text=reporter.legacy.write_report(report,data,'123','2')
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as out:
            for name in runtime_comments.REPORT_FILES:out.writestr(name,(report/name).read_bytes())
        decoded=runtime_comments.decode_report(archive.getvalue(),'example/core',
                         {'id':123,'run_attempt':2,'head_sha':'c'*40})
        self.assertEqual(decoded,text)
        self.assertIn(published,decoded)
        self.assertIn('llamaSources:',decoded)

    def test_fresh_package_cannot_claim_an_old_build(self):
        plan=self.make_plan();plan['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT);plan['reuse'].pop('upstream-nightly')
        path=self.root/'plan.json';path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'another commit'):
            assemble.assemble_plan(path,self.fixture.inputs,self.fixture.image,self.root/'bad',packing=None)

class ArtifactFetch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.repo=self.root/'repo';self.repo.mkdir()
        git('init','-q',cwd=self.repo);git('config','user.name','Fixture',cwd=self.repo);git('config','user.email','fixture@example.invalid',cwd=self.repo)
        (self.repo/'manifest.json').write_text('{"formatVersion":3}')
        git('add','.',cwd=self.repo);git('commit','-qm','fixture',cwd=self.repo)
        self.commit=git('rev-parse','HEAD',cwd=self.repo);git('branch','artifacts',cwd=self.repo)

    def test_local_git_resolution_is_bound_to_full_commit_and_raw_bytes(self):
        out=self.root/'out';receipt=pipeline.fetch_snapshot(str(self.repo),out)
        self.assertEqual(receipt['artifactCommit'],self.commit)
        self.assertEqual((out/'manifest.json').read_bytes(),(self.repo/'manifest.json').read_bytes())
        self.assertFalse((out/'.git').exists())

    def test_missing_artifact_branch_is_cold_and_source_branch_forbidden(self):
        git('branch','-D','artifacts',cwd=self.repo)
        self.assertIsNone(pipeline.fetch_snapshot(str(self.repo),self.root/'out'))
        with self.assertRaisesRegex(ValueError,'artifact branches'):
            pipeline.fetch_snapshot(str(self.repo),self.root/'out',ref='refs/heads/main')

    def test_symlink_is_rejected_before_materialization(self):
        (self.repo/'link').symlink_to('/etc/passwd');git('add','link',cwd=self.repo);git('commit','-qm','symlink',cwd=self.repo)
        commit=git('rev-parse','HEAD',cwd=self.repo)
        with self.assertRaisesRegex(ValueError,'Linked'):
            pipeline.extract_snapshot(self.repo,commit,self.root/'bad')
        self.assertFalse((self.root/'bad/link').exists())

    def test_hidden_control_files_and_case_collisions_rejected(self):
        (self.repo/'.gitattributes').write_text('* text');git('add','.',cwd=self.repo);git('commit','-qm','attrs',cwd=self.repo)
        with self.assertRaisesRegex(ValueError,'Hidden'):
            pipeline.extract_snapshot(self.repo,git('rev-parse','HEAD',cwd=self.repo),self.root/'attrs')
        git('rm','.gitattributes',cwd=self.repo)
        (self.repo/'File').write_text('1');(self.repo/'file').write_text('2')
        git('add','.',cwd=self.repo);git('commit','-qm','case collision',cwd=self.repo)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            pipeline.extract_snapshot(self.repo,git('rev-parse','HEAD',cwd=self.repo),self.root/'case')

    def test_non_commit_and_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'full artifact commit'):
            pipeline.extract_snapshot(self.repo,'HEAD',self.root/'bad')
        out=self.root/'exists';out.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):
            pipeline.extract_snapshot(self.repo,self.commit,out)
