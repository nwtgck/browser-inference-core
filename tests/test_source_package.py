"""Source catalog packaging tests. Empty Wasm is a packaging fixture, not inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import test_multi_runtime as fixtures
write_manifest=fixtures.write_manifest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('source_catalog',ROOT/'scripts/package_sources.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

class SourcePackage(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.MultiRuntime(methodName='test_assembly_requires_both_runtimes')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.root=self.fixture.root;self.image=self.fixture.inputs/'stable-diffusion-cpp'
        original=self.fixture.inputs/'llama-cpp';self.inputs={}
        for name in source.load_sources(ROOT/'llama-cpp')['sources']:
            entry=source.get_source(ROOT/'llama-cpp',name);dest=self.root/name
            shutil.copytree(original,dest)
            manifest=json.loads((dest/'manifest.json').read_text());template=manifest['profiles']['cpu-wasm32'];manifest['profiles']={}
            for profile in entry['profiles']:
                if profile!='cpu-wasm32':shutil.copytree(dest/'profiles/cpu-wasm32',dest/'profiles'/profile)
                manifest['profiles'][profile]=copy.deepcopy(template)
                for variant,p in manifest['profiles'][profile]['variants'].items():
                    p['profile']=profile;p['sourceId']=name;p['llamaCommit']=entry['commit']
            if 'cpu-wasm32' not in entry['profiles']:shutil.rmtree(dest/'profiles/cpu-wasm32')
            manifest['sourceId']=name;manifest['llamaCommit']=entry['commit']
            write_manifest(dest,manifest);self.inputs[name]=dest
        self.out=self.root/'collection'
    def build(self,**kwargs):
        return source.assemble(self.inputs,self.image,self.out,'c'*40,check_npm_pack=False,**kwargs)
    def test_namespaces_preserve_old_build_provenance_and_raw_glue(self):
        result=self.build()
        self.assertEqual(result['formatVersion'],4)
        for name,path in self.inputs.items():
            entry=result['runtimes']['llama-cpp']['sources'][name]
            self.assertEqual(entry['buildSourceCommit'],'a'*40)
            self.assertEqual((self.out/entry['manifest']).read_bytes(),(path/'manifest.json').read_bytes())
            for raw in path.glob('profiles/*/*/core.wasm'):
                self.assertEqual((self.out/Path(entry['manifest']).parent/raw.relative_to(path)).read_bytes(),raw.read_bytes())
        source.validate(self.out,check_npm_pack=True)
    def test_missing_source_not_silently_replaced(self):
        del self.inputs['upstream-nightly']
        with self.assertRaisesRegex(ValueError,'exactly'):self.build()
        self.assertFalse(self.out.exists())
    def test_wrong_upstream_pin_fails(self):
        path=self.inputs['upstream-nightly'];m=json.loads((path/'manifest.json').read_text());m['llamaCommit']='f'*40
        for p in m['profiles'].values():
            for d in p['variants'].values():d['llamaCommit']='f'*40
        write_manifest(path,m)
        with self.assertRaisesRegex(ValueError,'pin/profiles'):self.build()
    def test_wrong_source_and_profile_inventory_fail(self):
        path=self.inputs['upstream-nightly'];m=json.loads((path/'manifest.json').read_text());m['sourceId']='upstream-stable';write_manifest(path,m)
        with self.assertRaisesRegex(ValueError,'Wrong source|Source track'):self.build()
    def test_payload_tamper_and_symlink_rejected(self):
        self.build();p=self.out/'runtimes/llama-cpp/sources/upstream-nightly/profiles/cpu-wasm32/browser/core.wasm';p.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'identity'):source.validate(self.out,check_npm_pack=False)
        p.unlink();p.symlink_to(self.inputs['upstream-stable']/'profiles/cpu-wasm32/browser/core.wasm')
        with self.assertRaises(ValueError):source.validate(self.out,check_npm_pack=False)
    def test_full_gzip_brotli_pack_does_not_delete_raw_files(self):
        m=self.build(packing={'codecs':['gzip','brotli'],'pairs':[]})
        self.assertEqual(m['packedWasm']['decoderApiVersion'],1)
        c=json.loads((self.out/m['packedWasm']['catalog']).read_text())
        self.assertEqual(len(c['targets']),7)
        for t in c['targets'].values():
            self.assertEqual(len(t['representations']),2)
            self.assertTrue((self.out/t['raw']['path']).is_file())
        source.validate(self.out,check_npm_pack=False)
    def test_file_identity_supports_real_sized_wasm(self):
        p=self.root/'large.wasm';data=b'\0asm'+b'\0'*(5*1024*1024);p.write_bytes(data)
        self.assertEqual(source.identity(p),{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    def test_source_yaml_keeps_build_and_assembly_identities_separate(self):
        self.build(packing={'codecs':['brotli'],'pairs':[]})
        spec=importlib.util.spec_from_file_location('source_report',ROOT/'scripts/consumer_metadata.py')
        reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)
        digest=source.identity(self.out/'manifest.json')['sha256']
        data=reporter.metadata(self.out,'example/core','d'*40,{'specifier':'github:example/core#'+'d'*40},{},published_manifest_sha256=digest)
        self.assertEqual(data['runtime']['sourceCommit'],'c'*40)
        self.assertEqual(data['llamaSources']['upstream-nightly']['buildSourceCommit'],'a'*40)
        self.assertIn('upstream-stable',data['llamaSources'])
        self.assertEqual(data['packedWasm']['codecs'],['brotli'])
        self.assertIn('selectWasmAssets',data['packedWasm']['selectionContract'])
        self.assertIn('plan.runtime.entry',data['packedWasm']['selectionContract'])
        self.assertEqual(data['packedWasm']['loaderEntries']['brotli'], 'packed/llama-cpp/runtime/loader.mjs')
        text=reporter.legacy.write_report(self.root/'report',data,'123','1')
        self.assertIn('```yaml',text);self.assertLess(len(text.encode()),55000)
        self.assertIn('Historical search hints',text)

    def test_reuse_receipt_is_bound_to_actual_source_manifest(self):
        target = self.inputs['upstream-stable']/'manifest.json'
        record = {'artifactCommit':'b'*40, 'manifestIdentity':{'bytes':12, 'sha256':'a'*64},
                  'packages':[{'runtime':'llama-cpp', 'source':'upstream-stable',
                               'manifestIdentity':source.identity(target)}]}
        result=self.build(reuse_inputs=record)
        self.assertEqual(result['reuseInputs'],record)
        for mutate in ['digest','duplicate','ref']:
            changed=copy.deepcopy(record)
            if mutate=='digest': changed['packages'][0]['manifestIdentity']['sha256']='0'*64
            elif mutate=='duplicate': changed['packages'].append(changed['packages'][0])
            else: changed['artifactCommit']='refs/heads/artifacts'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                source.validate_reuse_inputs(changed,result['runtimes'])

    def test_existing_output_and_unknown_assembler_identity(self):
        self.out.mkdir();(self.out/'keep').write_text('keep')
        with self.assertRaises(ValueError):self.build()
        self.assertEqual((self.out/'keep').read_text(),'keep')
        shutil.rmtree(self.out)
        with self.assertRaises(ValueError):source.assemble(self.inputs,self.image,self.out,'HEAD')
