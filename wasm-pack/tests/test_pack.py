import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
import pack


def leb(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def wasm_fixture(value):
    # Valid module: repeated changed constants train a prediction rule; data in a
    # custom section exercises arbitrary byte matching without invalid opcodes.
    def section(kind, payload):
        return bytes([kind]) + leb(len(payload)) + payload
    def integer(value):
        out = bytearray()
        while True:
            byte = value & 127
            value >>= 7
            if value == 0 and byte & 64 == 0:
                out.append(byte)
                return bytes(out)
            out.append(byte | 128)
    body = b'\x00' + (b'\x41' + integer(value) + b'\x1a') * 12 + b'\x41' + integer(value) + b'\x0b'
    return (b'\x00asm\x01\x00\x00\x00'
            + section(1, b'\x01\x60\x00\x01\x7f')
            + section(3, b'\x01\x00')
            + section(7, b'\x01\x01f\x00\x00')
            + section(10, b'\x01' + leb(len(body)) + body)
            + section(0, b'\x04data' + bytes(range(256)) * 20))

class PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.encoder = Path(os.environ.get('BIC_TEST_WASM_ENCODER', ROOT.parent / 'build/wasm-pack-tools/wasm-encoder'))
        if not cls.encoder.is_file():
            raise RuntimeError('Build the native encoder with wasm-pack/build_tools.py first')
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.a = wasm_fixture(256)
        self.b = wasm_fixture(257)
        self.targets={}
        for id,data in [('stable--cpu-wasm32',self.a),('nightly--cpu-wasm32',self.b)]:
            file=id+'.wasm';(self.root/file).write_bytes(data)
            self.targets[id]={'identity':{'runtime':'llama-cpp','source':id.split('--')[0],'profile':'cpu-wasm32','variant':'browser'},'raw':{'path':file,**pack.identity(data)}}
    def test_full_and_delta_all_alternatives_roundtrip(self):
        report=pack.build(self.root,{'targets':self.targets},self.root/'pack',encoder=self.encoder)
        self.assertEqual(len(report['pairs']),2)
        c=json.loads((self.root/'pack/catalog.json').read_text())
        self.assertEqual(len(c['targets']),2)
        for t in c['targets'].values():self.assertEqual(len(t['representations']),6)
        self.assertEqual((self.root/'stable--cpu-wasm32.wasm').read_bytes(),self.a)
    def test_full_only_and_determinism(self):
        for name in ('first','second'):pack.build(self.root,{'targets':self.targets},self.root/name,codecs=['gzip'],pairs=[])
        a={p.relative_to(self.root/'first').as_posix():p.read_bytes() for p in (self.root/'first').rglob('*') if p.is_file()}
        b={p.relative_to(self.root/'second').as_posix():p.read_bytes() for p in (self.root/'second').rglob('*') if p.is_file()}
        self.assertEqual(a,b)
    def test_invalid_inputs_never_publish_output(self):
        for field,value in [('sha256','0'*64),('bytes',1),('path','../escape')]:
            spec=json.loads(json.dumps({'targets':self.targets}));spec['targets']['stable--cpu-wasm32']['raw'][field]=value
            with self.assertRaises(ValueError):pack.build(self.root,spec,self.root/'pack',pairs=[])
            self.assertFalse((self.root/'pack').exists())
    def test_symlink_and_existing_output_rejected(self):
        p=self.root/'stable--cpu-wasm32.wasm';p.unlink();p.symlink_to(self.root/'nightly--cpu-wasm32.wasm')
        with self.assertRaises(ValueError):pack.build(self.root,{'targets':self.targets},self.root/'pack',pairs=[])
        p.unlink();p.write_bytes(self.a);(self.root/'pack').mkdir();(self.root/'pack/sentinel').write_text('keep')
        with self.assertRaises(ValueError):pack.build(self.root,{'targets':self.targets},self.root/'pack',pairs=[])
        self.assertEqual((self.root/'pack/sentinel').read_text(),'keep')
    def test_unavailable_and_unbounded_configs(self):
        for args in [dict(codecs=['other']),dict(codecs=['gzip','gzip']),dict(codecs=['zstd']),dict(pairs=[('not-there','nightly--cpu-wasm32')])]:
            with self.assertRaises(ValueError):pack.build(self.root,{'targets':self.targets},self.root/'pack',**args)
    def test_codec_entry_graph_excludes_zstandard_in_native_plan(self):
        import os
        decoder = os.environ.get('BIC_TEST_ZSTD_DECODER')
        if not decoder:
            self.skipTest('Set BIC_TEST_ZSTD_DECODER to the built decoder')
        pack.build(self.root, {'targets': self.targets}, self.root / 'pack',
                   codecs=['gzip', 'brotli', 'zstd'], pairs=[], zstd_decoder=decoder)
        check = r"""
          import {readFile, copyFile, mkdir} from 'node:fs/promises';
          import {pathToFileURL} from 'node:url';
          import {join} from 'node:path';
          const directory=process.argv[1];
          const {selectWasmAssets}=await import(pathToFileURL(join(directory,'runtime/catalog.mjs')));
          const catalog=JSON.parse(await readFile(join(directory,'catalog.json'),'utf8'));
          for (const codec of ['gzip','brotli','zstd']) {
            const plan=selectWasmAssets(catalog,{targets:Object.keys(catalog.targets),codec});
            const paths=plan.files.map(f=>f.path);
            if (codec!=='zstd' && paths.some(p=>p.includes('zstd')||p.startsWith('licenses/'))) throw Error('unselected codec included');
            if (codec==='zstd' && !plan.runtime.entry.endsWith('loader-zstd.mjs')) throw Error('wrong entry');
            const isolated=join(directory,'selected-'+codec);
            for (const file of plan.files) {
              await mkdir(join(isolated,file.path,'..'),{recursive:true});
              await copyFile(join(directory,file.path),join(isolated,file.path));
            }
            const {createWasmLoader}=await import(pathToFileURL(join(isolated,plan.runtime.entry)));
            const loader=createWasmLoader({plan,readAsset:async f=>new Uint8Array(await readFile(join(isolated,f.path)))});
            if (codec !== 'brotli') await loader.load(Object.keys(plan.targets)[0]);
          }
        """
        subprocess.run(['node','--input-type=module','-e',check,str(self.root/'pack')],check=True)

    def test_zstd_real_decoder_roundtrip(self):
        # The source checkout does not bundle generated Wasm. An explicit test
        # environment path enables the separately rebuilt decoder integration.
        import os
        decoder=os.environ.get('BIC_TEST_ZSTD_DECODER')
        if not decoder:self.skipTest('Set BIC_TEST_ZSTD_DECODER to the built decoder')
        pack.build(self.root,{'targets':self.targets},self.root/'pack',codecs=['zstd'],encoder=self.encoder,zstd_decoder=decoder)

if __name__=='__main__':unittest.main()
