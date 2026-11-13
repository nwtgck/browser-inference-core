"""Compile packaged host and generated module declarations as a strict consumer.

The native declarations used here are fixtures, not evidence of a native build.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from build_tools.packaging.layout import ARTIFACT_DIRS, PACKAGE_NAME, package_exports
from build_tools.packaging.types import write_host_profiles
from build_tools.project import runtime_root
from build_tools.runtimes.stable_diffusion_cpp.bindings import write_module_types


class PublicTypes(unittest.TestCase):
    def setUp(self):
        compiler = os.environ.get('BICORE_TYPESCRIPT_COMPILER') or shutil.which('tsc')
        if not compiler:
            self.skipTest('Set BICORE_TYPESCRIPT_COMPILER to the prepared TypeScript compiler')
        if not Path(compiler).is_file():
            self.fail('Configured TypeScript compiler is unavailable: ' + compiler)
        self.command = [shutil.which('node') or 'node', str(Path(compiler).resolve()),
                        '--noEmit', '--strict', '--skipLibCheck', 'false',
                        '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--target', 'ES2022']
        temporary = tempfile.TemporaryDirectory(prefix='bicore-public-types-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'package.json').write_text(json.dumps({
            'name': PACKAGE_NAME, 'type': 'module', 'exports': package_exports(),
        }))
        for runtime in ('llama-cpp', 'stable-diffusion-cpp'):
            base = self.root / ARTIFACT_DIRS[runtime]
            (base / 'host').mkdir(parents=True)
            (base / 'api').mkdir()
            shutil.copy2(runtime_root(runtime) / 'host/index.d.ts', base / 'host/index.d.ts')
            name = 'llama_decode' if runtime == 'llama-cpp' else 'generate_image'
            (base / 'api/functions.d.ts').write_text(
                'export interface LowLevelFunctions { ' + name + '(context: bigint): Promise<number>; }\n')
        self.llama_profiles = ('cpu-wasm32', 'webgpu-wasm64-jspi', 'webgpu-wasm64-jspi-gpu-profile')
        for profile in self.llama_profiles:
            for variant in ('browser', 'test'):
                directory = self.root / 'runtimes/llama-cpp-browser-core/profiles' / profile / variant
                directory.mkdir(parents=True)
                result = 'number' if profile == 'cpu-wasm32' else 'bigint'
                diagnostic = '  diagnostic_only(): bigint;\n' if profile.endswith('gpu-profile') else ''
                probe = '  test_only(): number;\n' if variant == 'test' else ''
                (directory / 'core.d.ts').write_text(
                    "import type { CoreModule } from '../../../host/index.js';\n"
                    'export interface MainModule extends CoreModule {\n'
                    '  common_chat_no_tool_call_index(): ' + result + ';\n' + diagnostic + probe + '}\n'
                    'export default function create(): Promise<MainModule>;\n')
        write_host_profiles(self.root / 'runtimes/llama-cpp-browser-core/host/profiles.d.ts',
                            'llama-cpp', self.llama_profiles, ('browser', 'test'))
        self.schema = {'functions': [{
            'name': 'generate_image', 'export': '_sdc_generate_image', 'returnKind': 'signed',
            'parameters': [{'kind': 'pointer'}],
        }, {
            'name': 'record_result', 'export': '_sdc_record_result', 'returnKind': 'record',
            'parameters': [{'kind': 'unsigned'}, {'kind': 'boolean'}],
        }]}
        for profile in ('webgpu-wasm32-asyncify', 'webgpu-wasm64-jspi'):
            for variant in ('browser', 'test'):
                directory = self.root / 'runtimes/stable-diffusion-cpp-browser-core/profiles' / profile / variant
                directory.mkdir(parents=True)
                write_module_types(directory, self.schema, 'asyncify' if 'asyncify' in profile else 'jspi')
        write_host_profiles(self.root / 'runtimes/stable-diffusion-cpp-browser-core/host/profiles.d.ts',
                            'stable-diffusion-cpp', ('webgpu-wasm32-asyncify', 'webgpu-wasm64-jspi'),
                            ('browser', 'test'))

    def compile(self, source):
        (self.root / 'consumer.ts').write_text(source)
        return subprocess.run([*self.command, 'consumer.ts'], cwd=self.root,
                              text=True, capture_output=True, timeout=60)

    def test_model_graph_extension_is_public_for_real_type_and_node_consumers(self):
        from build_tools.runtimes.llama_cpp.generate_model_graph import generate
        clang = shutil.which('clang')
        if not clang:
            self.skipTest('clang required to generate the actual model graph declarations')
        relative = ARTIFACT_DIRS['llama-cpp'] + '/profiles/cpu-wasm32/browser'
        directory = self.root / relative
        exports = self.root / 'base-exports.json'; exports.write_text('[]')
        schema = generate(runtime_root('llama-cpp') / 'native/model-graph/model-graph.h',
                          directory, exports, clang)
        result = self.compile("""
import type { ModelGraphNativeExports } from 'browser-inference-core/llama-cpp/profiles/cpu-wasm32/browser/model-graph-v1.functions';
declare const native: ModelGraphNativeExports;
const vocabulary: bigint = native._lcb_model_graph_vocab_new(0n);
const result: number = native._lcb_model_graph_sample(0n, 0n, 0n, 1);
// @ts-expect-error normalized pointer parameters cannot accept a number.
native._lcb_model_graph_vocab_new(0);
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        specifier = 'browser-inference-core/llama-cpp/profiles/cpu-wasm32/browser/model-graph-v1.schema.json'
        script = ("import {readFileSync} from 'node:fs'; "
                  "const schema=JSON.parse(readFileSync(new URL(import.meta.resolve("
                  + json.dumps(specifier) + ")), 'utf8')); "
                  "console.log(schema.schemaSha256);")
        result = subprocess.run([shutil.which('node') or 'node', '--input-type=module', '-e', script],
                                cwd=self.root, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), schema['schemaSha256'])
        package = json.loads((self.root / 'package.json').read_text())
        del package['exports']['./llama-cpp/profiles/*/model-graph-v1.functions']
        (self.root / 'package.json').write_text(json.dumps(package))
        self.assertNotEqual(self.compile("import type { ModelGraphNativeExports } from 'browser-inference-core/llama-cpp/profiles/cpu-wasm32/browser/model-graph-v1.functions';").returncode, 0)

    def test_profile_variant_and_normalized_native_types_reach_consumer(self):
        result = self.compile('''
import { createCore } from 'browser-inference-core/llama-cpp/host';
import type { LowLevelFunctions } from 'browser-inference-core/llama-cpp/api/functions';
import factory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm64-jspi/browser/core.mjs';
import asyncifyFactory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-asyncify/browser/core.mjs';
import { attachCore, schema } from 'browser-inference-core/stable-diffusion-cpp/host';
const cpu = await createCore({profile: 'cpu-wasm32'});
const small: number = cpu.module.common_chat_no_tool_call_index();
const gpu = await createCore({profile: 'webgpu-wasm64-jspi'});
const wide: bigint = gpu.module.common_chat_no_tool_call_index();
// @ts-expect-error the selected wasm64 declaration must remain visible.
const wrongWidth: number = gpu.module.common_chat_no_tool_call_index();
// @ts-expect-error an unknown native method must not be any.
gpu.module.removed_native_method();
// @ts-expect-error diagnostics belong to the diagnostic profile.
gpu.module.diagnostic_only();
(await createCore({profile: 'webgpu-wasm64-jspi-gpu-profile'})).module.diagnostic_only();
// @ts-expect-error test-only exports must not appear on the browser variant.
cpu.module.test_only();
(await createCore({profile: 'cpu-wasm32', variant: 'test'})).module.test_only();
const image = await factory();
const status: Promise<number> = image._sdc_generate_image(0n);
const record: Promise<void> = image._sdc_record_result(0n, 1, 1);
const helper: number = image._sdc_pointer_bytes();
// @ts-expect-error normalized pointers use bigint even for wasm32.
image._sdc_generate_image(0);
// @ts-expect-error the normalized boolean ABI is numeric.
image._sdc_record_result(0n, 1, true);
// @ts-expect-error JSPI completion is asynchronous.
const synchronous: number = image._sdc_generate_image(0n);
const host = attachCore(image, schema, {suspension: 'direct'});
const exactModule: typeof image = host.module;
const completion: Promise<number> = host.api.generate_image(0n);
const asyncify = await asyncifyFactory();
// @ts-expect-error raw Asyncify returns do not represent completion.
asyncify._sdc_generate_image(0n);
attachCore(asyncify, schema, {suspension: 'asyncify'}).api.generate_image(0n);
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_changed_generated_signature_invalidates_stale_consumer(self):
        consumer = '''
import factory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm64-jspi/browser/core.mjs';
(await factory())._sdc_generate_image(0n);
'''
        self.assertEqual(self.compile(consumer).returncode, 0)
        self.schema['functions'][0]['parameters'][0]['kind'] = 'unsigned'
        write_module_types(self.root / 'runtimes/stable-diffusion-cpp-browser-core/profiles/webgpu-wasm64-jspi/browser', self.schema, 'jspi')
        result = self.compile(consumer)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('bigint', result.stdout)
        self.assertIn('number', result.stdout)

    def test_single_profile_package_imports_only_existing_declarations(self):
        base = self.root / 'runtimes/llama-cpp-browser-core'
        for profile in self.llama_profiles:
            if profile != 'cpu-wasm32':
                shutil.rmtree(base / 'profiles' / profile)
        shutil.rmtree(base / 'profiles/cpu-wasm32/test')
        write_host_profiles(base / 'host/profiles.d.ts', 'llama-cpp', ['cpu-wasm32'], ['browser'])
        image = self.root / 'runtimes/stable-diffusion-cpp-browser-core'
        shutil.rmtree(image / 'profiles/webgpu-wasm32-asyncify')
        shutil.rmtree(image / 'profiles/webgpu-wasm64-jspi/test')
        write_host_profiles(image / 'host/profiles.d.ts', 'stable-diffusion-cpp', ['webgpu-wasm64-jspi'], ['browser'])
        result = self.compile('''
import { createCore } from 'browser-inference-core/llama-cpp/host';
import { attachCore, schema } from 'browser-inference-core/stable-diffusion-cpp/host';
import image from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm64-jspi/browser/core.mjs';
(await createCore({profile: 'cpu-wasm32'})).module.common_chat_no_tool_call_index();
// @ts-expect-error unshipped profiles cannot be selected.
createCore({profile: 'webgpu-wasm64-jspi'});
// @ts-expect-error unshipped test variant cannot be selected.
createCore({profile: 'cpu-wasm32', variant: 'test'});
attachCore(await image(), schema, {suspension: 'direct'}).api.generate_image(0n);
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_callback_tuples_custom_instantiation_and_explicit_attachment_modes(self):
        result = self.compile('''
import { createCore, attachCore as attachLlama, schema as llamaSchema } from 'browser-inference-core/llama-cpp/host';
import factory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm64-jspi/browser/core.mjs';
import asyncifyFactory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-asyncify/browser/core.mjs';
import { attachCore, schema } from 'browser-inference-core/stable-diffusion-cpp/host';
const llama = await createCore({profile: 'webgpu-wasm64-jspi'});
const image = await factory();
for (const module of [llama.module, image]) {
  module.addFunction((level: number, text: bigint, data: bigint): void => {}, 'vijj');
  module.addFunction((pointer: bigint): bigint => pointer, 'jj');
  module.addFunction((first: number, second: number): number => first + second, 'iii');
  // @ts-expect-error native callbacks do not receive strings.
  module.addFunction((text: string): void => {}, 'vi');
  // @ts-expect-error native callbacks do not return strings.
  module.addFunction(() => 'wrong', 'i');
}
attachLlama(llama.module, llamaSchema, {suspension: 'direct'});
// @ts-expect-error attachment requires deliberate suspension selection.
attachLlama(llama.module, llamaSchema);
// @ts-expect-error attachment requires deliberate suspension selection.
attachCore(image, schema);
const asyncify = await asyncifyFactory();
// @ts-expect-error Asyncify attachment also requires explicit selection.
attachCore(asyncify, schema);
await factory({instantiateWasm(imports, onSuccess) {
  const compiled = new WebAssembly.Module(new Uint8Array());
  onSuccess(new WebAssembly.Instance(compiled, imports), compiled);
}});
await factory({async instantiateWasm(imports, onSuccess) {
  const result = await WebAssembly.instantiate(new Uint8Array(), imports);
  onSuccess(result.instance, result.module);
}});
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
