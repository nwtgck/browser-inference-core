"""Actual reflect-pad factory and shader arithmetic, without GPU execution."""
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
from test_webgpu_buffer_limit import definition


class PadReflect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='pad-reflect-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.prepared = {profile: source.prepare(cls.upstream, cls.directory / str(profile), gpu_profile=profile).parent
                        for profile in (False, True)}
        cls.shader = (cls.upstream / 'ggml/src/ggml-webgpu/wgsl-shaders/pad.wgsl').read_text()
        cls.raw_library = (cls.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp').read_text()

    def test_upstream_reflect_shader_factory_and_admission_are_preserved(self):
        original = (self.upstream / source.SOURCE_PATH).read_text()
        for profile, directory in self.prepared.items():
            with self.subTest(profile=profile):
                library = (directory / 'ggml-webgpu-shader-lib.hpp').read_text()
                self.assertEqual(definition(library, 'webgpu_pipeline get_pad_pipeline('),
                                 definition(self.raw_library, 'webgpu_pipeline get_pad_pipeline('))
                self.assertIn(self.shader, (directory / source.aligned_word_load.HEADER_NAME).read_text())
                cpp = (directory / 'ggml-webgpu.cpp').read_text()
                self.assertEqual(definition(cpp, 'static webgpu_encoded_op ggml_webgpu_pad('),
                                 definition(original, 'static webgpu_encoded_op ggml_webgpu_pad('))
                self.assertIn('case GGML_OP_PAD_REFLECT_1D:\n        case GGML_OP_PAD:', cpp)
                support = definition(cpp, 'static bool ggml_backend_webgpu_device_supports_op(')
                reflect = support.split('case GGML_OP_PAD_REFLECT_1D:', 1)[1].split('break;', 1)[0]
                normal = support.split('case GGML_OP_PAD:', 1)[1].split('break;', 1)[0]
                self.assertEqual(reflect, normal)

    def test_actual_reflect_indices_strides_and_padded_grid_match_scalar_oracle(self):
        reflect = self.shader.split('#ifdef REFLECT', 1)[1].split('#else\n#ifdef CIRCULAR', 1)[0]
        conditions = re.findall(r'(?:if|else if) \(([^)]+)\)', reflect)
        expressions = re.findall(r'\bx = ([^;]+);', reflect)
        address = re.search(r'value = src\[([^]]+)\];', reflect).group(1)
        self.assertEqual(len(conditions), 2)
        self.assertEqual(len(expressions), 3)
        self.assertIn('if (gid_i >= params.ne)', self.shader)
        fields = re.findall(r'\b(\w+): u32', self.shader.split('struct Params {', 1)[1].split('};', 1)[0])
        self.assertEqual(len(fields), 23)  # Existing uniform layout is unchanged.
        for size in (1, 2, 5, 9):
            for left, right in itertools.product(range(size), repeat=2):
                params = SimpleNamespace(lp0=left, src_ne0=size, offset_src=1,
                                         stride_src0=2, stride_src1=2 * size + 3,
                                         stride_src2=2 * (2 * size + 3) + 7,
                                         stride_src3=2 * (2 * (2 * size + 3) + 7) + 11)
                width = size + left + right
                count = width * 8
                seen = []
                # A padded 2D invocation grid, with eight global x lanes per row.
                for y in range((count + 7) // 8):
                    for x in range(8):
                        index = x + 8 * y
                        if index >= count:
                            continue
                        i0 = index % width
                        row = index // width
                        context = {'i0': i0, 'i1': row % 2, 'i2': row // 2 % 2,
                                   'i3': row // 4, 'params': params}
                        branch = (0 if eval(conditions[0], {'__builtins__': {}}, context)
                                  else 1 if eval(conditions[1], {'__builtins__': {}}, context) else 2)
                        context['x'] = eval(expressions[branch], {'__builtins__': {}}, context)
                        actual = eval('(' + address + ')', {'__builtins__': {}}, context)
                        logical = i0 - left
                        expected_x = -logical if logical < 0 else 2 * size - 2 - logical if logical >= size else logical
                        self.assertTrue(0 <= context['x'] < size)
                        self.assertEqual(actual, 1 + 2 * expected_x + context['i1'] * params.stride_src1 +
                                         context['i2'] * params.stride_src2 + context['i3'] * params.stride_src3)
                        seen.append(index)
                self.assertEqual(seen, list(range(count)))

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_factory_keeps_three_cache_keys_and_warm_hits_compile_nothing(self):
        header = (self.upstream / 'ggml/include/ggml.h').read_text()
        enums = definition(header, 'enum ggml_op {') + ';\n' + definition(header, 'enum ggml_type {') + ';\n'
        for profile, directory in self.prepared.items():
            with self.subTest(profile=profile):
                library = (directory / 'ggml-webgpu-shader-lib.hpp').read_text()
                structs = '\n'.join(definition(library, name) + ';' for name in
                                    ('struct ggml_webgpu_pad_pipeline_key {', 'struct ggml_webgpu_pad_pipeline_key_hash {'))
                combine = definition(library, 'template <typename T> inline void ggml_webgpu_hash_combine(')
                support = definition((directory / 'ggml-webgpu.cpp').read_text(),
                                     'static bool ggml_backend_webgpu_device_supports_op(')
                admission = support.split('case GGML_OP_PAD_REFLECT_1D:', 1)[1].split('break;', 1)[0]
                code = '''#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
''' + enums + combine + structs + '''
struct ggml_tensor {int type=GGML_TYPE_F32,op=GGML_OP_PAD;int params[9]{};};
int ggml_get_op_params_i32(const ggml_tensor* tensor,int index){assert(index==8);return tensor->params[index];}
struct ggml_webgpu_shader_lib_context {ggml_tensor* dst;uint32_t max_wg_size;};
struct ggml_webgpu_generic_shader_decisions {uint32_t wg_size=0;};
struct webgpu_pipeline {int id;std::shared_ptr<ggml_webgpu_generic_shader_decisions> context;};
struct Record {std::string variant;std::vector<std::string> defines;};
static std::vector<Record> records;
static std::vector<std::string> selected_defines;
const std::string wgsl_pad = ''' + json.dumps(self.shader) + ''';
struct Preprocessor {
int calls=0;
std::string preprocess(const std::string& shader,const std::vector<std::string>& defines){
assert(shader==wgsl_pad);++calls;selected_defines=defines;return shader;}
};
webgpu_pipeline ggml_webgpu_create_pipeline(int,const std::string& shader,const std::string& variant){
assert(shader==wgsl_pad);records.push_back({variant,selected_defines});return {int(records.size()),nullptr};}
struct Library {int device=0;Preprocessor preprocessor;
std::unordered_map<ggml_webgpu_pad_pipeline_key,webgpu_pipeline,ggml_webgpu_pad_pipeline_key_hash> pad_pipelines;
''' + definition(library, 'webgpu_pipeline get_pad_pipeline(') + '''
};
bool supports_reflect(int dtype,int source_type){
ggml_tensor dst,src;dst.type=dtype;src.type=source_type;const auto* op=&dst;const auto* src0=&src;bool supports_op=false;
''' + admission + '''
return supports_op;
}
int main(){
for(int dtype:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
for(int source_type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
assert(supports_reflect(dtype,source_type)==(dtype==GGML_TYPE_F32 && source_type==GGML_TYPE_F32));
std::array<int,3> order{0,1,2};
do {
Library library;records.clear();int ids[3]{};
for(int mode:order){
ggml_tensor dst;dst.op=mode==2?GGML_OP_PAD_REFLECT_1D:GGML_OP_PAD;dst.params[8]=mode!=0;
ggml_webgpu_shader_lib_context context{&dst,128};auto pipeline=library.get_pad_pipeline(context);ids[mode]=pipeline.id;
assert(pipeline.context->wg_size==128);
assert(records.back().variant==(mode==2?"pad_reflect":mode==1?"pad_circular":"pad"));
std::vector<std::string> expected;
if(mode==2)expected.push_back("REFLECT");else if(mode==1)expected.push_back("CIRCULAR");
expected.push_back("WG_SIZE=128");assert(records.back().defines==expected);
}
assert(ids[0]!=ids[1] && ids[1]!=ids[2] && ids[0]!=ids[2]);
assert(library.pad_pipelines.size()==3 && library.preprocessor.calls==3);
for(int mode:{2,0,1,2}){
ggml_tensor dst;dst.op=mode==2?GGML_OP_PAD_REFLECT_1D:GGML_OP_PAD;dst.params[8]=mode==1;
ggml_webgpu_shader_lib_context context{&dst,128};assert(library.get_pad_pipeline(context).id==ids[mode]);
}
assert(records.size()==3 && library.preprocessor.calls==3);
}while(std::next_permutation(order.begin(),order.end()));
}
'''
                cpp = self.directory / f'factory-{profile}.cpp'
                cpp.write_text(code)
                binary = self.directory / f'factory-{profile}'
                subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
