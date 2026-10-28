"""Generated ABI and source-extracted entry tests, not actual model/GPU loading."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_bindings import generate
from prepare_mtmd import PATCH_DIRECTORY, prepare


@unittest.skipUnless(shutil.which('clang') and shutil.which('g++'), 'Clang and GCC are required')
class StorageRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='lcb-storage-runtime-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (self.source / 'tools/mtmd/mtmd.cpp').is_file():
            self.skipTest('Initialize the pinned upstream checkout')

    def test_generated_api_uses_raw_i32_and_keeps_existing_parameter_layout(self):
        generated = self.work / 'generated'
        generate(self.source, generated, 'clang')
        schema = json.loads((generated / 'schema.json').read_text())
        functions = {function['name']: function for function in schema['functions']}
        old = functions['mtmd_init_from_file']
        new = functions['mtmd_init_from_file_with_vision_bf16_storage']
        self.assertEqual([(item['cType'],item['kind']) for item in new['parameters'][:3]],
                         [(item['cType'],item['kind']) for item in old['parameters']])
        self.assertEqual(new['parameters'][3]['cType'], 'int32_t')
        self.assertEqual(new['parameters'][3]['kind'], 'signed')
        self.assertEqual(new['returnKind'], old['returnKind'])
        self.assertEqual(len(old['parameters']), 3)
        self.assertIn('mtmd_vision_bf16_storage_version', functions)
        self.assertIn('mtmd_context_vision_bf16_storage', functions)
        record = next(record for record in schema['records'] if record['name'] == 'mtmd_context_params')
        self.assertNotIn('vision_bf16_storage', [field['name'] for field in record['fields']])
        exports = json.loads((generated / 'exports.json').read_text())
        jspi = json.loads((generated / 'jspi-exports.json').read_text())
        for name in ['mtmd_init_from_file', 'mtmd_init_from_file_with_vision_bf16_storage',
                     'mtmd_vision_bf16_storage_version', 'mtmd_context_vision_bf16_storage']:
            self.assertIn('_lcb_' + name, exports)
            self.assertIn('lcb_' + name, jspi)
        code = (generated / 'bindings.cpp').read_text()
        self.assertIn('int32_t a3', code)
        self.assertIn('(int32_t)a3', code)

    def test_actual_entry_bodies_reject_unknown_i32_before_constructor_and_allow_reentry(self):
        # The constructor below is an explicit no-I/O double. The production
        # init/query/version bodies are extracted verbatim, including their
        # validation and exception boundary; this does not exercise the loader.
        patched = prepare(self.source, self.work / 'overlay',
                          ROOT / PATCH_DIRECTORY / 'mtmd-vision-bf16-runtime.patch', filename='mtmd.cpp')
        code = patched.read_text()
        start = code.index('mtmd_context * mtmd_init_from_file(')
        end = code.index('void mtmd_free(mtmd_context * ctx)', start)
        entries = code[start:end]
        fixture = self.work / 'entry.cpp'
        fixture.write_text(r'''
#include "mtmd-vision-bf16.h"
#include <cassert>
#include <climits>
#include <stdexcept>
#include <string>
#define LOG_ERR(...) ((void)0)
static int constructions = 0;
struct mtmd_context {
    const mtmd_vision_bf16_storage vision_bf16_storage;
    mtmd_context(const char * path, const llama_model *, const mtmd_context_params & params,
                 bool = false, mtmd_vision_bf16_storage mode = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND) :
        vision_bf16_storage(mode) {
        ++constructions;
        if (std::string(path) == "throw") throw std::runtime_error("load rejected");
        if (std::string(path) == "outer") params.progress_callback(0.5f, params.progress_callback_user_data);
    }
};
''' + entries + r'''
int main() {
    mtmd_context_params params{};
    assert(mtmd_vision_bf16_storage_version() == 1);
    assert(mtmd_context_vision_bf16_storage(nullptr) == MTMD_VISION_BF16_STORAGE_INVALID);
    for (int32_t invalid : {INT_MIN, -2, -1, 2, INT_MAX}) {
        assert(!mtmd_init_from_file_with_vision_bf16_storage(nullptr, nullptr, params, invalid));
    }
    assert(constructions == 0);
    mtmd_context * nested = nullptr;
    params.progress_callback_user_data = &nested;
    params.progress_callback = [](float, void * data) {
        mtmd_context_params inner{};
        *static_cast<mtmd_context **>(data) = mtmd_init_from_file_with_vision_bf16_storage(
            "inner", nullptr, inner, MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
        return true;
    };
    auto * retained = mtmd_init_from_file_with_vision_bf16_storage(
        "outer", nullptr, params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(retained && nested);
    assert(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(mtmd_context_vision_bf16_storage(nested) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    auto * old = mtmd_init_from_file("old", nullptr, params);
    assert(old && mtmd_context_vision_bf16_storage(old) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    assert(!mtmd_init_from_file_with_vision_bf16_storage("throw", nullptr, params, 1));
    auto * again = mtmd_init_from_file_with_vision_bf16_storage("again", nullptr, params, 0);
    assert(again && mtmd_context_vision_bf16_storage(again) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    assert(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    delete again; delete old; delete nested; delete retained;
}
''')
        binary = self.work / 'entry'
        subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'bridge'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_cpu_loader_boundary_keeps_audio_and_rejects_only_gpu_requested_vision(self):
        # Execute the actual minimal clip init body with a metadata-only loader
        # double. This fixes modality/cleanup control flow, not GGUF execution.
        patched = prepare(self.source, self.work / 'cpu-clip',
                          ROOT / PATCH_DIRECTORY / 'mtmd-vision-bf16-clip-runtime.patch')
        code = patched.read_text()
        start = code.index('struct clip_init_result clip_init(')
        end = code.index('struct clip_cap clip_get_cap(', start)
        entries = code[start:end]
        fixture = self.work / 'modality.cpp'
        fixture.write_text(r'''
#include "mtmd-vision-bf16-internal.h"
#include <cassert>
#include <string>
#include <vector>
#define LOG_ERR(...) ((void)0)
static int allocated = 0;
static int weight_loads = 0;
static int classifications = 0;
enum projector_type { PROJECTOR_TYPE_MLP, PROJECTOR_TYPE_GEMMA3NV,
                      PROJECTOR_TYPE_LFM3A_GEN, PROJECTOR_TYPE_SOPRANO };
struct fixture_model { projector_type proj_type = PROJECTOR_TYPE_MLP; };
struct clip_ctx {
    const mtmd_vision_bf16_storage vision_bf16_storage;
    fixture_model model;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes = 1;
    clip_ctx(clip_context_params &, mtmd_vision_bf16_storage mode = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND) :
        vision_bf16_storage(mode) { ++allocated; }
    ~clip_ctx() { --allocated; }
};
projector_type clip_get_projector_type(const clip_ctx * ctx) { return ctx->model.proj_type; }
extern "C" size_t ggml_tensor_overhead(void) { return 1; }
extern "C" size_t ggml_graph_overhead(void) { return 1; }
struct clip_model_loader {
    bool has_vision;
    bool has_audio;
    bool has_gen_audio = false;
    clip_model_loader(const char * path, bool, mtmd_progress_callback, void *) :
        has_vision(std::string(path) == "vision"), has_audio(std::string(path) == "audio") {
        ++classifications;
    }
    void load_hparams(fixture_model &, clip_modality) {}
    void load_tensors(clip_ctx &) { ++weight_loads; }
    void init_ctx(clip_ctx &) {}
    void warmup(clip_ctx &) {}
};
''' + entries + r'''
int main() {
    clip_context_params params{};
    params.use_gpu = true;
    auto audio = lcb_clip_init_with_vision_bf16_storage("audio", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(audio.ctx_a && !audio.ctx_v && classifications == 1 && weight_loads == 1);
    assert(audio.ctx_a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    delete audio.ctx_a;
    auto rejected = lcb_clip_init_with_vision_bf16_storage("vision", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(!rejected.ctx_a && !rejected.ctx_v && allocated == 0);
    assert(classifications == 2 && weight_loads == 1);
    auto legacy = clip_init("vision", params);
    assert(legacy.ctx_v && legacy.ctx_v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    delete legacy.ctx_v;
    params.use_gpu = false;
    auto cpu = lcb_clip_init_with_vision_bf16_storage("vision", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(cpu.ctx_v && cpu.ctx_v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    delete cpu.ctx_v;
    assert(allocated == 0 && weight_loads == 3 && classifications == 4);
}
''')
        binary = self.work / 'modality'
        subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'bridge'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
