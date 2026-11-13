"""Generated ABI and source-extracted entry tests, not actual model/GPU loading."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp.bindings import generate
from build_tools.runtimes.llama_cpp.sources.mtmd import PATCH_DIRECTORY, prepare


@unittest.skipUnless(shutil.which('clang') and shutil.which('g++'), 'Clang and GCC are required')
class StorageRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='lcb-storage-runtime-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (self.source / 'tools/mtmd/mtmd.cpp').is_file():
            self.skipTest('Initialize the pinned upstream checkout')

    def test_actual_formatter_excludes_only_the_added_terminator(self):
        code = (self.source / 'tools/mtmd/mtmd.cpp').read_text()
        start = code.index('std::string mtmd_format(')
        end = code.index('\n}\n', start) + 3
        fixture = self.work / 'formatter.cpp'
        fixture.write_text(r'''
#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>
#define GGML_ASSERT(value) assert(value)
''' + code[start:end] + r'''
int main() {
    assert(mtmd_format("").empty());
    assert(mtmd_format("plain") == "plain");
    const auto prompt = mtmd_format("image %ux%u<|media_content|>", 480u, 336u);
    assert(prompt == "image 480x336<|media_content|>");
    assert(prompt.size() == std::string("image 480x336<|media_content|>").size());
    const auto embedded = mtmd_format("%c", 0);
    assert(embedded.size() == 1 && embedded[0] == '\0');
}
''')
        binary = self.work / 'formatter'
        subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

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
        tree = prepare(self.source, self.work / 'overlay', mode='cpu')
        code = (tree / 'mtmd.cpp').read_text()
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
                        '-I' + str(ROOT / 'native/vision'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_cpu_loader_boundary_keeps_audio_and_rejects_only_gpu_requested_vision(self):
        # Execute the actual model-load orchestration with metadata-only doubles.
        # It proves classification/policy/cleanup, not GGUF or model execution.
        tree = prepare(self.source, self.work / 'cpu-model', mode='cpu')
        code = (tree / 'mtmd-model.cpp').read_text()
        start = code.index('mtmd_models mtmd_model_load(')
        end = code.index('mtmd_model_caps mtmd_model_get_caps(', start)
        entries = code[start:end]
        fixture = self.work / 'modality.cpp'
        fixture.write_text(r'''
#include "mtmd-vision-bf16-policy.h"
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
static int classifications = 0, weight_loads = 0, live_weights = 0;
enum mtmd_modality { MTMD_MODALITY_VISION, MTMD_MODALITY_AUDIO, MTMD_MODALITY_GEN_AUDIO };
enum projector_type { PROJECTOR_TYPE_MLP, PROJECTOR_TYPE_GEMMA3NV, PROJECTOR_TYPE_CHATTERBOX };
struct mtmd_model {
    const mtmd_vision_bf16_storage vision_bf16_storage = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND;
    mtmd_modality modality = MTMD_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_MLP;
    struct { int custom_image_min_tokens = -1, custom_image_max_tokens = -1; } hparams;
    void * zero_pointer;
    bool weights = false;
    ~mtmd_model() { if (weights) --live_weights; }
};
struct mtmd_model_params {
    mtmd_vision_bf16_storage vision_bf16_storage = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND;
    bool gpu_requested = false;
    int image_min_tokens = -1, image_max_tokens = -1;
    bool no_alloc = false;
    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;
};
struct mtmd_models { std::unique_ptr<mtmd_model> v, a, gen_a; };
struct mtmd_model_loader {
    std::string path;
    bool has_vision, has_audio, has_gen_audio;
    mtmd_model_loader(const char * p, bool, mtmd_progress_callback, void *) : path(p),
        has_vision(path == "vision" || path == "mixed" || path == "gemma3n"),
        has_audio(path == "audio" || path == "mixed" || path == "gemma3n"),
        has_gen_audio(path == "gen" || path == "mixed") { ++classifications; }
    void load_hparams(mtmd_model & model, mtmd_modality modality) {
        model.modality = modality;
        if (path == "gemma3n" && modality == MTMD_MODALITY_VISION) model.proj_type = PROJECTOR_TYPE_GEMMA3NV;
        if (modality == MTMD_MODALITY_GEN_AUDIO) model.proj_type = PROJECTOR_TYPE_CHATTERBOX;
    }
    void load_tensors(mtmd_model & model, ggml_backend_buffer_type_t, bool no_alloc) {
        assert(model.zero_pointer == nullptr);
        assert(model.hparams.custom_image_min_tokens == 3 && model.hparams.custom_image_max_tokens == 7);
        if (!no_alloc) { ++weight_loads; ++live_weights; model.weights = true; }
    }
};
''' + entries + r'''
int main() {
    mtmd_model_params params;
    params.image_min_tokens = 3; params.image_max_tokens = 7;
    params.gpu_requested = true;
    params.vision_bf16_storage = MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED;
    {
        auto audio = mtmd_model_load("audio", params, nullptr);
        assert(audio.a && !audio.v && classifications == 1 && weight_loads == 1);
        assert(audio.a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    }
    assert(live_weights == 0);
    const int before = weight_loads;
    try { mtmd_model_load("vision", params, nullptr); assert(false); }
    catch (const std::runtime_error &) {}
    assert(classifications == 2 && weight_loads == before && live_weights == 0);
    params.no_alloc = true;
    try { mtmd_model_load("vision", params, nullptr); assert(false); }
    catch (const std::runtime_error &) {}
    assert(weight_loads == before);
    params.gpu_requested = false;
    {
        auto cpu = mtmd_model_load("vision", params, nullptr);
        assert(cpu.v && cpu.v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
        assert(weight_loads == before && live_weights == 0);
    }
    params.no_alloc = false;
    {
        auto mixed = mtmd_model_load("mixed", params, nullptr);
        assert(mixed.v && mixed.a && mixed.gen_a);
        assert(mixed.v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
        assert(mixed.a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
        assert(mixed.gen_a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
        assert(mixed.gen_a->proj_type == PROJECTOR_TYPE_CHATTERBOX && live_weights == 3);
    }
    assert(live_weights == 0);
    {
        auto gemma = mtmd_model_load("gemma3n", params, nullptr);
        assert(gemma.v && !gemma.a && !gemma.gen_a && live_weights == 1);
    }
    params.gpu_requested = true;
    params.vision_bf16_storage = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND;
    {
        auto legacy = mtmd_model_load("vision", params, nullptr);
        assert(legacy.v && legacy.v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    }
    assert(live_weights == 0);
}
''')
        binary = self.work / 'modality'
        subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'native/vision'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_constructor_exceptions_release_scheduler_before_distinct_backends(self):
        template = (ROOT / 'tests/mtmd-constructor/main.cpp.in').read_text()
        self.assertEqual(template.count('@ACTUAL_BODIES@'), 1)
        for mode in ('cpu', 'webgpu'):
            with self.subTest(mode=mode):
                tree = prepare(self.source, self.work / ('constructor-' + mode), mode=mode)
                source = (tree / 'mtmd-subcontext.cpp').read_text()
                start = source.index('mtmd_backends::mtmd_backends(')
                end = source.index('std::map<ggml_backend_dev_t, size_t> mtmd_backends::get_mem_compute()', start)
                borrowed = source.index('mtmd_subcontext::mtmd_subcontext(', end)
                borrowed_end = source.index('mtmd_image_f32_batch mtmd_subcontext::get_dummy_batch()', borrowed)
                code = template.replace('@ACTUAL_BODIES@', source[start:end] + source[borrowed:borrowed_end])
                fixture = self.work / (mode + '-constructor.cpp')
                fixture.write_text(code)
                binary = self.work / (mode + '-constructor')
                subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                                str(fixture), '-o', str(binary)], check=True, capture_output=True, text=True)
                subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_upload_backend_public_api_failures_and_drain_lifetime(self):
        binary = self.work / 'upload-backend'
        subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'native/vision'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(ROOT / 'tests/mtmd-upload-backend/main.cpp'), '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        subprocess.run([str(binary)], check=True, capture_output=True, text=True)
