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
        reported = functions['mtmd_init_from_file_with_vision_bf16_report']
        self.assertEqual(reported['parameters'][:4], new['parameters'])
        self.assertEqual([parameter['kind'] for parameter in reported['parameters'][4:]], ['pointer', 'unsigned'])
        statistics = next(record for record in schema['records'] if record['name'] == 'mtmd_vision_bf16_statistics')
        self.assertEqual([field['kind'] for field in statistics['fields']], ['unsigned', 'unsigned', 'signed'] + ['u64'] * 5)
        report = next(record for record in schema['records'] if record['name'] == 'mtmd_vision_bf16_load_report')
        self.assertEqual([field['kind'] for field in report['fields']], ['unsigned', 'unsigned'] + ['signed'] * 4)
        record = next(record for record in schema['records'] if record['name'] == 'mtmd_context_params')
        self.assertNotIn('vision_bf16_storage', [field['name'] for field in record['fields']])
        exports = json.loads((generated / 'exports.json').read_text())
        jspi = json.loads((generated / 'jspi-exports.json').read_text())
        for name in ['mtmd_init_from_file', 'mtmd_init_from_file_with_vision_bf16_storage',
                     'mtmd_vision_bf16_storage_version', 'mtmd_context_vision_bf16_storage',
                     'mtmd_vision_bf16_diagnostics_version', 'mtmd_vision_bf16_classify_metadata',
                     'mtmd_init_from_file_with_vision_bf16_report', 'mtmd_context_vision_bf16_statistics']:
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
#include "mtmd-vision-bf16-internal.h"
#include "mtmd-vision-bf16-classification.h"
#include <cassert>
#include <climits>
#include <stdexcept>
#include <string>
#undef LOG_ERR
#define LOG_ERR(...) ((void)0)
static int constructions = 0;
struct mtmd_context {
    const mtmd_vision_bf16_storage vision_bf16_storage;
    clip_ctx * ctx_v = nullptr;
    mtmd_context(const char * path, const llama_model *, const mtmd_context_params & params,
                 bool = false, mtmd_vision_bf16_storage mode = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND,
                 mtmd_vision_bf16_load_report * report = nullptr) :
        vision_bf16_storage(mode) {
        ++constructions;
        if (std::string(path) == "refused") {
            assert(report);
            *report = {1, sizeof(*report), MTMD_VISION_BF16_LOAD_RETENTION_REFUSED,
                MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED, MTMD_VISION_BF16_PHASE_PREUPLOAD,
                MTMD_VISION_BF16_CLEANUP_COMPLETE};
            throw std::runtime_error("already cleaned refusal");
        }
        if (std::string(path) == "throw") throw std::runtime_error("load rejected");
        if (std::string(path) == "outer") params.progress_callback(0.5f, params.progress_callback_user_data);
    }
};
mtmd_vision_bf16_statistics lcb_clip_vision_bf16_statistics(const clip_ctx *) {
    return lcb_mtmd_empty_statistics(MTMD_VISION_BF16_EFFECTIVE_NOT_APPLICABLE);
}
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
    auto report = lcb_mtmd_new_load_report();
    const int before = constructions;
    assert(mtmd_vision_bf16_diagnostics_version() == 1);
    assert(mtmd_vision_bf16_classify_metadata(1, "gemma4uv", nullptr) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
    report.version = 987;
    assert(!mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, 1, &report, sizeof(report)-1));
    assert(report.version == 987 && constructions == before);
    assert(!mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, 1, nullptr, sizeof(report)));
    assert(!mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, INT_MIN, &report, sizeof(report)));
    assert(report.reason == MTMD_VISION_BF16_REASON_INVALID_MODE && report.cleanup == MTMD_VISION_BF16_CLEANUP_NOT_REQUIRED);
    assert(constructions == before);
    assert(!mtmd_init_from_file_with_vision_bf16_report("refused", nullptr, params, 1, &report, sizeof(report)));
    assert(report.status == MTMD_VISION_BF16_LOAD_RETENTION_REFUSED && report.cleanup == MTMD_VISION_BF16_CLEANUP_COMPLETE);
    assert(!mtmd_init_from_file_with_vision_bf16_report("throw", nullptr, params, 1, &report, sizeof(report)));
    assert(report.status == MTMD_VISION_BF16_LOAD_FAILED && report.cleanup == MTMD_VISION_BF16_CLEANUP_UNKNOWN);
    auto * reported = mtmd_init_from_file_with_vision_bf16_report("ok", nullptr, params, 0, &report, sizeof(report));
    assert(reported && report.status == MTMD_VISION_BF16_LOAD_SUCCEEDED && report.phase == MTMD_VISION_BF16_PHASE_COMPLETE);
    auto stats = lcb_mtmd_empty_statistics(MTMD_VISION_BF16_EFFECTIVE_UNAVAILABLE);
    assert(!mtmd_context_vision_bf16_statistics(nullptr, &stats, sizeof(stats)));
    assert(stats.effective_storage == MTMD_VISION_BF16_EFFECTIVE_UNAVAILABLE);
    assert(!mtmd_context_vision_bf16_statistics(reported, &stats, sizeof(stats)-1));
    assert(mtmd_context_vision_bf16_statistics(reported, &stats, sizeof(stats)) == 1);
    assert(stats.effective_storage == MTMD_VISION_BF16_EFFECTIVE_NOT_APPLICABLE);
    delete reported;
    delete again; delete old; delete nested; delete retained;
}
''')
        binary = self.work / 'entry'
        subprocess.run(['g++', '-std=c++17', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'bridge'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                        str(fixture), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_clip_boundary_preserves_modality_cleanup_and_typed_preupload_refusal(self):
        # Execute the actual minimal clip init body with a metadata-only loader
        # double. This fixes modality/cleanup control flow, not GGUF execution.
        patched = prepare(self.source, self.work / 'cpu-clip',
                          ROOT / PATCH_DIRECTORY / 'mtmd-vision-bf16-clip-runtime.patch')
        code = patched.read_text()
        start = code.index('struct clip_init_result clip_init(')
        end = code.index('struct clip_cap clip_get_cap(', start)
        entries = code[start:end]
        webgpu = prepare(self.source, self.work / 'webgpu-clip',
                         ROOT / PATCH_DIRECTORY / 'mtmd-webgpu-bf16.patch').read_text()
        gpu_start = webgpu.index('struct clip_init_result clip_init(')
        gpu_end = webgpu.index('struct clip_cap clip_get_cap(', gpu_start)
        entries = '#if LCB_MTMD_WEBGPU_BF16_BACKEND\n' + webgpu[gpu_start:gpu_end] + '\n#else\n' + entries + '\n#endif\n'
        fixture = self.work / 'modality.cpp'
        fixture.write_text(r'''
#include "mtmd-vision-bf16-internal.h"
#include <cassert>
#include <string>
#include <vector>
#undef LOG_ERR
#define LOG_ERR(...) ((void)0)
static int allocated = 0;
static int weight_loads = 0;
static int classifications = 0;
static int schedulers = 0;
static int callback_sets = 0;
static void * callback_data = nullptr;
enum projector_type { PROJECTOR_TYPE_MLP, PROJECTOR_TYPE_GEMMA3NV,
                      PROJECTOR_TYPE_LFM3A_GEN, PROJECTOR_TYPE_SOPRANO, PROJECTOR_TYPE_CHATTERBOX };
struct fixture_model { projector_type proj_type = PROJECTOR_TYPE_MLP; };
struct scheduler_owner {
    ggml_backend_sched_t value = nullptr;
    void reset(ggml_backend_sched_t next) { value = next; }
    ggml_backend_sched_t get() const { return value; }
};
struct clip_ctx {
    const mtmd_vision_bf16_storage vision_bf16_storage;
    mtmd_vision_bf16_statistics vision_bf16_statistics = lcb_mtmd_empty_statistics(MTMD_VISION_BF16_EFFECTIVE_UPSTREAM);
    fixture_model model;
    ggml_backend_t backend = nullptr;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    std::vector<uint8_t> buf_compute_meta;
    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    scheduler_owner sched;
    int max_nodes = 1;
    clip_ctx(clip_context_params &, mtmd_vision_bf16_storage mode = MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND) :
        vision_bf16_storage(mode) { ++allocated; }
    ~clip_ctx() { --allocated; }
};
static bool lcb_mtmd_is_webgpu(ggml_backend_t) { return true; }
projector_type clip_get_projector_type(const clip_ctx * ctx) { return ctx->model.proj_type; }
extern "C" size_t ggml_tensor_overhead(void) { return 1; }
extern "C" size_t ggml_graph_overhead_custom(size_t nodes, bool) { return nodes; }
extern "C" ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t *, ggml_backend_buffer_type_t *,
                                                        int, size_t nodes, bool parallel, bool offload) {
    assert(nodes == 65536 && !parallel && offload); ++schedulers;
    return reinterpret_cast<ggml_backend_sched_t>(uintptr_t(1));
}
extern "C" void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched,
                                                       ggml_backend_sched_eval_callback callback, void * data) {
    assert(sched && callback && data == callback_data); ++callback_sets;
}
struct clip_model_loader {
    bool has_vision;
    bool has_audio;
    bool has_gen_audio;
    std::string path;
    clip_model_loader(const char * path, bool, mtmd_progress_callback, void *) :
        has_vision(std::string(path) == "vision" || std::string(path) == "refused" || std::string(path) == "late"),
        has_audio(std::string(path) == "audio" || std::string(path) == "refused" || std::string(path) == "late"),
        has_gen_audio(std::string(path) == "gen"), path(path) {
        ++classifications;
    }
    void load_hparams(fixture_model & model, clip_modality) {
        if (has_gen_audio) model.proj_type = PROJECTOR_TYPE_CHATTERBOX;
    }
    void load_tensors(clip_ctx & ctx) {
        if (path == "refused" && ctx.vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED)
            throw lcb_mtmd_retention_refused(MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED, "synthetic role refusal");
        ++weight_loads;
    }
    void init_ctx(clip_ctx &) { if (path == "late") throw std::runtime_error("ordinary post-upload failure"); }
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
#if LCB_MTMD_WEBGPU_BF16_BACKEND
    assert(rejected.ctx_v && allocated == 1 && weight_loads == 2);
    delete rejected.ctx_v;
#else
    assert(!rejected.ctx_a && !rejected.ctx_v && allocated == 0 && weight_loads == 1);
#endif
    const int first_weight_loads = weight_loads;
    assert(classifications == 2);
    auto legacy = clip_init("vision", params);
    assert(legacy.ctx_v && legacy.ctx_v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    delete legacy.ctx_v;
    params.use_gpu = false;
    auto cpu = lcb_clip_init_with_vision_bf16_storage("vision", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(cpu.ctx_v && cpu.ctx_v->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    delete cpu.ctx_v;
    assert(allocated == 0 && weight_loads == first_weight_loads + 2 && classifications == 4);
    params.cb_eval = [](ggml_tensor *, bool, void *) { return false; };
    params.cb_eval_user_data = callback_data = &params;
    auto gen = lcb_clip_init_with_vision_bf16_storage("gen", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    assert(gen.ctx_gen_a && !gen.ctx_v && !gen.ctx_a && gen.ctx_gen_a->max_nodes == 65536);
    assert(gen.ctx_gen_a->vision_bf16_storage == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    assert(schedulers == 1 && callback_sets == 1 && gen.ctx_gen_a->buf_compute_meta.size() == 131072);
    delete gen.ctx_gen_a;
    assert(allocated == 0 && weight_loads == first_weight_loads + 3 && classifications == 5);
#if LCB_MTMD_WEBGPU_BF16_BACKEND
    params.use_gpu = true;
    const int before_refusal = weight_loads;
    auto report = lcb_mtmd_new_load_report();
    auto refused = lcb_clip_init_with_vision_bf16_storage("refused", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, &report);
    assert(!refused.ctx_v && !refused.ctx_a && !refused.ctx_gen_a);
    assert(allocated == 0 && weight_loads == before_refusal);
    assert(report.status == MTMD_VISION_BF16_LOAD_RETENTION_REFUSED && report.phase == MTMD_VISION_BF16_PHASE_PREUPLOAD);
    assert(report.cleanup == MTMD_VISION_BF16_CLEANUP_COMPLETE && report.reason == MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED);
    auto legacy_retry = lcb_clip_init_with_vision_bf16_storage("refused", params, MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    assert(legacy_retry.ctx_v && legacy_retry.ctx_a && allocated == 2);
    delete legacy_retry.ctx_v; delete legacy_retry.ctx_a;
    report = lcb_mtmd_new_load_report();
    auto late = lcb_clip_init_with_vision_bf16_storage("late", params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, &report);
    assert(!late.ctx_v && !late.ctx_a && allocated == 0);
    assert(report.status == MTMD_VISION_BF16_LOAD_FAILED && report.cleanup == MTMD_VISION_BF16_CLEANUP_UNKNOWN);
#endif
}
''')
        for backend in (0, 1):
            with self.subTest(webgpu_bf16_backend=backend):
                binary = self.work / ('modality-' + str(backend))
                subprocess.run(['g++', '-std=c++17', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                                '-DLCB_MTMD_WEBGPU_BF16_BACKEND=' + str(backend),
                                '-I' + str(ROOT / 'bridge'), '-I' + str(self.source / 'include'),
                                '-I' + str(self.source / 'ggml/include'), '-I' + str(self.source / 'tools/mtmd'),
                                str(fixture), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
