#include "mtmd-subcontext.h"
#include "mtmd-graph.h"
#include "llama-arch.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "support/check.hpp"
#include "support/mtmd-metadata.hpp"
#include <climits>
#include <cstring>
#include <iostream>
#include <stdexcept>

// The public entry points and context constructor are complete production TUs.
// Small synthetic GGUF inputs pass through the real file loader. The only
// observers are the C GGUF boundary and unused text-model API declarations.
static unsigned loads;

// This external device double identifies a WebGPU buffer type but refuses all
// resident allocations. It proves the rejection boundary, not GPU execution.
struct preupload_device {
    unsigned allocations = 0;
    ggml_backend_reg registry{};
    ggml_backend_device device{};
    ggml_backend_buffer_type buffer = *ggml_backend_cpu_buffer_type();
    preupload_device() {
        registry.iface.get_name = [](ggml_backend_reg_t) { return "WebGPU"; };
        device.reg = &registry;
        device.context = this;
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<preupload_device *>(dev->context)->buffer;
        };
        buffer.device = &device;
        buffer.context = this;
        buffer.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t) -> ggml_backend_buffer_t {
            ++static_cast<preupload_device *>(buft->context)->allocations;
            throw std::runtime_error("controlled resident allocation failure");
        };
        buffer.iface.alloc_buffer_n = nullptr;
    }
};
extern "C" {
gguf_context * __real_gguf_init_from_file(const char *, gguf_init_params);
gguf_context * __wrap_gguf_init_from_file(const char * path, gguf_init_params params) {
    ++loads;
    return __real_gguf_init_from_file(path, params);
}
int32_t llama_model_n_embd_inp(const llama_model *) {
    throw std::runtime_error("unexpected text model embedding query");
}
const llama_vocab * llama_model_get_vocab(const llama_model *) {
    throw std::runtime_error("unexpected text model vocabulary query");
}
llama_rope_type llama_model_rope_type(const llama_model *) {
    throw std::runtime_error("unexpected text model rope query");
}
int32_t llama_vocab_n_tokens(const llama_vocab *) {
    throw std::runtime_error("unexpected vocabulary size query");
}
int32_t llama_token_to_piece(const llama_vocab *, llama_token, char *, int32_t, int32_t, bool) {
    throw std::runtime_error("unexpected text token conversion");
}
llama_model_params llama_model_default_params() {
    throw std::runtime_error("unexpected embedded text model defaults query");
}
llama_model * llama_model_init_from_user(gguf_context *, llama_model_set_tensor_data_t, void *, llama_model_params) {
    throw std::runtime_error("unexpected embedded text model loading");
}
void llama_model_free(llama_model *) {
    throw std::runtime_error("unexpected embedded text model release");
}
}
const char * llama_arch_name(llm_arch) {
    throw std::runtime_error("unexpected embedded text model architecture query");
}
std::string llama_arch_kv_name(llm_arch, llm_kv) {
    throw std::runtime_error("unexpected embedded text model metadata query");
}
std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(mtmd_subcontext *, const mtmd_image_f32_batch &,
                                                   const mtmd_subctx_encode_params *) {
    throw std::runtime_error("unexpected graph factory reached by entry fixture");
}

int main(int argc, char ** argv) {
    BICORE_CHECK(argc == 2);
    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    const auto input = root / "gen-audio.gguf";
    write_mtmd_fixture(input, false, false, true);
    ggml_backend_register(ggml_backend_cpu_reg());
    auto params = mtmd_context_params_default();
    params.use_gpu = false;
    params.n_threads = 1;
    params.warmup = false;
    BICORE_CHECK(mtmd_vision_bf16_storage_version() == 1);
    BICORE_CHECK(mtmd_vision_bf16_diagnostics_version() == 1);
    BICORE_CHECK(mtmd_vision_bf16_classify_metadata(1, "muse-glimmer", nullptr) == MTMD_VISION_BF16_CANDIDATE_REVIEWED);
    mtmd_vision_bf16_load_report report;
    std::memset(&report, 0xa5, sizeof(report));
    const auto untouched = report;
    for (uint32_t size : {uint32_t(0), uint32_t(sizeof(report) - 1), uint32_t(sizeof(report) + 1)}) {
        BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, 1, &report, size));
        BICORE_CHECK(std::memcmp(&report, &untouched, sizeof(report)) == 0);
    }
    BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, 1, nullptr, sizeof(report)));
    BICORE_CHECK(
        !mtmd_init_from_file_with_vision_bf16_report(nullptr, nullptr, params, INT_MIN, &report, sizeof(report)));
    BICORE_CHECK(report.version == 1 && report.size == sizeof(report));
    BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_FAILED);
    BICORE_CHECK(report.reason == MTMD_VISION_BF16_REASON_INVALID_MODE);
    BICORE_CHECK(report.cleanup == MTMD_VISION_BF16_CLEANUP_NOT_REQUIRED);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(nullptr) == MTMD_VISION_BF16_STORAGE_INVALID);
    for (int32_t invalid : {INT_MIN, -2, -1, 2, INT_MAX}) {
        BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_storage(nullptr, nullptr, params, invalid));
    }
    BICORE_CHECK(loads == 0);
    mtmd_context * nested = nullptr;
    struct CallbackState {
        mtmd_context *& nested;
        const std::filesystem::path & input;
    } callback_state{nested, input};
    params.progress_callback_user_data = &callback_state;
    params.progress_callback = [](float, void * data) {
        auto & state = *static_cast<CallbackState *>(data);
        if (state.nested != nullptr)
            return true;
        auto inner = mtmd_context_params_default();
        inner.use_gpu = false;
        inner.n_threads = 1;
        inner.warmup = false;
        state.nested = mtmd_init_from_file_with_vision_bf16_storage(state.input.c_str(), nullptr, inner,
                                                                    MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
        BICORE_CHECK(state.nested != nullptr);
        return true;
    };
    auto * retained = mtmd_init_from_file_with_vision_bf16_report(
        input.c_str(), nullptr, params, MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED, &report, sizeof(report));
    BICORE_CHECK(retained != nullptr && nested != nullptr && loads == 2);
    BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_SUCCEEDED && report.reason == MTMD_VISION_BF16_REASON_NONE);
    BICORE_CHECK(report.phase == MTMD_VISION_BF16_PHASE_COMPLETE);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(nested) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    auto * old = mtmd_init_from_file(input.c_str(), nullptr, params);
    BICORE_CHECK(old != nullptr && mtmd_context_vision_bf16_storage(old) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_storage((root / "missing.gguf").c_str(), nullptr, params, 1));
    BICORE_CHECK(!mtmd_init_from_file_with_vision_bf16_report((root / "missing.gguf").c_str(), nullptr, params, 1,
                                                              &report, sizeof(report)));
    BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_FAILED && report.reason == MTMD_VISION_BF16_REASON_OTHER);
    params.progress_callback = nullptr;
    params.progress_callback_user_data = nullptr;
    // The WebGPU-selected production loader must reject an unreviewed BF16
    // role before allocation. The CPU-selected loader has no such policy.
#if LCB_MTMD_WEBGPU_BF16_RUNTIME == 1
    {
        const auto vision = root / "unreviewed-bias.gguf";
        write_mtmd_fixture(vision, true, false, false, GGML_TYPE_BF16);
        preupload_device fake;
        params.use_gpu = true;
        params.device = &fake.device;
        const auto before = loads;
        BICORE_CHECK(
            !mtmd_init_from_file_with_vision_bf16_report(vision.c_str(), nullptr, params, 1, &report, sizeof(report)));
        BICORE_CHECK(loads == before + 1 && fake.allocations == 0);
        BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_RETENTION_REFUSED);
        BICORE_CHECK(report.reason == MTMD_VISION_BF16_REASON_WEIGHT_UNREVIEWED);
        BICORE_CHECK(report.phase == MTMD_VISION_BF16_PHASE_PREUPLOAD);
        BICORE_CHECK(report.cleanup == MTMD_VISION_BF16_CLEANUP_COMPLETE);
        // A subsequent legacy attempt reaches allocation, without native retry.
        BICORE_CHECK(
            !mtmd_init_from_file_with_vision_bf16_report(vision.c_str(), nullptr, params, 0, &report, sizeof(report)));
        BICORE_CHECK(loads == before + 2 && fake.allocations == 1);
        BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_FAILED);
        BICORE_CHECK(report.reason == MTMD_VISION_BF16_REASON_OTHER);
        BICORE_CHECK(report.phase == MTMD_VISION_BF16_PHASE_UNKNOWN);
        params.device = nullptr;
        params.use_gpu = false;
    }
#endif
    // A cancellation from the real loader is never eligible for policy retry.
    params.progress_callback = [](float, void *) { return false; };
    const auto before_cancel = loads;
    BICORE_CHECK(
        !mtmd_init_from_file_with_vision_bf16_report(input.c_str(), nullptr, params, 1, &report, sizeof(report)));
    BICORE_CHECK(loads == before_cancel + 1);
    BICORE_CHECK(report.status == MTMD_VISION_BF16_LOAD_FAILED && report.reason == MTMD_VISION_BF16_REASON_OTHER);
    params.progress_callback = nullptr;
    auto * again = mtmd_init_from_file_with_vision_bf16_storage(input.c_str(), nullptr, params, 0);
    BICORE_CHECK(again != nullptr && mtmd_context_vision_bf16_storage(again) == MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND);
    BICORE_CHECK(mtmd_context_vision_bf16_storage(retained) == MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED);
    for (auto * context : {again, old, nested, retained})
        mtmd_free(context);
    std::cout << "complete mtmd entry validates raw i32 before loading and preserves reentry policies\n";
}
