#include "mtmd-graph.h"
#include "ggml-cpu.h"
#include "support/check.hpp"
#include "support/allocation-fault.hpp"
#include <iostream>
#include <stdexcept>
#include <type_traits>

// Ownership uses the real CPU backend, scheduler and complete subcontext TU.
// A failing graph factory is the external boundary for this lifetime test;
// complete graph builders are exercised by mtmd-model-graphs instead.
static unsigned graph_requests;
std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(mtmd_subcontext *, const mtmd_image_f32_batch &,
                                                   const mtmd_subctx_encode_params *) {
    ++graph_requests;
    throw std::runtime_error("controlled graph factory failure");
}

static void check_owner(mtmd_backends & owner, ggml_backend_t cpu, ggml_backend_sched_t scheduler) {
    BICORE_CHECK(owner.backend == cpu);
    BICORE_CHECK(owner.backend_cpu == cpu);
    BICORE_CHECK(owner.sched.get() == scheduler);
    BICORE_CHECK(owner.ptrs.size() == 1 && owner.ptrs[0] == cpu);
    BICORE_CHECK(owner.bufts.size() == 1);
    BICORE_CHECK(std::string(ggml_backend_name(cpu)) == "CPU");
    BICORE_CHECK(owner.get_mem_compute().size() == 1);
}

int main() {
    BICORE_CHECK(mtmd_format("").empty());
    BICORE_CHECK(mtmd_format("plain") == "plain");
    const auto formatted = mtmd_format("image %ux%u<|media_content|>", 480u, 336u);
    BICORE_CHECK(formatted == "image 480x336<|media_content|>");
    const auto embedded = mtmd_format("%c", 0);
    BICORE_CHECK(embedded.size() == 1 && embedded[0] == '\0');
    static_assert(!std::is_copy_constructible_v<mtmd_backends>);
    static_assert(!std::is_copy_assignable_v<mtmd_backends>);
    ggml_backend_register(ggml_backend_cpu_reg());
    mtmd_backends owner(nullptr, 2, 128, nullptr, nullptr);
    BICORE_CHECK(owner.threadpool != nullptr);
    const auto cpu = owner.backend_cpu;
    const auto scheduler = owner.sched.get();
    BICORE_CHECK(cpu != nullptr && scheduler != nullptr);
    ggml_tensor embedding{};
    embedding.ne[1] = 4;
    mtmd_model model;
    model.modality = MTMD_MODALITY_GEN_AUDIO;
    model.proj_type = PROJECTOR_TYPE_MLP;
    model.mm_2_w = &embedding;
    model.max_nodes = 128;
    const mtmd_subctx_params no_warmup{MTMD_FLASH_ATTN_TYPE_DISABLED, false};
    {
        mtmd_subcontext first(model, owner, no_warmup);
        mtmd_subcontext second(model, owner, no_warmup);
        BICORE_CHECK(&first.backends == &owner && &second.backends == &owner);
        BICORE_CHECK(&first.model == &model && &second.model == &model);
        BICORE_CHECK(!first.buf_compute_meta.empty() && !second.buf_compute_meta.empty());
        check_owner(owner, cpu, scheduler);
    }
    BICORE_CHECK(graph_requests == 0);
    check_owner(owner, cpu, scheduler);
    allocation_fault::state = {true, -1, 0};
    { mtmd_subcontext measured(model, owner, no_warmup); }
    allocation_fault::state.observing = false;
    const auto allocation_count = allocation_fault::state.allocations;
    BICORE_CHECK(allocation_count > 0);
    for (size_t index = 0; index < allocation_count; ++index) {
        allocation_fault::state = {true, ptrdiff_t(index), 0};
        bool caught = false;
        try {
            mtmd_subcontext failed(model, owner, no_warmup);
        } catch (const std::bad_alloc &) {
            caught = true;
        }
        allocation_fault::state.observing = false;
        allocation_fault::state.fail_after = -1;
        BICORE_CHECK(caught);
        check_owner(owner, cpu, scheduler);
    }
    for (const auto modality : {MTMD_MODALITY_GEN_AUDIO, MTMD_MODALITY_VISION}) {
        model.modality = modality;
        model.hparams.warmup_image_size = 2;
        bool caught = false;
        try {
            mtmd_subcontext failed(model, owner, {MTMD_FLASH_ATTN_TYPE_DISABLED, true});
        } catch (const std::runtime_error & error) {
            caught = true;
            BICORE_CHECK(std::string(error.what()) == "controlled graph factory failure");
        }
        BICORE_CHECK(caught);
        check_owner(owner, cpu, scheduler);
    }
    BICORE_CHECK(graph_requests == 2);
    model.modality = MTMD_MODALITY_GEN_AUDIO;
    { mtmd_subcontext reused(model, owner, no_warmup); }
    check_owner(owner, cpu, scheduler);
    std::cout << "real shared owner survives borrowed destruction, graph failure and reentry\n";
}
