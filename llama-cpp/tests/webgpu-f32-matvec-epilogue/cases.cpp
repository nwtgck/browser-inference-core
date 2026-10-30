static std::shared_ptr<Global> allocation_global;
static uintptr_t next_buffer_id = 1;
static void free_memory(ggml_backend_buffer_t b) {
    delete static_cast<ggml_backend_webgpu_buffer_context *>(b->context);
}
static ggml_backend_buffer_t alloc_memory(ggml_backend_buffer_type_t type, size_t size) {
    ggml_backend_buffer_i iface{};
    iface.free_buffer = free_memory;
    iface.get_base = ggml_backend_webgpu_buffer_get_base;
    auto * meta =
        new ggml_backend_webgpu_buffer_context{{next_buffer_id++, size, wgpu::BufferUsage::Storage}, allocation_global};
    return ggml_backend_buffer_init(type, iface, meta, size);
}
static const char * memory_name(ggml_backend_buffer_type_t) {
    return "recorded-f32-epilogue-metadata";
}
static size_t memory_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
static ggml_backend_buffer_type memory_type = [] {
    ggml_backend_buffer_type t{};
    t.iface.get_name = memory_name;
    t.iface.get_alignment = memory_alignment;
    t.iface.alloc_buffer = alloc_memory;
    return t;
}();
static int checks = 0;
static void require(bool b, const char * why) {
    ++checks;
    if (!b)
        throw std::runtime_error(why);
}
struct Fixture {
    ggml_context * ctx = ggml_init({1 << 20, nullptr, true});
    ggml_cgraph * graph = nullptr;
    ggml_tensor *w = nullptr, *x = nullptr, *mm = nullptr, *first = nullptr, *unary = nullptr, *output = nullptr;
    ggml_gallocr_t allocator = nullptr;
    int index = -1;
    Fixture(int k = 10240, int m = 4, int n = 1, bool input_produced = false, bool lora = false, bool crop = false,
            bool offset_input = false) {
        require(ctx, "context");
        graph = ggml_new_graph_custom(ctx, 128, false);
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        auto * input_root = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k + (offset_input ? 1 : 0), n);
        ggml_set_input(w);
        ggml_set_input(input_root);
        x = offset_input ? ggml_view_2d(ctx, input_root, k, n, k * 4, 4) : input_root;
        if (input_produced)
            x = ggml_scale(ctx, x, 0.75f);
        mm = ggml_mul_mat(ctx, w, x);
        auto * source = mm;
        if (lora)
            source = ggml_add(ctx, source, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n));
        first = ggml_scale_bias(ctx, source, 0.25f, 0.125f);
        source = first;
        if (crop) {
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            source = ggml_get_rows(ctx, source, ids);
        }
        unary = ggml_sigmoid(ctx, source);
        output = ggml_scale_bias(ctx, unary, 2.0f, -0.125f);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < graph->n_nodes; i++)
            if (graph->nodes[i] == mm)
                index = i;
        require(index >= 0, "matvec index");
    }
    void allocate(webgpu_context & c) {
        allocation_global = c->global_ctx;
        allocator = ggml_gallocr_new(&memory_type);
        require(ggml_gallocr_alloc_graph(allocator, graph), "actual gallocr allocation");
    }
    ~Fixture() {
        if (allocator)
            ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
};
static void graph_checks() {
    for (int k : {256, 10240, 10241}) {
        Fixture f(k);
        f32_matvec_epilogue_plan p;
        require(plan_f32_matvec_epilogue(f.graph, f.index, p), "ordinary exact4 graph");
        for (int truncated = 1; truncated <= 3; truncated++) {
            auto view = ggml_graph_view(f.graph, f.index, f.index + truncated);
            require(!plan_f32_matvec_epilogue(&view, 0, p), "callback/truncated graph boundary");
        }
        const auto flags = f.mm->flags;
        f.mm->flags |= GGML_TENSOR_FLAG_OUTPUT;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "observed MM fallback");
        f.mm->flags = flags;
        f.mm->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "noncompute MM fallback");
        f.mm->flags = flags;
        f.mm->view_src = f.x;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "intermediate view fallback");
        f.mm->view_src = nullptr;
        int32_t old;
        std::memcpy(&old, f.unary->op_params, 4);
        int32_t silu = GGML_UNARY_OP_SILU;
        std::memcpy(f.unary->op_params, &silu, 4);
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "SILU excluded");
        std::memcpy(f.unary->op_params, &old, 4);
        f.first->src[1] = f.x;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "extra source fallback");
        f.first->src[1] = nullptr;
        auto * side = ggml_scale(f.ctx, f.output, 3);
        ggml_build_forward_expand(f.graph, side);
        ggml_build_forward_expand(f.graph, ggml_scale(f.ctx, f.output, 5));
        int output_index = -1;
        for (int i = 0; i < f.graph->n_nodes; i++)
            if (f.graph->nodes[i] == f.output)
                output_index = i;
        require(output_index >= 0 && ggml_node_get_use_count(f.graph, output_index) >= 2,
                "final output has actual multiuse");
        require(plan_f32_matvec_epilogue(f.graph, f.index, p), "final multiuse allowed");
    }
    for (int which = 0; which < 3; which++) {
        Fixture f;
        auto * node = which == 0 ? f.mm : which == 1 ? f.first : f.unary;
        ggml_build_forward_expand(f.graph, ggml_scale(f.ctx, node, 3));
        f32_matvec_epilogue_plan p;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "internal multiuse fallback");
    }
    for (int which = 0; which < 4; which++) {
        Fixture f(10240, which == 0 ? 3 : 4, which == 1 ? 2 : 1, false, which == 2, which == 3);
        f32_matvec_epilogue_plan p;
        require(!plan_f32_matvec_epilogue(f.graph, f.index, p), "M/N/LoRA/crop excluded");
        if (which == 2) {
            scale_unary_plan old;
            int at = -1;
            for (int i = 0; i < f.graph->n_nodes; i++)
                if (f.graph->nodes[i] == f.first)
                    at = i;
            require(plan_scale_unary(f.graph, at, 3, old), "fallback leaves existing3op available");
        }
    }
}
static void allocation_checks() {
    int natural_reused = 0, natural_disjoint = 0;
    for (bool produced : {false, true})
        for (int k : {256, 10240, 10241}) {
            auto c = std::make_shared<Context>();
            Fixture f(k, 4, 1, produced);
            f.allocate(c);
            f32_matvec_epilogue_plan p;
            int skip = 1;
            require(ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip).has_value() && skip == 4,
                    "real graph+gallocr admits");
            if (f.output->buffer == f.mm->buffer && f.output->data == f.mm->data)
                natural_reused++;
            else
                natural_disjoint++;
            auto fail = [&](const char * why) {
                int n = 1;
                require(!ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, n).has_value() && n == 1, why);
            };
            const auto original = f.output->data;
            f.output->data = f.x->data;
            fail("final dst overlaps input at fusion start");
            f.output->data = original;
            const auto wdata = f.w->data;
            f.w->data = f.x->data;
            fail("weight/input pair payload overlap");
            f.w->data = wdata;
            const auto original_meta =
                static_cast<ggml_backend_webgpu_buffer_context *>(f.w->buffer->context)->global_ctx;
            auto * meta = static_cast<ggml_backend_webgpu_buffer_context *>(f.w->buffer->context);
            meta->global_ctx = std::make_shared<Global>();
            fail("foreign device/context");
            meta->global_ctx = original_meta;
            meta->buffer.usage = wgpu::BufferUsage::None;
            fail("missing Storage");
            meta->buffer.usage = wgpu::BufferUsage::Storage;
            const auto extent = meta->buffer.bytes;
            meta->buffer.bytes = 4;
            fail("physical buffer extent");
            meta->buffer.bytes = extent;
            auto & l = c->global_ctx->capabilities.limits;
            const auto saved = l;
            l.maxStorageBuffersPerShaderStage = 2;
            fail("storage limit");
            l = saved;
            l.maxBindingsPerBindGroup = 3;
            fail("total binding limit");
            l = saved;
            l.maxUniformBufferBindingSize = 64;
            fail("80-byte uniform requires128-byte runtime slot");
            l = saved;
            const auto slot = c->param_arena.slot_size;
            c->param_arena.slot_size = 79;
            fail("actual slot smaller than80");
            c->param_arena.slot_size = l.maxUniformBufferBindingSize + 1;
            fail("actual slot exceeds uniform limit");
            c->param_arena.slot_size = slot;
            l.maxComputeInvocationsPerWorkgroup = 128;
            fail("WG256 limit");
            l = saved;
            l.minStorageBufferOffsetAlignment = 3;
            fail("invalid alignment");
            l = saved;
            // Separate tensor roots and distinct wrapper objects can still share one native GPUBuffer.
            ggml_backend_buffer clone = *f.output->buffer;
            auto cloned_meta = *meta;
            clone.context = &cloned_meta;
            auto * oldbuf = f.output->buffer;
            f.output->buffer = &clone;
            f.output->data = f.x->data;
            fail("distinct roots/wrappers same native buffer overlap");
            f.output->buffer = oldbuf;
            f.output->data = original;
            // Pure range edge: payloads touch but aligned prefixes overlap. The actual admission must reject.
            const auto original_x = f.x->data;
            f.x->data = reinterpret_cast<void *>(uintptr_t(original_x) + 4);
            f.output->data = reinterpret_cast<void *>(uintptr_t(f.x->data) + ggml_nbytes(f.x) + 4);
            require(uintptr_t(f.output->data) - uintptr_t(webgpu_ptr_base) + 16 <= oldbuf->size,
                    "aligned-only fixture is in real allocation");
            require(static_cast<ggml_backend_webgpu_buffer_context *>(f.x->buffer->context)->buffer.Get() ==
                        static_cast<ggml_backend_webgpu_buffer_context *>(f.output->buffer->context)->buffer.Get(),
                    "aligned-only fixture same native buffer");
            require(scale_unary_disjoint(uintptr_t(f.x->data), ggml_nbytes(f.x), uintptr_t(f.output->data), 16),
                    "aligned-only payloads disjoint");
            const uint64_t xoffset = uintptr_t(f.x->data) - uintptr_t(webgpu_ptr_base),
                           yoffset = uintptr_t(f.output->data) - uintptr_t(webgpu_ptr_base);
            require(!scale_unary_disjoint(xoffset & ~uint64_t(255), (xoffset % 256) + ggml_nbytes(f.x),
                                          yoffset & ~uint64_t(255), (yoffset % 256) + 16),
                    "aligned-only bindings overlap");
            fail("aligned binding overlap after disjoint payload");
            f.output->data = original;
            f.x->data = original_x;
            f.x->view_src = f.w;
            f.x->view_offs = 4;
            fail("view/root offset mismatch");
            f.x->view_src = nullptr;
            f.x->view_offs = 0;
            require(ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip).has_value(),
                    "restored admission and skip");
        }
    require(natural_reused > 0, "actual allocator final output reuses unused MM allocation");
    std::cout << "{\"allocationReusedMM\":" << natural_reused << ",\"allocationDistinctMM\":" << natural_disjoint
              << "}\n";
}
static void runtime_and_factory_checks(const char * directory) {
    using Kind = ggml_webgpu_shader_lib_context::matvec_post_op;
    for (bool sg : {false, true})
        for (int subgroup : {16, 32})
            for (int k : {1, 4, 127, 128, 10240, 10241, 16383, 16384}) {
                auto c = std::make_shared<Context>();
                auto & cap = c->global_ctx->capabilities;
                cap.supports_subgroups = sg;
                cap.min_subgroup_size = cap.max_subgroup_size = subgroup;
                Fixture f(k);
                f.allocate(c);
                int skip = 1;
                const auto original = ggml_webgpu_mul_mat(c, f.w, f.x, f.mm);
                require(original.num_kernels == 1 && original.dispatches[0].params.size() == 16, "None keeps16words");
                const auto sizes_before = size_queries, usages_before = usage_queries;
                const auto fused = ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip);
                require(fused.has_value() && skip == 4 && fused->num_kernels == 1, "actual fused single encode");
                std::set<void *> unique;
                for (auto * t : {f.w, f.x, f.output})
                    unique.insert(static_cast<ggml_backend_webgpu_buffer_context *>(t->buffer->context)->buffer.Get());
                require(size_queries - sizes_before == unique.size() && usage_queries - usages_before == unique.size(),
                        "native facts once per distinct buffer");
                const auto &old = original.dispatches[0], &post = fused->dispatches[0];
                require(post.params.size() == 20 && post.workgroups == old.workgroups, "80B params unchanged grid");
                auto expected = old.params;
                expected[2] = uint32_t(ggml_webgpu_tensor_misalignment(c, f.output) / 4);
                require(std::equal(expected.begin(), expected.end(), post.params.begin()),
                        "base16word onlyfinaldst offset");
                require(post.params[16] == ggml_webgpu_u32_from_f32(.25f) &&
                            post.params[17] == ggml_webgpu_u32_from_f32(.125f) &&
                            post.params[18] == ggml_webgpu_u32_from_f32(2.f) &&
                            post.params[19] == ggml_webgpu_u32_from_f32(-.125f),
                        "copied coefficients");
                const auto * d = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(post.pipeline.context.get());
                require(d->wg_size == 256 && d->outputs_per_wg == 4 && d->vec_size == (k % 4 ? 1u : 4u),
                        "original reduction geometry");
                require(post.pipeline.id != old.pipeline.id &&
                            post.pipeline.name == old.pipeline.name + "_scale_sigmoid_scale",
                        "separate kind label/cache");
                require(ggml_webgpu_mul_mat(c, f.w, f.x, f.mm).dispatches[0].pipeline.id == old.pipeline.id,
                        "None cache afterpost");
                int skip2 = 1;
                require(ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip2)->dispatches[0].pipeline.id ==
                            post.pipeline.id,
                        "post cache stable");
                ggml_set_op_params_f32(f.first, 0, .375f);
                int skip3 = 1;
                const auto changed = ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip3);
                require(changed->dispatches[0].pipeline.id == post.pipeline.id &&
                            changed->dispatches[0].params[16] == ggml_webgpu_u32_from_f32(.375f),
                        "coefficients not cachekey and ownedcopy");
                require(post.params[16] == ggml_webgpu_u32_from_f32(.25f), "old descriptor retains copiedcoefficients");
                const auto prior_created = created, prior_encoded = encoded_calls;
                require(ggml_webgpu_mul_mat(c, f.w, f.x, f.output, Kind(99), nullptr).num_kernels == 0,
                        "unknown post rejects");
                require(ggml_webgpu_mul_mat(c, f.w, f.x, f.output, Kind::scale_sigmoid_scale, nullptr).num_kernels == 0,
                        "missing coeff rejects");
                require(created == prior_created && encoded_calls == prior_encoded, "invalid post no pipelineorencode");
                const std::string stem =
                    std::string(directory) + "/" + (sg ? "sg" : "wg") + "-" + (k % 4 ? "scalar" : "vec4");
                std::ofstream(stem + "-none.wgsl") << old.pipeline.wgsl;
                std::ofstream(stem + "-post.wgsl") << post.pipeline.wgsl;
            }
    {
        auto c = std::make_shared<Context>();
        Fixture f(16385);
        f.allocate(c);
        int skip = 1;
        const auto before = encoded_calls;
        require(!ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip) && skip == 1 && encoded_calls == before,
                "K upperbound beforeencode");
    }
    for (bool sg : {false, true}) {
        auto c = std::make_shared<Context>();
        c->global_ctx->capabilities.supports_subgroups = sg;
        Fixture f(10240, 4, 1, false, false, false, true);
        f.allocate(c);
        int skip = 1;
        const auto fused = ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip);
        require(fused && skip == 4, "dense offset view admitted");
        const auto & d = fused->dispatches[0];
        require(d.params[1] % 4 == 1, "actual binding-relative unaligned input");
        require(static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(d.pipeline.context.get())->vec_size == 1,
                "offset forces scalar factory");
        f.x->nb[1] += 4;
        skip = 1;
        const auto before = encoded_calls;
        require(!ggml_webgpu_f32_matvec_epilogue(c, f.graph, f.index, skip) && skip == 1 && encoded_calls == before,
                "padded input remains fallback");
    }
    for (bool post_first : {false, true}) {
        RecordingLibrary lib;
        Fixture f(10240);
        ggml_webgpu_shader_lib_context ctx{};
        ctx.src0 = f.w;
        ctx.src1 = f.x;
        ctx.dst = f.output;
        ctx.max_wg_size = 256;
        unsigned plain = 0, post = 0;
        for (int i = 0; i < 4; i++) {
            ctx.matvec_epilogue = ((i % 2 == 0) == post_first) ? Kind::scale_sigmoid_scale : Kind::none;
            const auto p = lib.get_mul_mat_vec_pipeline(ctx, false, true);
            auto & saved = ctx.matvec_epilogue == Kind::none ? plain : post;
            require(!saved || saved == p.id, "reversed kind cache stability");
            saved = p.id;
        }
        require(plain && post && plain != post, "reverse kind distinct");
        const auto before = created;
        ctx.matvec_epilogue = Kind(255);
        require(!lib.get_mul_mat_vec_pipeline(ctx, false, true).context && created == before,
                "unknown factory no lookupcreation");
        ctx.matvec_epilogue = Kind::scale_sigmoid_scale;
        require(!lib.get_mul_mat_vec_pipeline(ctx, true, true).context && created == before,
                "overlap factory rejected");
    }
}
int main(int argc, char ** argv) {
    try {
        require(argc == 2, "shader output directory");
        graph_checks();
        allocation_checks();
        runtime_and_factory_checks(argv[1]);
        std::cout << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
