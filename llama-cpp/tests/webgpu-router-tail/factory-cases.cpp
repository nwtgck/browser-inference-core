}
;
int main(int argc, char ** argv) {
    if (argc != 3) {
        throw std::runtime_error("output-directory workgroup-size");
    }
    const std::string directory = argv[1];
    const uint32_t wg = std::stoul(argv[2]);
    if (wg < 32 || wg > 1024 || (wg & (wg - 1))) {
        throw std::runtime_error("workgroup");
    }
    ggml_tensor source{}, other{}, output{};
    source.type = other.type = output.type = GGML_TYPE_F32;
    for (auto * tensor : {&source, &other, &output}) {
        for (auto & n : tensor->ne) {
            n = 1;
        }
    }
    source.data = reinterpret_cast<void *>(uintptr_t(0x10000));
    other.data = reinterpret_cast<void *>(uintptr_t(0x20000));
    output.data = reinterpret_cast<void *>(uintptr_t(0x30000));
    ggml_webgpu_shader_lib_context context{};
    context.src0 = &source;
    context.src1 = &other;
    context.dst = &output;
    context.max_wg_size = wg;
    Library library;
    auto save = [&](const std::string & name, const webgpu_pipeline & pipeline) {
        std::ofstream(directory + "/" + name + ".wgsl") << pipeline.code;
        std::cout << name << ' ' << pipeline.name << '\n';
    };
    other.type = GGML_TYPE_I32;
    output.op = GGML_OP_GET_ROWS;
    save("get_rows", library.get_get_rows_pipeline(context, false));
    other.type = GGML_TYPE_F32;
    output.op = GGML_OP_SUM_ROWS;
    save("sum_rows", library.get_sum_rows_pipeline(context));
    output.op = GGML_OP_CLAMP;
    save("clamp", library.get_unary_pipeline(context));
    output.data = source.data;
    save("clamp_inplace", library.get_unary_pipeline(context));
    output.data = reinterpret_cast<void *>(uintptr_t(0x30000));
    output.op = GGML_OP_DIV;
    save("div", library.get_binary_pipeline(context, false));
    output.data = source.data;
    save("div_inplace", library.get_binary_pipeline(context, false));
    save("router_tail", library.get_router_tail_pipeline(wg, false));
    save("router_tail_merged", library.get_router_tail_pipeline(wg, true));
    if (library.get_router_tail_pipeline(wg, false).id == library.get_router_tail_pipeline(wg, true).id) {
        return 4;
    }
}
