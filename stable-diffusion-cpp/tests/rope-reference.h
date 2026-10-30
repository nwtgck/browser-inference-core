// Frozen pre-optimization graph from pinned stable-diffusion.cpp (MIT).
// Keep independent of the prepared-matrix helper under test.
__STATIC_INLINE__ ggml_tensor * reference_apply_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe,
                                                     bool rope_interleaved = true) {
    // x: [N, L, n_head, d_head]
    // pe: [L, d_head/2, 2, 2], [[cos, -sin], [sin, cos]]
    int64_t d_head = x->ne[0];
    int64_t n_head = x->ne[1];
    int64_t L = x->ne[2];
    int64_t N = x->ne[3];
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [N, n_head, L, d_head]
    if (rope_interleaved) {
        x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N); // [N * n_head, L, d_head/2, 2]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));      // [2, N * n_head, L, d_head/2]
    } else {
        x = ggml_reshape_4d(ctx, x, d_head / 2, 2, L, n_head * N);      // [N * n_head, L, 2, d_head/2]
        x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 3, 1)); // [2, N * n_head, L, d_head/2]
    }

    int64_t offset = x->nb[2] * x->ne[2];
    auto x_0 =
        ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0); // [N * n_head, L, d_head/2]
    auto x_1 =
        ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1); // [N * n_head, L, d_head/2]
    x_0 = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]); // [N * n_head, L, d_head/2, 1]
    x_1 = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]); // [N * n_head, L, d_head/2, 1]
    auto temp_x = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
    x_0 = ggml_repeat(ctx, x_0, temp_x); // [N * n_head, L, d_head/2, 2]
    x_1 = ggml_repeat(ctx, x_1, temp_x); // [N * n_head, L, d_head/2, 2]

    pe = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2)); // [2, L, d_head/2, 2]
    offset = pe->nb[2] * pe->ne[2];
    auto pe_0 =
        ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0); // [L, d_head/2, 2]
    auto pe_1 =
        ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1); // [L, d_head/2, 2]

    auto x_out =
        ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1)); // [N * n_head, L, d_head/2, 2]
    if (!rope_interleaved) {
        x_out = ggml_cont(ctx, ggml_permute(ctx, x_out, 1, 0, 2, 3)); // [N * n_head, L, x, d_head/2]
    }
    x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N); // [N*n_head, L, d_head]
    return x_out;
}
