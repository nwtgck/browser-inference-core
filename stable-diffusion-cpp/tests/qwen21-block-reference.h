#pragma once
// Frozen block/model from applied 008 (raw SD 88411ef1 + image patches through 0030).
// Do not call the candidate activation/modulation helpers from this reference.
#include "model/diffusion/qwen_image_2_1.hpp"
namespace qwen21_reference {
using namespace Qwen;
    class ReferenceQwenImage21TransformerBlock : public GGMLBlock {
    public:
        ReferenceQwenImage21TransformerBlock(const QwenImage21Config& config) {
            blocks["img_norm1"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["img_norm2"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["attn"]      = std::make_shared<QwenImage21Attention>(config);
            if (config.fused_mlp) {
                blocks["img_mlp.gate_up"] = std::make_shared<Linear>(config.hidden_size, 2 * config.intermediate_size, false);
            } else {
                blocks["img_mlp.proj"]       = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
                blocks["img_mlp.gate_layer"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
            }
            blocks["img_mlp.out"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, false);
        }

        static ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* params, int64_t prefix_length, bool gate = false) {
            auto rows  = ggml_ext_chunk(ctx, params, 2, 1);
            auto apply = [&](ggml_tensor* part, ggml_tensor* row) {
                row = gate ? ggml_tanh(ctx, row) : ggml_scale_bias(ctx, row, 1.f, 1.f);
                return ggml_mul(ctx, part, row);
            };
            auto target = apply(ggml_ext_slice(ctx, x, 1, prefix_length, x->ne[1]), rows[0]);
            if (prefix_length == 0) {
                return target;
            }
            auto prefix = apply(ggml_ext_slice(ctx, x, 1, 0, prefix_length), rows[1]);
            return ggml_concat(ctx, prefix, target, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const std::vector<ggml_tensor*>& modulation, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, const Rope::PreparedMatrix* prepared_pe = nullptr) {
            const int64_t prefix_length = cache.mode == QwenImage21PrefixCache::Mode::REUSE ? 0 : layout.prefix_length;
            auto h                      = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"])->forward(ctx, x);
            h                           = modulate(ctx->ggml_ctx, h, modulation[0], prefix_length);
            h                           = std::dynamic_pointer_cast<QwenImage21Attention>(blocks["attn"])->forward(ctx, h, pe, layout.segments, masks, cache, prepared_pe);
            x                           = ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[1], prefix_length, true));
            h                           = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"])->forward(ctx, x);
            h                           = modulate(ctx->ggml_ctx, h, modulation[2], prefix_length);
            ggml_tensor* gate;
            auto fused = blocks.find("img_mlp.gate_up");
            if (fused != blocks.end()) {
                auto gate_up = std::dynamic_pointer_cast<Linear>(fused->second)->forward(ctx, h);
                auto parts   = ggml_ext_chunk(ctx->ggml_ctx, gate_up, 2, 0);
                gate         = parts[0];
                h            = parts[1];
            } else {
                gate = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.gate_layer"])->forward(ctx, h);
                h    = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.proj"])->forward(ctx, h);
            }
            h = ggml_mul(ctx->ggml_ctx, h, ggml_silu(ctx->ggml_ctx, gate));
            h = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.out"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, modulate(ctx->ggml_ctx, h, modulation[3], prefix_length, true));
        }
    };

    class ReferenceQwenImage21Model : public GGMLBlock {
        QwenImage21Config config;

    public:
        ReferenceQwenImage21Model(const QwenImage21Config& config)
            : config(config) {
            blocks["time_text_embed.timestep_embedder"] = std::make_shared<TimestepEmbedding>(256, config.hidden_size, 0, 0, false);
            blocks["txt_in"]                            = std::make_shared<QwenImage21TextProjection>(config);
            blocks["img_in"]                            = std::make_shared<Linear>(config.in_channels, config.hidden_size, false);
            blocks["modulation.1"]                      = std::make_shared<Linear>(config.hidden_size, 4 * config.hidden_size, false);
            blocks["norm_out.linear"]                   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
            blocks["norm_out.norm"]                     = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["proj_out"]                          = std::make_shared<Linear>(config.hidden_size, config.out_channels, false);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<ReferenceQwenImage21TransformerBlock>(config);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache) {
            auto time = ggml_concat(ctx->ggml_ctx, timestep, ggml_ext_zeros_like(ctx->ggml_ctx, timestep), 0);
            // Runtime flow timesteps already use the [0, 1000] scale.
            time               = ggml_ext_timestep_embedding(ctx->ggml_ctx, time, 256, 10000, 1.f);
            time               = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_text_embed.timestep_embedder"])->forward(ctx, time);
            time               = ggml_silu(ctx->ggml_ctx, time);
            auto modulation    = std::dynamic_pointer_cast<Linear>(blocks["modulation.1"])->forward(ctx, time);
            auto mod           = ggml_ext_chunk(ctx->ggml_ctx, modulation, 4, 0);
            auto img_in        = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            ggml_tensor* joint = nullptr;
            if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                joint = img_in->forward(ctx, DiT::patchify(ctx->ggml_ctx, x, 1, 1));
            } else {
                auto text = std::dynamic_pointer_cast<QwenImage21TextProjection>(blocks["txt_in"])->forward(ctx, context);
                for (const auto& segment : layout.segments) {
                    ggml_tensor* h;
                    if (segment.image_index < 0) {
                        h = ggml_ext_slice(ctx->ggml_ctx, text, 1, segment.context_start,
                                           segment.context_start + segment.end - segment.start);
                    } else {
                        auto image = segment.image_index == static_cast<int>(refs.size()) ? x : refs[segment.image_index];
                        h          = img_in->forward(ctx, DiT::patchify(ctx->ggml_ctx, image, 1, 1));
                    }
                    joint = joint == nullptr ? h : ggml_concat(ctx->ggml_ctx, joint, h, 1);
                }
            }
            sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.prelude", "joint");
            std::optional<Rope::PreparedMatrix> prepared_pe;
#ifdef SD_BROWSER_WEBGPU
            if (sd_backend_is(ctx->backend, "WebGPU")) {
                prepared_pe = Rope::prepare_matrix(ctx->ggml_ctx, pe);
            }
#endif
            for (int i = 0; i < config.num_layers; ++i) {
                const std::string layer = "transformer_blocks." + std::to_string(i);
                auto layer_cache        = cache;
                layer_cache.name        = cache.name + "." + std::to_string(i);
                layer_cache.cut_group   = "qwen_image_2_1." + layer;
                auto block              = std::dynamic_pointer_cast<ReferenceQwenImage21TransformerBlock>(blocks[layer]);
                joint                   = block->forward(ctx, joint, mod, pe, layout, masks, layer_cache, prepared_pe ? &*prepared_pe : nullptr);
                sd::ggml_graph_cut::mark_graph_cut(joint, layer_cache.cut_group, "joint");
            }
            if (cache.mode != QwenImage21PrefixCache::Mode::REUSE) {
                joint = ggml_ext_slice(ctx->ggml_ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            }
            auto scale = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_ext_chunk(ctx->ggml_ctx, time, 2, 1)[0]);
            joint      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out.norm"])->forward(ctx, joint);
            joint      = ggml_mul(ctx->ggml_ctx, joint, ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f));
            joint      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, joint);
            return DiT::unpatchify_and_crop(ctx->ggml_ctx, joint, x->ne[1], x->ne[0], 1, 1);
        }
    };

}
