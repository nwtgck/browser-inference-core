#include "ggml.h"
#include "ggml-cpu.h"
#include "stable-diffusion.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t sdc_abi_version();
extern "C" int sdc_test_qwen_timestep(const char*);
extern "C" int sdc_test_bf16_weights(const char*);
extern "C" int sdc_test_graph_walk();
extern "C" int sdc_test_conv3d_bias(const char*);
extern "C" int sdc_test_webgpu_performance(const char*);
extern "C" uint32_t sdc_model_io_capabilities();
extern "C" uint64_t sdc_test_safetensors_offset(const char*);
extern "C" uint32_t sdc_test_safetensors_value(const char*);
extern "C" uint32_t sdc_test_model_tensor_count(const char*);
extern "C" void sdc_sd_ctx_params_init(uint64_t);
extern "C" void sdc_sd_img_gen_params_init(uint64_t);
extern "C" void sdc_sd_vid_gen_params_init(uint64_t);
extern "C" int32_t sdc_generate_video(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t);
extern "C" uint64_t sdc_test_gguf_offset(const char*);
extern "C" uint32_t sdc_test_gguf_value(const char*);
extern "C" void sdc_test_callbacks();
extern "C" void sdc_sd_set_log_callback(uint64_t, uint64_t);
extern "C" void sdc_sd_set_progress_callback(uint64_t, uint64_t);
static int logs_seen=0, progress_seen=0;
static std::string placement_message;
static std::vector<std::string> placement_ops_messages;
static void log_callback(sd_log_level_t, const char* message, void* data) {
    if (std::string(message).find("native callback probe") != std::string::npos && uintptr_t(data)==17) ++logs_seen;
    if (std::string(message).find("browser-placement-ops-v1 ") != std::string::npos) placement_ops_messages.push_back(message);
    if (std::string(message).find("browser-placement-v1 ") != std::string::npos) placement_message = message;
}
static void progress_callback(int step, int steps, float seconds, void* data) {
    if (step==1 && steps==4 && seconds==0.125f && uintptr_t(data)==19) ++progress_seen;
}
ggml_tensor* ggml_ext_group_norm_32(ggml_context*, ggml_tensor*);
static void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
static void u32(std::ofstream& out, uint32_t v) { for (int i=0;i<4;++i) out.put(char(v >> (8*i))); }
static void u64(std::ofstream& out, uint64_t v) { for (int i=0;i<8;++i) out.put(char(v >> (8*i))); }
static void str(std::ofstream& out, const std::string& s) { u64(out,s.size());out.write(s.data(),s.size()); }
// Every filler is 1 GiB, so the GGUF can be indexed on Wasm32 without any
// individual tensor exceeding its address width. Only the final 4 bytes are read.
static uint64_t sparse_gguf(const char* path, uint32_t gib, uint32_t version) {
    std::ofstream out(path, std::ios::binary);
    u32(out,0x46554747);u32(out,version);u64(out,gib+1);u64(out,2);
    str(out,"tokenizer.ggml.tokens");u32(out,9);u32(out,8);u64(out,2);str(out,"alpha");str(out,"beta");
    str(out,"test.array");u32(out,9);u32(out,4);u64(out,2);u32(out,7);u32(out,9);
    for(uint32_t i=0;i<=gib;++i) {
        str(out,i==gib ? "weight" : "padding"+std::to_string(i));
        u32(out,1);u64(out,i==gib ? 1 : 268435456);u32(out,0);u64(out,uint64_t(i)*(1ULL<<30));
    }
    const uint64_t data=(uint64_t(out.tellp())+31)/32*32;
    const uint64_t offset=data+uint64_t(gib)*(1ULL<<30);
    out.seekp(static_cast<std::streamoff>(offset));u32(out,0x3f800000); // float32 1.0
    out.close(); return offset;
}

static uint64_t sparse_safetensors(const char* path, uint64_t gap) {
    const std::string header = "{\"padding\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(gap) +
      "],\"data_offsets\":[0," + std::to_string(gap) + "]},\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
      std::to_string(gap) + "," + std::to_string(gap+4) + "]}}";
    std::ofstream out(path, std::ios::binary); u64(out,header.size());out.write(header.data(),header.size());
    const uint64_t offset=8+header.size()+gap;
    out.seekp(static_cast<std::streamoff>(offset));u32(out,0x3f800000);out.close();return offset;
}
static void gguf_shard(const char* path, int index, const std::string& name, int total=2) {
    std::ofstream out(path,std::ios::binary);
    u32(out,0x46554747);u32(out,3);u64(out,1);u64(out,3);
    str(out,"split.no");u32(out,2);out.put(char(index));out.put(0);
    str(out,"split.count");u32(out,2);out.put(2);out.put(0);
    str(out,"split.tensors.count");u32(out,5);u32(out,total);
    str(out,name);u32(out,1);u64(out,1);u32(out,0);u64(out,0);
    out.seekp((static_cast<std::streamoff>(out.tellp())+31)/32*32);u32(out,0x3f800000);
}
static void model_io_checks() {
    check(sdc_model_io_capabilities()==3,"file capability bits");
    for(uint64_t gib : {0ULL,2ULL,4ULL,8ULL,20ULL}) {
        const char* file="sparse.safetensors";
        const uint64_t offset=sparse_safetensors(file,gib<<30);
        check(sdc_test_safetensors_offset(file)==offset,"safetensors 64-bit offsets");
        check(sdc_test_safetensors_value(file)==0x3f800000,"safetensors sparse payload");
        std::remove(file);std::printf("safetensors offset=%llu: passed\n",(unsigned long long)offset);
    }
    const char* first="parts-00001-of-00002.gguf";const char* second="parts-00002-of-00002.gguf";
    gguf_shard(first,0,"weight1");gguf_shard(second,1,"weight2");
    check(sdc_test_model_tensor_count(first)==2,"complete GGUF group");
    check(sdc_test_model_tensor_count(second)==2,"GGUF group from second shard");
    gguf_shard(second,1,"weight1");check(sdc_test_model_tensor_count(first)==0,"duplicate GGUF tensors");
    gguf_shard(second,0,"weight2");check(sdc_test_model_tensor_count(first)==0,"wrong GGUF shard number");
    gguf_shard(second,1,"weight2",3);check(sdc_test_model_tensor_count(first)==0,"inconsistent GGUF tensor total");
    std::remove(second);check(sdc_test_model_tensor_count(first)==0,"missing GGUF shard");std::remove(first);
    for(const std::string& header : {
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}",
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,4]}}",
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-1,4]}}"}) {
        const char* file="bad.safetensors";std::ofstream out(file,std::ios::binary);u64(out,header.size());out.write(header.data(),header.size());u32(out,0);out.close();
        check(sdc_test_safetensors_offset(file)==UINT64_MAX,"invalid safetensors header");std::remove(file);
    }
    for(const std::string& reference : {"../escape.safetensors","/escape.safetensors","sub/../escape.safetensors","C:/escape.safetensors"}) {
        std::ofstream out("model.safetensors.index.json");out<<"{\"weight_map\":{\"weight\":\""<<reference<<"\"}}";out.close();
        check(sdc_test_model_tensor_count("model.safetensors.index.json")==0,"unsafe index path");std::remove("model.safetensors.index.json");
    }
    sparse_safetensors("tensor.safetensors",0);
    {std::ofstream out("model.safetensors.index.json");out<<"{\"weight_map\":{\"weight\":\"tensor.safetensors\"}}";}
    check(sdc_test_model_tensor_count("model.safetensors.index.json")==1,"safetensors index reads local shard");
    {std::ofstream out("loop.index.json");out<<"{\"weight_map\":{\"weight\":\"model.safetensors.index.json\"}}";}
    check(sdc_test_model_tensor_count("loop.index.json")==0,"index recursion must be rejected");
    std::remove("loop.index.json");
    sparse_safetensors("duplicate.safetensors",0);
    {std::ofstream out("model.safetensors.index.json");out<<"{\"weight_map\":{\"weight\":\"tensor.safetensors\",\"other\":\"duplicate.safetensors\"}}";}
    check(sdc_test_model_tensor_count("model.safetensors.index.json")==0,"duplicate shard tensors must be rejected");
    std::remove("duplicate.safetensors");
    std::remove("tensor.safetensors");std::remove("model.safetensors.index.json");
}

int main() {
    try {
        check(sdc_abi_version()==2,"ABI version");
        check(sdc_test_graph_walk()==1,"deep graph traversal, selection, ordering and shared operand counts");
        sdc_sd_set_log_callback(uint64_t(uintptr_t(&log_callback)),17);
        check(sdc_test_qwen_timestep("CPU")==1,"Qwen BF16 timestep activation placement and numerical parity");
        sdc_sd_set_log_callback(0,0);
        check(placement_message.find("bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0") != std::string::npos,
              "CPU workspace reports the original BF16 weight matmul");
        check(placement_message.find("webgpu_weights=0 host_weights=1 other_weights=0 webgpu_cpu_bf16=0 webgpu_cpu_bf16_use_bytes=0") != std::string::npos,
              "CPU placement is not reported as GPU weight transfer");
        check(placement_message.find("probe") == std::string::npos, "Placement diagnostics omit tensor names");
        check(sdc_test_bf16_weights("CPU")==1,"CPU loading retains BF16, F16 and quantized parameter types");
        check(sdc_test_conv3d_bias("CPU")==1,"3D convolution bias placement and numerical parity");
        check(sdc_test_webgpu_performance("CPU")==1,"Bounded graph arithmetic and cache ownership");
        check(placement_ops_messages.empty(),"Per-operation diagnostics default off");
        sdc_sd_set_log_callback(uint64_t(uintptr_t(&log_callback)),17);
        sd_set_graph_diagnostics(true);
        check(sdc_test_qwen_timestep("CPU")==1,"Opt-in operation placement graph");
        check(placement_ops_messages.size()==1 && placement_ops_messages[0].find("MUL_MAT:2")!=std::string::npos &&
              placement_ops_messages[0].find("probe")==std::string::npos,"Bounded, name-free compute placement histogram");
        sd_set_graph_diagnostics(false);
        check(sdc_test_qwen_timestep("CPU")==1 && placement_ops_messages.size()==1,"Disabled operation diagnostics emit nothing");
        sdc_sd_set_log_callback(0,0);
        model_io_checks();
        sd_ctx_params_t context{};sdc_sd_ctx_params_init(uint64_t(uintptr_t(&context)));
        sd_ctx_params_t reference{};sd_ctx_params_init(&reference);
        check(context.n_threads==reference.n_threads && context.enable_mmap==reference.enable_mmap &&
            context.auto_fit==reference.auto_fit && context.conditioning_cache_size==reference.conditioning_cache_size,
            "binding must retain upstream defaults");
        check(context.webgpu_bf16_type==SD_TYPE_F32,"BF16 WebGPU policy defaults to exact F32 widening");
        sd_img_gen_params_t image{};sdc_sd_img_gen_params_init(uint64_t(uintptr_t(&image)));
        sd_img_gen_params_t defaults{};sd_img_gen_params_init(&defaults);
        check(image.width==defaults.width && image.height==defaults.height && image.batch_count==defaults.batch_count &&
            image.sample_params.sample_method==defaults.sample_params.sample_method,"no generation policy in core");
        { // Exercise the generated video ABI, not a TypeScript field mock.
            sd_vid_gen_params_t video{}, reference_video{};
            sdc_sd_vid_gen_params_init(uint64_t(uintptr_t(&video)));
            sd_vid_gen_params_init(&reference_video);
            check(video.video_frames==reference_video.video_frames && video.fps==reference_video.fps &&
                  video.seed==reference_video.seed && video.vae_tiling_params.rel_size_x==reference_video.vae_tiling_params.rel_size_x &&
                  video.sample_params.sample_method==reference_video.sample_params.sample_method,"video bindings preserve native defaults");
            sd_image_t sentinel_image{};sd_audio_t sentinel_audio{};
            sd_image_t* frames=&sentinel_image;sd_audio_t* audio=&sentinel_audio;int count=73,fps=24;
            check(sdc_generate_video(0,uint64_t(uintptr_t(&video)),uint64_t(uintptr_t(&frames)),
                                     uint64_t(uintptr_t(&count)),uint64_t(uintptr_t(&audio)),uint64_t(uintptr_t(&fps)))==0,
                  "null-context video generation fails without inference");
            check(frames==nullptr && audio==nullptr && count==0 && fps==0,"video failure zeroes all distinct output slots");
        }
        for(uint32_t gib : {0,2,4,8}) {
            const char* path="large-offset-test.gguf";
            const uint64_t offset=sparse_gguf(path,gib,gib==2?2:3);
            check(sdc_test_gguf_offset(path)==offset,"64-bit GGUF metadata offset");
            check(sdc_test_gguf_value(path)==0x3f800000,"C++ random-access tensor read");
            std::remove(path);
            std::printf("unsplit GGUF sparse fixture: offset=%llu, tensor bytes read=4\n",(unsigned long long)offset);
        }
        { // Wrong version / truncated metadata must not fall through to another parser.
            const char* path="invalid-test.gguf";std::ofstream out(path,std::ios::binary);
            u32(out,0x46554747);u32(out,99);u64(out,1);u64(out,0);out.close();
            check(sdc_test_gguf_offset(path)==UINT64_MAX,"invalid GGUF rejection");std::remove(path);
        }
        for (int kind=0; kind<7; ++kind) {
            // 0: higher-rank extension; others: malformed metadata or tensor table.
            const char* path="metadata-test.gguf";std::ofstream out(path,std::ios::binary);
            u32(out,0x46554747);u32(out,3);u64(out,kind==6?2:1);u64(out,kind==2||kind==3?1:0);
            if(kind==2) { str(out,"general.alignment");u32(out,4);u32(out,3); }
            if(kind==3) { str(out,"bad.array");u32(out,9);u32(out,4);u64(out,UINT64_MAX); }
            for(int tensor=0;tensor<(kind==6?2:1);++tensor) {
                str(out,"weight"); const int rank=kind==0?5:1;u32(out,rank);
                for(int dim=0;dim<rank;++dim)u64(out,kind==4?UINT64_MAX:1);
                u32(out,kind==5?UINT32_MAX:0);u64(out,kind==1?1024:0);
            }
            const uint64_t data=(uint64_t(out.tellp())+31)/32*32;
            out.seekp(static_cast<std::streamoff>(data));u32(out,0x3f800000);out.close();
            if(kind==0) check(sdc_test_gguf_value(path)==0x3f800000,"extended GGUF rank flattening");
            else check(sdc_test_gguf_offset(path)==UINT64_MAX,"malformed GGUF must be rejected");
            std::remove(path);
        }
        // Exercise both small buffered skips and large 64-bit metadata seeks.
        // The metadata array is sparse; no multi-GiB allocation/read is needed.
        for (uint64_t gap : {0ULL, 1ULL, 4096ULL, 4097ULL, 8192ULL, (1ULL << 32) + 16}) {
            const char* path = "cursor-metadata-test.gguf";
            std::ofstream out(path, std::ios::binary);
            u32(out, 0x46554747); u32(out, 3); u64(out, 1); u64(out, 1);
            str(out, "test.skipped.bytes"); u32(out, 9); u32(out, 0); u64(out, gap);
            out.seekp(static_cast<std::streamoff>(uint64_t(out.tellp()) + gap));
            str(out, "weight"); u32(out, 1); u64(out, 1); u32(out, 0); u64(out, 0);
            const uint64_t offset = (uint64_t(out.tellp()) + 31) / 32 * 32;
            out.seekp(static_cast<std::streamoff>(offset)); u32(out, 0x3f800000); out.close();
            check(sdc_test_gguf_offset(path) == offset, "logical cursor after skipped metadata");
            check(sdc_test_gguf_value(path) == 0x3f800000, "payload after skipped metadata");
            std::printf("metadata-gap=%llu offset=%llu: passed\n", (unsigned long long)gap, (unsigned long long)offset);
            std::remove(path);
        }
        sdc_sd_set_log_callback(uint64_t(uintptr_t(&log_callback)),17);
        sdc_sd_set_progress_callback(uint64_t(uintptr_t(&progress_callback)),19);
        sdc_test_callbacks();
        check(logs_seen==1 && progress_seen==1,"native callback registration/data");
        sdc_sd_set_log_callback(0,0);sdc_sd_set_progress_callback(0,0);
        sdc_test_callbacks();check(logs_seen==1 && progress_seen==1,"native callback removal");
        for (int batch : {1, 2}) {
            ggml_init_params params{64 * 1024 * 1024, nullptr, false};
            auto ctx = ggml_init(params);
            auto input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 11, 64, batch);
            auto values = static_cast<float*>(input->data);
            for (int64_t i = 0; i < ggml_nelements(input); ++i) values[i] = std::sin(float(i) * 0.17f);
            auto reference = ggml_group_norm(ctx, input, 32, 1e-6f);
            auto actual = ggml_ext_group_norm_32(ctx, input);
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, reference);ggml_build_forward_expand(graph, actual);
            check(ggml_graph_compute_with_ctx(ctx, graph, 1)==GGML_STATUS_SUCCESS,"graph compute");
            float maximum = 0;
            for (int64_t i = 0; i < ggml_nelements(input); ++i)
                maximum = std::max(maximum, std::abs(static_cast<float*>(reference->data)[i] - static_cast<float*>(actual->data)[i]));
            std::printf("group-norm batch=%d max-absolute-error=%.9g\n", batch, maximum);
            check(std::isfinite(maximum) && maximum<=1e-4f,"normalization parity");ggml_free(ctx);
        }
        std::puts("Native public binding, large-file metadata/read and normalization tests passed; no model inference.");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
