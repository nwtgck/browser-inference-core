#include "ggml.h"
#include "pre_wgsl.hpp"
#include "ggml-wgsl-shaders.hpp"
#include "ggml-webgpu-ssm-conv-silu-shaders.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
template<class T> static void ggml_webgpu_hash_combine(size_t & seed,const T & value) { seed ^= std::hash<T>{}(value)+0x9e3779b9+(seed<<6)+(seed>>2); }
struct ggml_webgpu_shader_lib_context { ggml_tensor * dst=nullptr,* src1=nullptr; };
struct webgpu_pipeline { unsigned id; std::string code,name; std::shared_ptr<void> context; };
static unsigned made=0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int,const std::string & code,const std::string & name) { return {++made,code,name,{}}; }
#undef GGML_ABORT
#define GGML_ABORT(...) throw std::runtime_error("unsupported factory type")
// @ACTUAL_KEYS@
class Library {
    int device=0;
    pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<ggml_webgpu_ssm_conv_pipeline_key,webgpu_pipeline,ggml_webgpu_ssm_conv_pipeline_key_hash> ssm_conv_pipelines;
public:
