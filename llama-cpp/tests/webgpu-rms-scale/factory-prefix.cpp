#include "pre_wgsl.hpp"
#include "ggml-webgpu-rms-scale-shaders.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
struct webgpu_pipeline { unsigned id; std::string code, name; };
static unsigned made=0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int,const std::string & code,const std::string & name){return {++made,code,name};}
class Library {
    int device=0;
    pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<int,webgpu_pipeline> rms_scale_pipelines;
public:
