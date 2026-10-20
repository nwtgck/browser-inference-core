"""Compile the actual prepared factory with only pipeline creation recorded."""
from pathlib import Path
import subprocess

def emit(upstream: Path, prepared: Path, work: Path) -> Path:
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    start = text.index('    webgpu_pipeline get_moe_sum_pipeline(')
    stop = text.index('\n    }', start) + len('\n    }')
    function = text[start:stop]
    code = r'''#include "pre_wgsl.hpp"
#include "ggml-webgpu-moe-sum-shaders.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
struct webgpu_pipeline { unsigned id; std::string code, name; };
static unsigned made = 0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & name) {
    return {++made, code, name};
}
class Library {
    int device = 0;
    pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<int, webgpu_pipeline> moe_sum_pipelines;
public:
''' + function + r'''
};
int main(int argc, char ** argv) {
    if (argc != 2) throw std::runtime_error("output directory required");
    Library library;
    for (uint32_t experts : {3u, 4u, 8u, 10u, 16u, 32u}) {
        auto a = library.get_moe_sum_pipeline(experts);
        auto b = library.get_moe_sum_pipeline(experts);
        if (a.id != b.id || a.name != "moe_expert_sum_" + std::to_string(experts) ||
            a.code.find("N_EXPERTS") != std::string::npos) throw std::runtime_error("factory/cache mismatch");
        std::ofstream out(std::string(argv[1]) + "/moe-" + std::to_string(experts) + ".wgsl");
        out << a.code;
        if (!out) throw std::runtime_error("shader export failed");
    }
    if (made != 6) throw std::runtime_error("cache key mismatch");
    std::cout << "6 actual production factory variants and cache hits passed\n";
}
'''
    work.mkdir(parents=True, exist_ok=True)
    (work / 'factory.cpp').write_text(code)
    exe = work / 'factory'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(prepared), '-isystem', str(upstream / 'ggml/src/ggml-webgpu'),
                    str(work / 'factory.cpp'), '-o', str(exe)], check=True)
    return exe
