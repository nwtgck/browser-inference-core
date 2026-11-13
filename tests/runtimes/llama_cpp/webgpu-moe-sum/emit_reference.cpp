#include "pre_wgsl.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
int main(int argc, char ** argv) {
    if (argc != 3) {
        throw std::runtime_error("usage: emit_reference binary.wgsl output-directory");
    }
    std::ifstream input(argv[1]);
    if (!input) {
        throw std::runtime_error("missing binary.wgsl");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    pre_wgsl::Preprocessor preprocessor;
    for (const auto & mode : {"SRC_OVERLAP", "INPLACE"}) {
        std::ofstream output(std::string(argv[2]) + "/" + mode + ".wgsl");
        if (!output) {
            throw std::runtime_error("output path unavailable");
        }
        output << preprocessor.preprocess(contents.str(), {"TYPE_F32", "OP_ADD", "WG_SIZE=256", mode});
    }
}
