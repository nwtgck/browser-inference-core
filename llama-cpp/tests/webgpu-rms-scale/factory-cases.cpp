}
;
static std::string contents(const char * path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
int main(int argc, char ** argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("output, row_norm source, scale source, scale WG required");
        const unsigned scale_wg = std::stoul(argv[4]);
        if (!scale_wg || scale_wg > 1024)
            throw std::runtime_error("invalid scale WG");
        for (bool reverse : {false, true}) {
            Library library;
            std::set<unsigned> ids;
            for (bool order : {false, true}) {
                const bool inplace = order != reverse;
                const auto a = library.get_rms_scale_pipeline(inplace), b = library.get_rms_scale_pipeline(inplace);
                if (a.id != b.id || a.code.find('#') != std::string::npos ||
                    a.code.find("@workgroup_size(128)") == std::string::npos)
                    throw std::runtime_error("actual factory/cache mismatch");
                ids.insert(a.id);
                std::ofstream(std::string(argv[1]) + "/fused-" + std::to_string(inplace) + ".wgsl") << a.code;
            }
            if (ids.size() != 2)
                throw std::runtime_error("inplace cache collision");
        }
        pre_wgsl::Preprocessor preprocessor;
        for (bool inplace : {false, true}) {
            std::vector<std::string> defines{"RMS_NORM", "SRC_F32", "DST_F32", "WG_SIZE=128"};
            if (inplace)
                defines.push_back("INPLACE");
            std::ofstream(std::string(argv[1]) + (inplace ? "/rms-inplace.wgsl" : "/rms.wgsl"))
                << preprocessor.preprocess(contents(argv[2]), defines);
            defines = {"WG_SIZE=" + std::to_string(scale_wg)};
            if (inplace)
                defines.push_back("INPLACE");
            std::ofstream(std::string(argv[1]) + (inplace ? "/scale-inplace.wgsl" : "/scale.wgsl"))
                << preprocessor.preprocess(contents(argv[3]), defines);
        }
        std::cout << "2 actual RMS/scale variants in both cache orders and 4 upstream references emitted\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
