// Execute only the new source-bound terminal boundaries with CPU API counters.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

static unsigned apis = 0, unmaps = 0, submits = 0, host_writes = 0, queries = 0, checks = 0;
struct Fatal {};
#define GGML_ABORT(...)                                                                                                \
    throw Fatal {}
namespace wgpu {
enum class BufferUsage { None, CopySrc };
static BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return BufferUsage(unsigned(a) & unsigned(b));
}
} // namespace wgpu
struct Control {
    bool lost = false, terminal = false;
    bool healthy() const {
        ++queries;
        return !lost && !terminal;
    }
};
struct Generation {
    Control control;
    std::uint64_t generation = 1;
};
struct Buffer {
    explicit operator bool() const {
        return true;
    }
    wgpu::BufferUsage GetUsage() const {
        ++apis;
        return wgpu::BufferUsage::CopySrc;
    }
    void Unmap() {
        ++unmaps;
    }
};
struct Queue {
    void Submit(int, const int *) {
        ++apis;
        ++submits;
    }
};
struct Global {
    std::shared_ptr<Generation> optional_device = std::make_shared<Generation>();
    Buffer get_tensor_staging_buf;
    Queue queue;
};
using webgpu_global_context = std::shared_ptr<Global>;
struct Source {
    webgpu_global_context global_ctx;
    Buffer buffer;
};
struct Encoder {
    webgpu_global_context global;
    bool lose;
    int Finish() {
        ++apis;
        if (lose)
            global->optional_device->control.lost = true;
        return 1;
    }
};
#include GLOBAL_HELPERS

static bool validate(Source * source, webgpu_global_context global = {}) {
#include FIRST_API_GUARD
    return true;
}
static void submit(webgpu_global_context global, bool lose) {
    Encoder encoder{global, lose};
#include SUBMIT_GUARD
}
static void after_map(webgpu_global_context global) {
#include POST_MAP_GUARD
    ++host_writes;
    global->get_tensor_staging_buf.Unmap();
}
static bool zero_requests(const void * requests, std::size_t count) {
    constexpr std::size_t max_requests = 256;
#include EMPTY_REQUEST_GUARD
    return false;
}
static bool empty_global(webgpu_global_context global) {
#include EMPTY_GLOBAL_GUARD
    return false;
}
static void before_resources(webgpu_global_context global) {
#include LOOP_GUARD
    ++apis;
}
static void require(bool okay) {
    ++checks;
    if (!okay)
        throw std::runtime_error("Provider contract failed");
}
static bool fatal(auto body) {
    try {
        body();
    } catch (const Fatal &) {
        return true;
    }
    return false;
}
static void reset() {
    apis = unmaps = submits = host_writes = queries = 0;
}

int main() {
    try {
        reset();
        require(zero_requests(nullptr, 0) && apis == 0 && queries == 0);
        require(empty_global({}) && apis == 0 && queries == 0);
        for (bool loss : {false, true}) {
            auto global = std::make_shared<Global>();
            global->optional_device->control.lost = loss;
            global->optional_device->control.terminal = !loss;
            Source source{global, {}};
            reset();
            require(fatal([&] { validate(&source); }) && apis == 0 && queries == 1);
            reset();
            require(fatal([&] { before_resources(global); }) && apis == 0 && queries == 1);
            reset();
            require(fatal([&] { after_map(global); }) && unmaps == 1 && host_writes == 0 && apis == 0);
        }
        auto global = std::make_shared<Global>();
        Source source{global, {}};
        reset();
        require(validate(&source) && apis == 1);
        reset();
        require(!validate(&source, std::make_shared<Global>()) && apis == 0 && queries == 0);
        reset();
        require(fatal([&] { submit(global, true); }) && apis == 1 && submits == 0);
        global->optional_device->control.lost = false;
        reset();
        submit(global, false);
        require(apis == 2 && submits == 1);
        reset();
        after_map(global);
        require(host_writes == 1 && unmaps == 1);
        std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"gpu\":false}\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
