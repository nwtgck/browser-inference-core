// Actual inline upload owners with controlled public ggml APIs. These doubles
// do not establish a real device's queue implementation or physical memory use.
#include "mtmd-retained-upload.h"
#include <cassert>
#include <string>
#include <vector>

struct ggml_backend_reg {
    const char * name;
};
struct ggml_backend_device {
    ggml_backend_reg_t reg;
};
struct ggml_backend_buffer_type {
    ggml_backend_dev_t device;
};
struct ggml_backend {
    ggml_backend_dev_t device;
};
static ggml_backend_reg registry{"WebGPU"};
static ggml_backend_device device{&registry}, other{&registry};
static ggml_backend_buffer_type buffer_type{&device};
static ggml_backend backend{&device};
static struct {
    bool fail = false, mismatch = false, live = false;
    size_t pending = 0;
    int initializes = 0, frees = 0, waits = 0;
    std::vector<std::string> events;
} state;

extern "C" ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    return buft->device;
}
extern "C" ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t dev) {
    return dev->reg;
}
extern "C" const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    return reg->name;
}
extern "C" ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t b) {
    return b->device;
}
extern "C" ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t dev, const char * params) {
    assert(dev == &device && params == nullptr && !state.live);
    ++state.initializes;
    state.events.push_back("initialize");
    if (state.fail)
        return nullptr;
    backend.device = state.mismatch ? &other : dev;
    state.live = true;
    return &backend;
}
extern "C" void ggml_backend_free(ggml_backend_t b) {
    assert(b == &backend && state.live && state.pending == 0);
    state.live = false;
    ++state.frees;
    state.events.push_back("free");
}
extern "C" void ggml_backend_synchronize(ggml_backend_t b) {
    assert(b == &backend && state.live);
    state.pending = 0;
    ++state.waits;
    state.events.push_back("wait");
}
template <class F> static void rejected(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::runtime_error &) {
        caught = true;
    }
    assert(caught);
}
static void reset() {
    assert(!state.live && state.pending == 0);
    state = {};
    registry.name = "WebGPU";
    buffer_type.device = &device;
}
struct weight_owner {
    ~weight_owner() {
        assert(!state.live && state.pending == 0);
        state.events.push_back("weights-free");
    }
};
int main() {
    { lcb_mtmd_upload_backend owner; }
    assert(state.initializes == 0 && state.frees == 0);
    {
        lcb_mtmd_upload_backend owner;
        rejected([&] { owner.acquire(nullptr); });
        registry.name = "CPU";
        rejected([&] { owner.acquire(&buffer_type); });
        registry.name = "WebGPU";
        buffer_type.device = nullptr;
        rejected([&] { owner.acquire(&buffer_type); });
        assert(state.initializes == 0 && state.frees == 0);
    }
    reset();
    state.fail = true;
    {
        lcb_mtmd_upload_backend owner;
        rejected([&] { owner.acquire(&buffer_type); });
    }
    assert(state.initializes == 1 && state.frees == 0);
    reset();
    {
        lcb_mtmd_upload_backend owner;
        state.mismatch = true;
        rejected([&] { owner.acquire(&buffer_type); });
        assert(state.frees == 1 && !state.live);
        state.mismatch = false;
        owner.acquire(&buffer_type);
        rejected([&] { owner.acquire(&buffer_type); });
        assert(state.initializes == 2);
    }
    assert(state.frees == 2 && state.waits == 0);
    for (bool fail_write : {false, true}) {
        reset();
        bool caught = false;
        try {
            weight_owner weights;
            lcb_mtmd_upload_backend owner;
            lcb_mtmd_upload_queue queue(owner.acquire(&buffer_type));
            queue.reserve(7); // Charge before the possibly partial submission.
            state.pending += 7;
            if (fail_write)
                throw std::runtime_error("partial write or cancellation");
            queue.drain();
            assert(state.waits == 1);
        } catch (const std::runtime_error &) {
            caught = true;
        }
        assert(caught == fail_write);
        assert(state.events == std::vector<std::string>({"initialize", "wait", "free", "weights-free"}));
    }
    reset();
    {
        lcb_mtmd_upload_backend owner;
        lcb_mtmd_upload_queue queue(owner.acquire(&buffer_type));
        assert(!queue.reserve(0));
        rejected([&] { queue.reserve(SIZE_MAX); });
    }
    assert(state.waits == 0 && state.frees == 1);
    return 0;
}
