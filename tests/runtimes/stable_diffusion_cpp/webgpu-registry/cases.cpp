namespace {
void check(bool ok, const char * message) {
    if (!ok)
        throw std::runtime_error(message);
}
webgpu_global_context context(ggml_backend_dev_t d) {
    return static_cast<ggml_backend_webgpu_device_context *>(d->context)->webgpu_global_ctx;
}
void stable(ggml_backend_reg_t reg, ggml_backend_dev_t dev) {
    auto original = context(dev);
    wgpu::Buffer live_buffer{original->device.p};
    auto queue = original->queue;
    auto device_id = original->device.p->id, adapter_id = original->adapter.p->id,
         instance_id = original->instance.p->id;
    // A hypothetical second adapter lookup must not poison existing work.
    fixture::fail_adapter = true;
    for (int i = 0; i < 50; ++i) {
        check(ggml_backend_webgpu_reg() == reg, "registry address changed");
        check(ggml_backend_webgpu_reg_get_device_count(reg) == 1, "repeated registry reset count");
        auto next = ggml_backend_webgpu_reg_get_device(reg, 0);
        check(next == dev && context(next) == original, "device context identity changed");
        check(original->device.p->id == device_id && original->adapter.p->id == adapter_id &&
                  original->instance.p->id == instance_id,
              "handles replaced");
        check(live_buffer.owner == original->device.p && queue.p == original->queue.p, "old buffer/queue detached");
        check(original->memset_params_buf.owner == original->device.p, "shared buffer retained wrong device");
    }
    check(fixture::adapters == 1 && fixture::devices == 1 && fixture::queues == 1 && fixture::buffers == 1 &&
              fixture::memsets == 1,
          "repeated backend initialization");
    bool bad = false;
    try {
        (void)ggml_backend_webgpu_reg_get_device(reg, 99);
    } catch (const std::runtime_error &) {
        bad = true;
    }
    check(bad, "invalid device index accepted");
}
} // namespace
int main(int argc, char ** argv) {
    try {
        std::string mode = argc == 2 ? argv[1] : "normal";
        fixture::adapter_wait_error = mode == "adapter-wait-error";
        fixture::device_wait_error = mode == "device-wait-error";
        fixture::missing_adapter_callback = mode == "missing-adapter-callback";
        fixture::missing_device_callback = mode == "missing-device-callback";
        fixture::fail_instance = mode == "null-instance";
        fixture::fail_adapter = mode == "null-adapter";
        fixture::f16 = mode != "no-f16";
        fixture::subgroups = mode != "no-subgroups";
        fixture::implicit = mode != "no-implicit-sync";
        fixture::fail_device = mode == "device-failure";
        fixture::async_callback = mode == "concurrent" || mode == "held-buffers";
        bool unavailable = fixture::fail_instance || fixture::fail_adapter || !fixture::f16 ||
                           fixture::adapter_wait_error || fixture::missing_adapter_callback;
#ifndef __EMSCRIPTEN__
        unavailable = unavailable || !fixture::implicit;
#endif
        if (mode == "concurrent") {
            std::atomic<bool> start{false};
            std::mutex lock;
            std::string failure;
            std::vector<std::thread> threads;
            std::vector<ggml_backend_reg_t> regs(16);
            std::vector<ggml_backend_dev_t> devs(16);
            for (int t = 0; t < 16; ++t)
                threads.emplace_back([&, t] {
                    while (!start.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    try {
                        for (int i = 0; i < 16; ++i) {
                            regs[t] = ggml_backend_webgpu_reg();
                            check(ggml_backend_webgpu_reg_get_device_count(regs[t]) == 1,
                                  "concurrent registration count");
                            devs[t] = ggml_backend_webgpu_reg_get_device(regs[t], 0);
                            check(context(devs[t])->queue.p != nullptr, "device published before queue ready");
                        }
                    } catch (const std::exception & e) {
                        std::lock_guard<std::mutex> guard(lock);
                        failure = e.what();
                    }
                });
            start.store(true, std::memory_order_release);
            for (auto & t : threads)
                t.join();
            check(failure.empty(), failure.c_str());
            for (int t = 1; t < 16; ++t)
                check(regs[t] == regs[0] && devs[t] == devs[0], "concurrent duplicate device");
            stable(regs[0], devs[0]);
        } else {
            auto reg = ggml_backend_webgpu_reg();
            if (unavailable) {
                check(ggml_backend_webgpu_reg_get_device_count(reg) == 0, "unavailable adapter admitted");
                fixture::fail_instance = false;
                fixture::fail_adapter = false;
                fixture::f16 = true;
                fixture::implicit = true;
                for (int i = 0; i < 50; ++i)
                    check(ggml_backend_webgpu_reg() == reg && ggml_backend_webgpu_reg_get_device_count(reg) == 0,
                          "failed registry retried implicitly");
                check(fixture::instances == 1 && fixture::adapters == (mode == "null-instance" ? 0 : 1) &&
                          fixture::devices == 0,
                      "unexpected initialization after discovery failure");
                for (auto & callback : fixture::pending)
                    callback();
                fixture::pending.clear();
                auto ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
                check(ggml_backend_webgpu_reg_get_device_count(reg) == 0 &&
                          ctx->webgpu_global_ctx->adapter.p == nullptr,
                      "late adapter result published");
            } else if (mode == "device-failure" || fixture::device_wait_error || fixture::missing_device_callback) {
                check(ggml_backend_webgpu_reg_get_device_count(reg) == 1, "available adapter not registered");
                bool caught = false;
                try {
                    (void)ggml_backend_webgpu_reg_get_device(reg, 0);
                } catch (const std::runtime_error &) {
                    caught = true;
                }
                check(caught && fixture::devices == 1 && fixture::queues == 0 && fixture::buffers == 0,
                      "partial device was published or retried");
                for (auto & callback : fixture::pending)
                    callback();
                fixture::pending.clear();
                auto ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
                check(ctx->webgpu_global_ctx->device.p == nullptr, "late device result published");
            } else {
                if (mode == "reordered")
                    for (int i = 0; i < 50; ++i)
                        check(ggml_backend_webgpu_reg() == reg && ggml_backend_webgpu_reg_get_device_count(reg) == 1,
                              "discovery-only query changed count");
                auto dev = ggml_backend_webgpu_reg_get_device(reg, 0);
                auto c = context(dev);
                check(c->capabilities.supports_subgroups == fixture::subgroups, "capabilities changed");
                if (mode == "device-lost") {
                    c->device.p->lost = true;
                    c->device.p->on_lost(c->device, wgpu::DeviceLostReason::Unknown, "fixture loss");
                }
                stable(reg, dev);
                if (mode == "device-lost")
                    check(c->device.p->lost, "lost device silently replaced");
            }
        }
        check(fixture::instances == 1, "multiple Instances");
#ifdef __EMSCRIPTEN__
        const char * branch = "browser";
#else
        const char * branch = "native";
#endif
        std::cout << "{\"branch\":\"" << branch << "\",\"case\":\"" << mode << "\",\"adapters\":" << fixture::adapters
                  << ",\"devices\":" << fixture::devices << ",\"passed\":true,\"gpuExecuted\":false}\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
