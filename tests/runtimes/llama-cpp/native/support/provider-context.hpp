#pragma once

// Owned setup shared by current complete-provider native fixtures.
static webgpu_context recording_context() {
    auto ctx = std::make_shared<webgpu_context_struct>();
    ctx->global_ctx = std::make_shared<webgpu_global_context_struct>();
    auto & global = *ctx->global_ctx;
    global.instance = wgpu::Instance::Acquire(new WGPUInstanceImpl);
    global.device = wgpu::Device::Acquire(new WGPUDeviceImpl);
    global.queue = global.device.GetQueue();
    global.optional_device = std::make_shared<ggml_webgpu_optional::DeviceGeneration>();
    ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(global.device, global.instance, global.optional_device);
    ctx->param_arena.init(global.device, 128, 8, 256);
    ctx->active_command_encoder = global.device.CreateCommandEncoder();
#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_set =
        wgpu::QuerySet::Acquire(new WGPUQuerySetImpl{1, false, WEBGPU_MAX_PROFILE_QUERY_COUNT});
#endif
    return ctx;
}
