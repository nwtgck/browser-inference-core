static std::vector<std::unique_ptr<Event>> allocated_events;
static bool deny_prefetch_event = false;
static std::vector<std::string> prefetch_trace;
static ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t) {
    static ggml_backend_buffer_type type{false};
    return &type;
}
static bool ggml_backend_supports_buft(ggml_backend_t, ggml_backend_buffer_type_t) {
    return true;
}
static ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t) {
    if (deny_prefetch_event) {
        return nullptr;
    }
    allocated_events.push_back(std::make_unique<Event>());
    return allocated_events.back().get();
}
static void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    event->backend = backend;
    prefetch_trace.push_back("record");
}
static void actual_constructor_discovery(ggml_backend_sched_t sched, ggml_backend_t * backends,
                                         ggml_backend_buffer_type_t * bufts, int n_backends) {
    // @ACTUAL_CONSTRUCTOR_DISCOVERY@
}
// @ACTUAL_PREFETCH@
