#include "speech.h"
#include "wasm-probes.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void status_contract() {
    check(speech_api_version_major() == SPEECH_API_VERSION_MAJOR, "major version differs from the compiled header");
    check(speech_api_version_minor() == SPEECH_API_VERSION_MINOR, "minor version differs from the compiled header");
    check(speech_version() && std::strlen(speech_version()) > 0, "missing source version");
    check(std::strcmp(speech_status_name(SPEECH_CANCELLED), "cancelled") == 0, "cancellation status differs");
    check(std::strcmp(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "invalid_argument") == 0,
          "invalid-argument status differs");
}

void option_contract() {
    const size_t count = speech_option_count();
    check(count > static_cast<size_t>(SPEECH_OPT_SPEAK_FIRST), "the dialogue options are unavailable");
    for (size_t i = 0; i < count; i++) {
        const auto option = static_cast<speech_option>(i);
        const char * name = speech_option_name(option);
        check(name && *name, "an option has no public name");
        speech_option parsed = SPEECH_OPT_VOICE;
        check(speech_option_from_name(name, &parsed) == SPEECH_OK && parsed == option,
              "option names do not round-trip through the public API");
    }
    check(speech_option_type(SPEECH_OPT_SEED) == SPEECH_TYPE_INT, "seed lost its integer type");
    check(speech_option_type(SPEECH_OPT_SPEAK_FIRST) == SPEECH_TYPE_BOOL, "speak_first lost its boolean type");
    speech_option parsed = SPEECH_OPT_VOICE;
    check(speech_option_from_name("unknown_option", &parsed) == SPEECH_ERROR_INVALID_ARGUMENT,
          "an unknown option was accepted");
    check(speech_last_error() && *speech_last_error(), "option failure has no error message");
}

void ownership_and_error_contract() {
    speech_load_params * raw = nullptr;
    check(speech_load_params_new(&raw) == SPEECH_OK && raw, "load parameters were not allocated");
    const std::unique_ptr<speech_load_params, decltype(&speech_load_params_free)> params(raw, speech_load_params_free);
    check(speech_load_params_set_device(params.get(), "gpu") == SPEECH_OK, "GPU selection was not accepted");
    check(speech_load_params_set_threads(params.get(), 1) == SPEECH_OK, "single-thread selection was not accepted");
    check(speech_load_params_set_threads(params.get(), 0) == SPEECH_ERROR_INVALID_ARGUMENT,
          "invalid thread count was not rejected");
    check(speech_last_error_option() && std::strcmp(speech_last_error_option(), "threads") == 0,
          "thread failure lost its option name");

    auto * request = reinterpret_cast<speech_request *>(uintptr_t{1});
    check(speech_request_new(nullptr, &request) == SPEECH_ERROR_INVALID_ARGUMENT && request == nullptr,
          "a failed request allocation left an owned handle");
    auto * dialogue = reinterpret_cast<speech_dialogue *>(uintptr_t{1});
    const auto on_frame = [](const float *, size_t, const char *, void *) { return 0; };
    check(speech_dialogue_start(nullptr, 24000, on_frame, nullptr, &dialogue) == SPEECH_ERROR_INVALID_ARGUMENT &&
              dialogue == nullptr,
          "a failed dialogue start left an owned handle");
    check(speech_dialogue_push(nullptr, nullptr, 0) == SPEECH_ERROR_INVALID_ARGUMENT,
          "dialogue push accepted a missing handle");
    check(speech_dialogue_end(nullptr) == SPEECH_ERROR_INVALID_ARGUMENT, "dialogue end accepted a missing handle");
    speech_dialogue_free(nullptr);
    speech_request_free(nullptr);
    speech_model_free(nullptr);
    speech_request_cancel(nullptr);
}

void callback_contract() {
    struct Observation {
        int calls = 0;
        speech_log_level level = SPEECH_LOG_DEBUG;
        std::string text;
    } observed;
    const auto on_log = [](speech_log_level level, const char * text, void * data) {
        auto & observation = *static_cast<Observation *>(data);
        observation.calls++;
        observation.level = level;
        observation.text = text;
    };
    speech_log_set(on_log, &observed);
    spc_test_emit_log();
    speech_log_set(nullptr, nullptr);
    check(observed.calls == 1 && observed.level == SPEECH_LOG_INFO && observed.text == "speech callback probe\n",
          "the upstream log callback did not preserve its level, text and caller data");
}
} // namespace

int main() {
    try {
        status_contract();
        option_contract();
        ownership_and_error_contract();
        callback_contract();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
