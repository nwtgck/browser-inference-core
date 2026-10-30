#include "common/chat.h"
#include "common/json-schema-to-grammar.h"
#include "common/reasoning-budget.h"

#include <emscripten/bind.h>
#include <cstdint>
#include <stdexcept>

namespace {
template <typename T> T * pointer(uint64_t address) {
    if (address > UINTPTR_MAX)
        throw std::range_error("pointer exceeds address space");
    return reinterpret_cast<T *>(static_cast<uintptr_t>(address));
}

template <typename T> uint64_t address(T * value) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
}

// The upstream template type is incomplete and has a custom deleter.
// This owner lets ordinary Embind delete() invoke that upstream deleter.
struct templates_owner {
    common_chat_templates_ptr value;

    templates_owner(uint64_t model, const std::string & source, const std::string & bos, const std::string & eos)
        : value(common_chat_templates_init(pointer<const llama_model>(model), source, bos, eos)) {}
};
// The upstream session owns its prompt/parser state. Inputs and templates are
// used only during construction; no JavaScript object's lifetime is borrowed.
struct session_owner {
    common_chat_session value;

    session_owner() = default;
    session_owner(const templates_owner & templates, uint64_t vocab, const common_chat_templates_inputs & inputs,
                  const common_chat_session_params & params)
        : value(templates.value.get(), pointer<const llama_vocab>(vocab), inputs, params) {}
};
} // namespace

// These are type/member registrations, not a second chat data model.
#define FIELD(type, member) .property(#member, &type::member)
#define VALUE(name) .value(#name, name)

EMSCRIPTEN_BINDINGS(llama_common_chat) {
    using namespace emscripten;

    // Select the full signature of every directly registered named callable,
    // including static functions and const/non-const methods. A future upstream
    // overload must not make deduction ambiguous or change our JavaScript API.
    // Keep these explicit even when the name currently has only one overload;
    // a removed/changed signature should fail compilation, not bind a substitute.
    // Member selectors also name the owning class: leaving ClassType deduced can
    // become ambiguous when upstream adds a same-name member function template.
    // Typed adapter lambdas below still handle pointer/ownership conversions.
    // Fields and typed constructors do not pass unresolved function addresses.

    enum_<common_chat_role>("common_chat_role") VALUE(COMMON_CHAT_ROLE_UNKNOWN) VALUE(COMMON_CHAT_ROLE_SYSTEM)
        VALUE(COMMON_CHAT_ROLE_ASSISTANT) VALUE(COMMON_CHAT_ROLE_USER) VALUE(COMMON_CHAT_ROLE_TOOL);
    enum_<common_chat_tool_choice>("common_chat_tool_choice") VALUE(COMMON_CHAT_TOOL_CHOICE_AUTO)
        VALUE(COMMON_CHAT_TOOL_CHOICE_REQUIRED) VALUE(COMMON_CHAT_TOOL_CHOICE_NONE);
    enum_<common_chat_format>("common_chat_format") VALUE(COMMON_CHAT_FORMAT_CONTENT_ONLY)
        VALUE(COMMON_CHAT_FORMAT_PEG_SIMPLE) VALUE(COMMON_CHAT_FORMAT_PEG_NATIVE) VALUE(COMMON_CHAT_FORMAT_PEG_GEMMA4)
            VALUE(COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) VALUE(COMMON_CHAT_FORMAT_COUNT);
    enum_<common_chat_continuation>("common_chat_continuation") VALUE(COMMON_CHAT_CONTINUATION_NONE)
        VALUE(COMMON_CHAT_CONTINUATION_AUTO) VALUE(COMMON_CHAT_CONTINUATION_REASONING)
            VALUE(COMMON_CHAT_CONTINUATION_CONTENT);
    enum_<common_reasoning_format>("common_reasoning_format") VALUE(COMMON_REASONING_FORMAT_NONE)
        VALUE(COMMON_REASONING_FORMAT_AUTO) VALUE(COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY)
            VALUE(COMMON_REASONING_FORMAT_DEEPSEEK);
    enum_<common_grammar_trigger_type>("common_grammar_trigger_type") VALUE(COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN)
        VALUE(COMMON_GRAMMAR_TRIGGER_TYPE_WORD) VALUE(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN)
            VALUE(COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL);
    enum_<common_reasoning_budget_state>("common_reasoning_budget_state") VALUE(REASONING_BUDGET_IDLE)
        VALUE(REASONING_BUDGET_COUNTING) VALUE(REASONING_BUDGET_FORCING) VALUE(REASONING_BUDGET_WAITING_UTF8)
            VALUE(REASONING_BUDGET_DONE);

    register_vector<std::string>("string_vector");
    register_vector<llama_token>("llama_tokens");
    register_vector<llama_tokens>("llama_token_sequences");
    register_vector<size_t>("size_vector");
    register_vector<common_chat_msg>("common_chat_messages");
    register_vector<common_chat_tool>("common_chat_tools");
    register_vector<common_chat_tool_call>("common_chat_tool_calls");
    register_vector<common_chat_msg_content_part>("common_chat_content_parts");
    register_vector<common_chat_prompt_part>("common_chat_prompt_parts");
    register_vector<common_chat_msg_diff>("common_chat_message_diffs");
    register_vector<common_grammar_trigger>("common_grammar_triggers");
    register_vector<common_chat_msg_delimiter>("common_chat_delimiter_vector");
    register_vector<common_chat_msg_span>("common_chat_span_vector");
    register_map<std::string, std::string>("string_map");
    register_map<std::string, bool>("bool_map");
    register_map<size_t, size_t>("size_map");

    class_<common_json>("common_json")
        .constructor<>()
        .class_function("parse", select_overload<common_json(const std::string &)>(&common_json::parse))
        .class_function("parse_no_throw",
                        select_overload<common_json(const std::string &)>(&common_json::parse_no_throw))
        .class_function("array", select_overload<common_json()>(&common_json::array))
        .class_function("object", select_overload<common_json()>(&common_json::object))
        .function("dump", select_overload<std::string(int) const, common_json>(&common_json::dump))
        .function("is_discarded", select_overload<bool() const, common_json>(&common_json::is_discarded))
        .function("is_null", select_overload<bool() const, common_json>(&common_json::is_null))
        .function("is_object", select_overload<bool() const, common_json>(&common_json::is_object))
        .function("is_array", select_overload<bool() const, common_json>(&common_json::is_array))
        .function("size", select_overload<size_t() const, common_json>(&common_json::size));

    using clock = std::chrono::system_clock;
    class_<clock::duration>("system_clock_duration")
        .constructor<>()
        .constructor<clock::duration::rep>()
        .function("count", select_overload<clock::duration::rep() const, clock::duration>(&clock::duration::count));
    class_<clock::time_point>("system_clock_time_point")
        .constructor<>()
        .constructor<clock::duration>()
        .function("time_since_epoch",
                  select_overload<clock::duration() const, clock::time_point>(&clock::time_point::time_since_epoch));
    function(
        "system_clock_period_num", +[]() -> int64_t { return clock::period::num; });
    function(
        "system_clock_period_den", +[]() -> int64_t { return clock::period::den; });
    function("system_clock_now", select_overload<clock::time_point()>(&clock::now));

    class_<common_chat_tool_call>("common_chat_tool_call")
        .constructor<>() FIELD(common_chat_tool_call, name) FIELD(common_chat_tool_call, arguments)
            FIELD(common_chat_tool_call, id);
    class_<common_chat_msg_content_part>("common_chat_msg_content_part")
        .constructor<>() FIELD(common_chat_msg_content_part, type) FIELD(common_chat_msg_content_part, text)
            FIELD(common_chat_msg_content_part, name);
    class_<common_chat_msg>("common_chat_msg")
        .constructor<>() FIELD(common_chat_msg, role) FIELD(common_chat_msg, content)
            FIELD(common_chat_msg, content_parts) FIELD(common_chat_msg, tool_calls)
                FIELD(common_chat_msg, reasoning_content) FIELD(common_chat_msg, tool_name)
                    FIELD(common_chat_msg, tool_call_id) FIELD(common_chat_msg, extra_json)
        .function("to_json_oaicompat",
                  select_overload<common_json(bool) const, common_chat_msg>(&common_chat_msg::to_json_oaicompat))
        .function("render_content", select_overload<std::string(const std::string &) const, common_chat_msg>(
                                        &common_chat_msg::render_content))
        .function("empty", select_overload<bool() const, common_chat_msg>(&common_chat_msg::empty))
        .function("contains_media", select_overload<bool() const, common_chat_msg>(&common_chat_msg::contains_media));
    class_<common_chat_tool>("common_chat_tool")
        .constructor<>() FIELD(common_chat_tool, name) FIELD(common_chat_tool, description)
            FIELD(common_chat_tool, parameters) FIELD(common_chat_tool, defer_loading);
    class_<common_chat_msg_diff>("common_chat_msg_diff")
        .constructor<>() FIELD(common_chat_msg_diff, reasoning_content_delta) FIELD(common_chat_msg_diff, content_delta)
            FIELD(common_chat_msg_diff, tool_call_index) FIELD(common_chat_msg_diff, tool_call_delta)
        .class_function(
            "compute_diffs",
            select_overload<std::vector<common_chat_msg_diff>(const common_chat_msg &, const common_chat_msg &)>(
                &common_chat_msg_diff::compute_diffs));
    function(
        "common_chat_no_tool_call_index", +[]() -> size_t { return std::string::npos; });
    class_<common_grammar_trigger>("common_grammar_trigger")
        .constructor<>() FIELD(common_grammar_trigger, type) FIELD(common_grammar_trigger, value)
            FIELD(common_grammar_trigger, token);
    class_<common_chat_msg_span>("common_chat_msg_span")
        .constructor<>() FIELD(common_chat_msg_span, role) FIELD(common_chat_msg_span, pos)
            FIELD(common_chat_msg_span, len)
        .function("valid", select_overload<bool() const, common_chat_msg_span>(&common_chat_msg_span::valid));
    class_<common_chat_msg_spans>("common_chat_msg_spans")
        .constructor<>() FIELD(common_chat_msg_spans, spans)
        .function("add", select_overload<void(common_chat_role, size_t, size_t), common_chat_msg_spans>(
                             &common_chat_msg_spans::add))
        .function("is_user_start",
                  select_overload<bool(int32_t) const, common_chat_msg_spans>(&common_chat_msg_spans::is_user_start))
        .function("last_user_message_pos", select_overload<int32_t() const, common_chat_msg_spans>(
                                               &common_chat_msg_spans::last_user_message_pos));
    class_<common_chat_msg_delimiter>("common_chat_msg_delimiter")
        .constructor<>() FIELD(common_chat_msg_delimiter, role) FIELD(common_chat_msg_delimiter, delimiter)
            FIELD(common_chat_msg_delimiter, tokens);
    class_<common_chat_msg_delimiters>("common_chat_msg_delimiters")
        .constructor<>() FIELD(common_chat_msg_delimiters, delimiters)
        .function("add", select_overload<void(common_chat_role, const std::string &), common_chat_msg_delimiters>(
                             &common_chat_msg_delimiters::add))
        .function(
            "tokenize", +[](common_chat_msg_delimiters & self,
                            uint64_t vocab) { self.tokenize(pointer<const llama_vocab>(vocab)); })
        .function("split",
                  select_overload<common_chat_msg_spans(const llama_tokens &, const std::map<size_t, size_t> &) const,
                                  common_chat_msg_delimiters>(&common_chat_msg_delimiters::split));
    class_<common_chat_templates_inputs>("common_chat_templates_inputs")
        .constructor<>() FIELD(common_chat_templates_inputs, messages) FIELD(common_chat_templates_inputs, grammar)
            FIELD(common_chat_templates_inputs, json_schema) FIELD(common_chat_templates_inputs, add_generation_prompt)
                FIELD(common_chat_templates_inputs, continue_final_message)
                    FIELD(common_chat_templates_inputs, use_jinja) FIELD(common_chat_templates_inputs, tools)
                        FIELD(common_chat_templates_inputs, tool_choice)
                            FIELD(common_chat_templates_inputs, parallel_tool_calls)
                                FIELD(common_chat_templates_inputs, reasoning_format)
                                    FIELD(common_chat_templates_inputs, enable_thinking)
                                        FIELD(common_chat_templates_inputs, now)
                                            FIELD(common_chat_templates_inputs, chat_template_kwargs)
                                                FIELD(common_chat_templates_inputs, add_bos)
                                                    FIELD(common_chat_templates_inputs, add_eos)
                                                        FIELD(common_chat_templates_inputs, force_pure_content);
    // Keep template/input segments and byte-aligned token identity. Flattening
    // these values at the boundary would undo upstream's special-token handling.
    class_<common_chat_prompt_part>("common_chat_prompt_part")
        .constructor<>() FIELD(common_chat_prompt_part, text) FIELD(common_chat_prompt_part, is_input);
    class_<common_chat_prompt>("common_chat_prompt")
        .constructor<>()
        .constructor<std::string>() FIELD(common_chat_prompt, parts)
        .function("str", select_overload<std::string() const, common_chat_prompt>(&common_chat_prompt::str))
        .function("append",
                  select_overload<void(const std::string &, bool), common_chat_prompt>(&common_chat_prompt::append))
        .function("replace_last", select_overload<void(const std::string &, const std::string &), common_chat_prompt>(
                                      &common_chat_prompt::replace_last));
    class_<common_chat_input>("common_chat_input")
        .constructor<>()
        .constructor<std::string>() FIELD(common_chat_input, text) FIELD(common_chat_input, tokens)
        .function("size", select_overload<size_t() const, common_chat_input>(&common_chat_input::size))
        .function("empty", select_overload<bool() const, common_chat_input>(&common_chat_input::empty))
        .function("append", select_overload<void(const std::string &, llama_token), common_chat_input>(
                                &common_chat_input::append))
        .function("append_input",
                  select_overload<void(const common_chat_input &), common_chat_input>(&common_chat_input::append))
        .function("prepend", select_overload<void(const std::string &), common_chat_input>(&common_chat_input::prepend))
        .function("prepend_input",
                  select_overload<void(const common_chat_input &), common_chat_input>(&common_chat_input::prepend))
        .function("truncate", select_overload<void(size_t), common_chat_input>(&common_chat_input::truncate))
        .function("substr", select_overload<common_chat_input(size_t, size_t) const, common_chat_input>(
                                &common_chat_input::substr));
    function(
        "common_chat_input_tokenize", +[](uint64_t vocab, const std::string & text) {
            return common_chat_input_tokenize(pointer<const llama_vocab>(vocab), text);
        });
    class_<common_chat_params>("common_chat_params")
        .constructor<>() FIELD(common_chat_params, format) FIELD(common_chat_params, prompt)
            FIELD(common_chat_params, grammar) FIELD(common_chat_params, grammar_lazy)
                FIELD(common_chat_params, generation_prompt) FIELD(common_chat_params, supports_thinking)
                    FIELD(common_chat_params, thinking_start_tag) FIELD(common_chat_params, thinking_end_tags)
                        FIELD(common_chat_params, grammar_triggers) FIELD(common_chat_params, preserved_tokens)
                            FIELD(common_chat_params, additional_stops) FIELD(common_chat_params, parser)
                                FIELD(common_chat_params, message_delimiters);
    class_<common_chat_parser_params>("common_chat_parser_params")
        .constructor<>()
        .constructor<const common_chat_params &>() FIELD(common_chat_parser_params, format)
            FIELD(common_chat_parser_params, generation_prompt) FIELD(common_chat_parser_params, debug)
                FIELD(common_chat_parser_params, parser);
    class_<common_chat_prompt_preset>("common_chat_prompt_preset")
        .constructor<>() FIELD(common_chat_prompt_preset, system) FIELD(common_chat_prompt_preset, user)
            FIELD(common_chat_prompt_preset, has_language);
    class_<common_peg_arena>("common_peg_arena")
        .constructor<>()
        .function("empty", select_overload<bool() const, common_peg_arena>(&common_peg_arena::empty))
        .function("size", select_overload<size_t() const, common_peg_arena>(&common_peg_arena::size));
    class_<common_chat_session_params>("common_chat_session_params")
        .constructor<>() FIELD(common_chat_session_params, echo) FIELD(common_chat_session_params, debug);
    class_<session_owner>("common_chat_session")
        .constructor<>()
        .constructor<const templates_owner &, uint64_t, const common_chat_templates_inputs &,
                     const common_chat_session_params &>()
        // Copy upstream reference returns: handles survive subsequent feed(),
        // finish() and session deletion, and cannot mutate the session's state.
        .function(
            "prompt", +[](const session_owner & self) { return self.value.prompt(); })
        .function(
            "format", +[](const session_owner & self) { return self.value.format(); })
        .function(
            "msg", +[](const session_owner & self) { return self.value.msg(); })
        .function(
            "parser", +[](const session_owner & self) { return self.value.parser(); })
        .function(
            "grammar", +[](const session_owner & self) { return self.value.grammar(); })
        .function(
            "generation_prompt", +[](const session_owner & self) { return self.value.generation_prompt(); })
        .function(
            "thinking_start_tag", +[](const session_owner & self) { return self.value.thinking_start_tag(); })
        .function(
            "thinking_end_tags", +[](const session_owner & self) { return self.value.thinking_end_tags(); })
        .function(
            "additional_stops", +[](const session_owner & self) { return self.value.additional_stops(); })
        .function(
            "message_delimiters", +[](const session_owner & self) { return self.value.message_delimiters(); })
        .function(
            "has_template", +[](const session_owner & self) { return self.value.has_template(); })
        .function(
            "feed", +[](session_owner & self, const common_chat_input & chunk) { return self.value.feed(chunk); })
        .function(
            "finish", +[](session_owner & self, const common_chat_input & chunk) { return self.value.finish(chunk); });

    class_<templates_owner>("common_chat_templates")
        .constructor<uint64_t, const std::string &, const std::string &, const std::string &>()
        .function(
            "apply",
            +[](const templates_owner & self, const common_chat_templates_inputs & inputs) {
                return common_chat_templates_apply(self.value.get(), inputs);
            })
        .function(
            "source",
            +[](const templates_owner & self, const std::string & variant) {
                return common_chat_templates_source(self.value.get(), variant);
            })
        .function(
            "was_explicit",
            +[](const templates_owner & self) { return common_chat_templates_was_explicit(self.value.get()); })
        .function(
            "get_caps",
            +[](const templates_owner & self) -> common_json {
                return select_overload<common_json(const common_chat_templates *)>(&common_chat_templates_get_caps)(
                    self.value.get());
            })
        .function(
            "support_enable_thinking",
            +[](const templates_owner & self) {
                return common_chat_templates_support_enable_thinking(self.value.get());
            })
        .function(
            "format_single",
            +[](const templates_owner & self, const std::vector<common_chat_msg> & past,
                const common_chat_msg & message, bool add_assistant, bool use_jinja) {
                return common_chat_format_single(self.value.get(), past, message, add_assistant, use_jinja);
            })
        .function(
            "format_example",
            +[](const templates_owner & self, bool use_jinja, const std::map<std::string, std::string> & kwargs) {
                return common_chat_format_example(self.value.get(), use_jinja, kwargs);
            })
        .function(
            "get_asr_prompt",
            +[](const templates_owner & self) { return common_chat_get_asr_prompt(self.value.get()); })
        .function(
            "get_asr_prompt_with_language",
            +[](const templates_owner & self, const std::string & language) {
                return common_chat_get_asr_prompt(self.value.get(), language);
            })
        .function(
            "get_tts_prompt", +[](const templates_owner & self, const std::string & text, const std::string & language,
                                  const std::string & voice) {
                return common_chat_get_tts_prompt(self.value.get(), text, language, voice);
            });

    function("common_chat_verify_template",
             select_overload<bool(const std::string &, bool)>(&common_chat_verify_template));
    function("common_chat_parse",
             select_overload<common_chat_msg(const common_chat_input &, bool, const common_chat_parser_params &)>(
                 &common_chat_parse));
    function("common_chat_peg_parse",
             select_overload<common_chat_msg(const common_peg_arena &, const common_chat_input &, bool,
                                             const common_chat_parser_params &)>(&common_chat_peg_parse));
    function("common_chat_msgs_parse_oaicompat",
             select_overload<std::vector<common_chat_msg>(const common_json &)>(&common_chat_msgs_parse_oaicompat));
    function(
        "common_chat_msgs_to_json_oaicompat",
        select_overload<common_json(const std::vector<common_chat_msg> &, bool)>(&common_chat_msgs_to_json_oaicompat));
    function("common_chat_tools_parse_oaicompat",
             select_overload<std::vector<common_chat_tool>(const common_json &)>(&common_chat_tools_parse_oaicompat));
    function("common_chat_tools_to_json_oaicompat",
             select_overload<common_json(const std::vector<common_chat_tool> &)>(&common_chat_tools_to_json_oaicompat));
    function("common_chat_continuation_parse",
             select_overload<common_chat_continuation(const common_json &)>(&common_chat_continuation_parse));
    function("common_chat_tool_choice_parse_oaicompat",
             select_overload<common_chat_tool_choice(const std::string &)>(&common_chat_tool_choice_parse_oaicompat));
    function("common_chat_role_from_string",
             select_overload<common_chat_role(const std::string &)>(&common_chat_role_from_string));
    function(
        "common_chat_role_to_string",
        +[](common_chat_role role) { return std::string(common_chat_role_to_string(role)); });
    function(
        "common_chat_format_name",
        +[](common_chat_format format) { return std::string(common_chat_format_name(format)); });
    function(
        "common_reasoning_format_name",
        +[](common_reasoning_format format) { return std::string(common_reasoning_format_name(format)); });
    function("common_reasoning_format_from_name",
             select_overload<common_reasoning_format(const std::string &)>(&common_reasoning_format_from_name));
    // Upstream also overloads this name for common_chat_schema_document (v0.4.1).
    // Keep the existing JSON + force_gbnf binding; an unqualified address is
    // ambiguous to Embind's function template once that overload is present.
    function("json_schema_to_grammar",
             select_overload<std::string(const common_json &, bool)>(&json_schema_to_grammar));
    function(
        "common_reasoning_budget_init",
        +[](uint64_t vocab, const std::vector<llama_tokens> & starts, const std::vector<llama_tokens> & ends,
            const llama_tokens & forced, int32_t budget, common_reasoning_budget_state initial) {
            return address(
                common_reasoning_budget_init(pointer<const llama_vocab>(vocab), starts, ends, forced, budget, initial));
        });
    function(
        "common_reasoning_budget_get_state",
        +[](uint64_t sampler) { return common_reasoning_budget_get_state(pointer<const llama_sampler>(sampler)); });
    // A const vector pointer cannot use register_vector's mutable class handle.
    // Copy the short marker so callers own a readable value with normal delete().
    function(
        "common_reasoning_budget_get_end_match_copy", +[](uint64_t sampler) -> llama_tokens {
            const auto * match = common_reasoning_budget_get_end_match(pointer<const llama_sampler>(sampler));
            return match ? *match : llama_tokens{};
        });
    function(
        "common_reasoning_budget_force",
        +[](uint64_t sampler) { return common_reasoning_budget_force(pointer<llama_sampler>(sampler)); });
}

#undef FIELD
#undef VALUE
