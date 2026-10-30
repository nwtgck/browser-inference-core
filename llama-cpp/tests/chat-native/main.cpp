// Native integration of the public APIs bound by chat-embind.cpp.
// This does not exercise Emscripten value marshalling or a trained model.
#include "chat.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

static int checks = 0;
static void require(bool condition, const char * label) {
    if (!condition)
        throw std::runtime_error(label);
    ++checks;
}
static std::string serialized(const common_chat_msg & msg) {
    return msg.to_json_oaicompat(false).dump(-1);
}
int main(int argc, char ** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("supply the pinned Qwen3 template path");
        std::ifstream stream(argv[1]);
        require(bool(stream), "open template");
        const std::string source{std::istreambuf_iterator<char>(stream), {}};
        common_chat_prompt prompt("<start>");
        prompt.append("<start>", true);
        prompt.replace_last("<start>", "<changed>");
        require(prompt.str() == "<changed><start>", "replacement preserves user text");
        require(prompt.parts.size() == 2 && prompt.parts[1].is_input, "segmented prompt retained");
        common_chat_input input("a");
        input.append(u8"あ", 17);
        require(input.text == u8"aあ" && input.size() == 4, "UTF-8 byte length");
        require(input.tokens == std::vector<llama_token>({-1, 17, -1, -1}), "UTF-8 token positions");
        input.append("", 18);
        require(input.size() == 4 && input.tokens.size() == 4, "empty token piece");
        auto suffix = input.substr(1, 3);
        require(suffix.text == u8"あ" && suffix.tokens[0] == 17, "token preserving slice");
        suffix.prepend("!");
        require(suffix.tokens == input.tokens, "plain prefix has no token identity");
        suffix.truncate(1);
        require(suffix.text == "!" && suffix.tokens == std::vector<llama_token>({-1}),
                "truncate keeps lengths aligned");
        suffix.append(input);
        common_chat_input prefix;
        prefix.append("prefix", 19);
        suffix.prepend(prefix);
        require(suffix.tokens.size() == suffix.text.size() && suffix.tokens[0] == 19,
                "chunk append and prepend retain token positions");

        common_chat_templates_inputs inputs;
        inputs.messages =
            common_chat_msgs_parse_oaicompat(common_json::parse(R"([{"role":"user","content":"Weather in Tokyo?"}])"));
        inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(
            R"([{"type":"function","function":{"name":"lookup","description":"Find weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])"));
        inputs.enable_thinking = false;
        auto templates = common_chat_templates_init(nullptr, source, "", "");
        auto applied = common_chat_templates_apply(templates.get(), inputs);
        require(applied.prompt.str().find("lookup") != std::string::npos, "tool prompt");
        require(applied.prompt.str().find("Weather in Tokyo?") != std::string::npos, "message prompt");
        require(!applied.parser.empty(), "native parser returned without serialization");
        const std::string generated =
            "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Tokyo\"}}\n</tool_call>";
        common_chat_parser_params params(applied);
        auto arena = applied.parser;
        auto parsed = common_chat_peg_parse(arena, common_chat_input(generated), false, params);
        require(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "lookup", "tool parser name");
        require(parsed.tool_calls[0].arguments.find("Tokyo") != std::string::npos, "tool parser arguments");
        params.parser = arena;
        require(serialized(common_chat_parse(common_chat_input(generated), false, params)) == serialized(parsed),
                "parser member parity");
        common_chat_peg_parse(arena, common_chat_input(generated.substr(0, 12)), true, params);
        require(serialized(common_chat_peg_parse(arena, common_chat_input(generated), false, params)) ==
                    serialized(parsed),
                "arena reusable after partial parsing");
        common_chat_session session(templates.get(), nullptr, inputs, {});
        require(session.has_template() && session.prompt().str() == applied.prompt.str(), "session template parity");
        templates.reset(); // The bridge does not retain the templates handle.
        const auto early = session.feed(common_chat_input(generated.substr(0, 12)));
        const auto early_json = serialized(early);
        const auto finished = session.finish(common_chat_input(generated.substr(12)));
        require(serialized(finished) == serialized(parsed), "session incremental tool parsing");
        require(serialized(early) == early_json, "copied message independent of session mutation");
        common_chat_msg snapshot;
        {
            common_chat_session plain;
            snapshot = plain.feed(common_chat_input("hel"));
            const auto full = plain.finish(common_chat_input("lo"));
            require(full.content == "hello", "plain session streaming");
            const auto diffs = common_chat_msg_diff::compute_diffs(snapshot, full);
            require(diffs.size() == 1 && diffs[0].content_delta == "lo", "incremental content diff");
        }
        require(snapshot.content == "hel", "snapshot survives session destruction");
        require(serialized(common_chat_parse(common_chat_input(generated), false, params)) == serialized(parsed),
                "copied arena survives templates destruction");
        std::cout << "chat-native: " << checks << " checks passed (no Wasm marshalling)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "chat-native: " << error.what() << '\n';
        return 1;
    }
}
