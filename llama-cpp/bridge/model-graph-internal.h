#pragma once
#include "common/chat.h"
#include <cstdint>
#include <string>

// Only the Embind owner needs this C++ entry point. No function pointer is
// imported from a model file or exposed to the declaration language.
common_chat_templates_ptr lcb_model_graph_chat_templates(uint64_t vocabulary_owner,
                                                         const std::string & override_source);

// Implemented by the reviewed common/chat.cpp overlay. Keep the original chat
// header unique: copying a pragma-once header would break transitive includes.
common_chat_templates_ptr
common_chat_templates_init_from_vocab(const struct llama_vocab * vocab, std::string default_template_src,
                                      const std::string & template_tool_use_src, const std::string & template_tts_src,
                                      bool has_explicit_template, const std::string & bos_token_override = "",
                                      const std::string & eos_token_override = "");
