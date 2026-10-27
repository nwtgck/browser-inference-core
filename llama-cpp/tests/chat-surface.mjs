/** Real generated-module checks; independent of the example runtime. */
export async function checkChatSurface(native, template) {
  const check = (value, label) => { if (!value) throw new Error(label); };
  const objects = [];
  const own = value => { objects.push(value); return value; };
  const json = value => own(native.common_json.parse(JSON.stringify(value)));
  const toJS = value => JSON.parse(value.dump(-1));
  const input = text => own(new native.common_chat_input(text));
  try {
    // Exercise static and const-member bindings as well as the JSON grammar
    // free function below. Explicit native signatures must preserve the API.
    const emptyJSON = own(new native.common_json());
    const emptyArray = own(native.common_json.array());
    const emptyObject = own(native.common_json.object());
    check(emptyJSON.is_null(), 'Default JSON null changed');
    check(emptyArray.is_array() && Number(emptyArray.size()) === 0, 'JSON array factory changed');
    check(emptyObject.is_object() && Number(emptyObject.size()) === 0, 'JSON object factory changed');
    const textMessage = own(new native.common_chat_msg());
    check(textMessage.empty() && !textMessage.contains_media(), 'Default message predicates changed');
    textMessage.content = 'typed member';
    check(!textMessage.empty() && textMessage.render_content('|') === 'typed member', 'Message member binding changed');
    textMessage.role = 'user';
    textMessage.extra_json = '{"routing":"kept"}';
    const extraMessages = own(new native.common_chat_messages());
    extraMessages.push_back(textMessage);
    check(JSON.parse(own(extraMessages.get(0)).extra_json).routing === 'kept', 'Extra message metadata lost on native copy');
    const parsedExtra = own(native.common_chat_msgs_parse_oaicompat(json([{ role: 'user', content: 'hello', routing: 'parsed' }])));
    check(JSON.parse(own(parsedExtra.get(0)).extra_json).routing === 'parsed', 'Extra message metadata missing from parser');
    const reference = own(new native.common_chat_msg_content_part());
    reference.type = 'tool_reference'; reference.name = 'lookup';
    const referenceParts = own(new native.common_chat_content_parts());
    referenceParts.push_back(reference);
    const referenceMessage = own(new native.common_chat_msg());
    referenceMessage.role = 'user'; referenceMessage.content_parts = referenceParts;
    check(toJS(own(referenceMessage.to_json_oaicompat(false))).content[0].name === 'lookup', 'Tool reference name lost');
    const deferred = own(new native.common_chat_tool());
    check(deferred.defer_loading === false, 'Deferred tool default changed');
    deferred.name = 'later'; deferred.description = ''; deferred.parameters = '{}'; deferred.defer_loading = true;
    const deferredTools = own(new native.common_chat_tools());
    deferredTools.push_back(deferred);
    check(toJS(own(native.common_chat_tools_to_json_oaicompat(deferredTools)))[0].function.defer_loading === true, 'Deferred tool flag lost');
    const templates = own(new native.common_chat_templates(0n, template, '', ''));
    check(templates.was_explicit(), 'Template override lost');
    const asr = own(templates.get_asr_prompt());
    const localizedAsr = own(templates.get_asr_prompt_with_language('English'));
    check(typeof asr.has_language === 'boolean' && typeof localizedAsr.has_language === 'boolean', 'ASR language metadata missing');
    check(templates.get_tts_prompt('hello', 'English', '') === '', 'An absent TTS template must stay empty');
    const caps = own(templates.get_caps());
    check(Number(caps.size()) > 0, 'Missing template capabilities');
    const tools = [{ type: 'function', function: {
      name: 'lookup', description: 'Find weather',
      parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    } }];
    const messages = [{ role: 'user', content: 'Weather in Tokyo?' }];
    const inputs = own(new native.common_chat_templates_inputs());
    check(Number(own(inputs.messages).size()) === 0, 'Native empty message default changed');
    check(inputs.tool_choice === native.common_chat_tool_choice.COMMON_CHAT_TOOL_CHOICE_AUTO, 'Native tool choice default changed');
    inputs.messages = own(native.common_chat_msgs_parse_oaicompat(json(messages)));
    inputs.tools = own(native.common_chat_tools_parse_oaicompat(json(tools)));
    inputs.enable_thinking = false;
    const params = own(templates.apply(inputs));
    const prompt = own(params.prompt);
    check(prompt.str().includes('lookup') && prompt.str().includes('Weather in Tokyo?'), 'Jinja tools/messages missing');
    check(typeof params.format.value === 'number', 'Chat format must retain its native enum');
    check(typeof params.grammar === 'string' && typeof params.grammar_lazy === 'boolean', 'Grammar metadata missing');
    check(typeof params.generation_prompt === 'string' && typeof params.supports_thinking === 'boolean', 'Generation metadata missing');
    check(typeof params.thinking_start_tag === 'string', 'Reasoning metadata missing');
    for (const field of ['thinking_end_tags', 'grammar_triggers', 'preserved_tokens', 'additional_stops']) {
      check(Number(own(params[field]).size()) >= 0, `Missing native vector: ${field}`);
    }
    const delimiters = own(params.message_delimiters);
    check(Number(own(delimiters.delimiters).size()) >= 0, 'Message delimiters missing');
    const arena = own(params.parser);
    check(!arena.empty() && Number(arena.size()) > 0, 'Missing native tool parser');
    const parserParams = own(new native.common_chat_parser_params(params));
    const emptyArena = own(new native.common_peg_arena());
    check(emptyArena.empty() && Number(emptyArena.size()) === 0, 'Default arena predicates changed');
    const generated = '<tool_call>\n{"name":"lookup","arguments":{"city":"Tokyo"}}\n</tool_call>';
    const parsed = own(native.common_chat_peg_parse(arena, input(generated), false, parserParams));
    const parsedJSON = toJS(own(parsed.to_json_oaicompat(false)));
    check(parsedJSON.tool_calls?.[0]?.function.name === 'lookup', 'Tool name not parsed');
    check(JSON.parse(parsedJSON.tool_calls[0].function.arguments).city === 'Tokyo', 'Tool arguments not parsed');
    own(native.common_chat_peg_parse(arena, input(generated.slice(0, -5)), true, parserParams));
    // v0.7.0 removed serialization. Assignment copies the upstream arena into
    // the parser parameters; repeated calls reuse that member without JSON.
    parserParams.parser = arena;
    const fromParams = own(native.common_chat_parse(input(generated), false, parserParams));
    check(own(fromParams.to_json_oaicompat(false)).dump(-1) === own(parsed.to_json_oaicompat(false)).dump(-1), 'Native parser member changed result');
    const sessionParams = own(new native.common_chat_session_params());
    const session = own(new native.common_chat_session(templates, 0n, inputs, sessionParams));
    check(session.has_template() && own(session.prompt()).str() === prompt.str(), 'Session template prompt changed');
    const firstChunk = own(session.feed(input(generated.slice(0, 12))));
    const firstSnapshot = own(firstChunk.to_json_oaicompat(false)).dump(-1);
    const completed = own(session.finish(input(generated.slice(12))));
    check(own(completed.to_json_oaicompat(false)).dump(-1) === own(parsed.to_json_oaicompat(false)).dump(-1), 'Session tool parsing changed');
    check(own(firstChunk.to_json_oaicompat(false)).dump(-1) === firstSnapshot, 'Session result handle aliases mutable state');
    check(!own(session.parser()).empty(), 'Session parser missing');
    inputs.messages = own(native.common_chat_msgs_parse_oaicompat(json([...messages,
      { ...parsedJSON, tool_calls: parsedJSON.tool_calls.map(call => ({ ...call, id: 'caller-assigned' })) },
      { role: 'tool', content: 'Sunny', name: 'lookup', tool_call_id: 'caller-assigned' }])));
    check(own(own(templates.apply(inputs)).prompt).str().includes('<tool_response>\nSunny'), 'Tool result history missing');

    const plainParams = own(new native.common_chat_parser_params());
    const previous = own(native.common_chat_parse(input('hel'), true, plainParams));
    const current = own(native.common_chat_parse(input('hello'), false, plainParams));
    const diffs = own(native.common_chat_msg_diff.compute_diffs(previous, current));
    const diff = own(diffs.get(0));
    check(diff.content_delta === 'lo', 'Streaming text diff failed');
    check(diff.tool_call_index === native.common_chat_no_tool_call_index(), 'Native missing-index sentinel changed');
    // Inputs keep token positions per UTF-8 byte, not per JS UTF-16 code unit.
    const tokenInput = input('a');
    tokenInput.append('あ', 17);
    check(tokenInput.text === 'aあ' && Number(tokenInput.size()) === 4, 'Token input byte length changed');
    check(Array.from(own(tokenInput.tokens)).join(',') === '-1,17,-1,-1', 'Token byte alignment changed');
    const byteIndex = value => typeof tokenInput.size() === 'bigint' ? BigInt(value) : value;
    const suffix = own(tokenInput.substr(byteIndex(1), byteIndex(3)));
    check(suffix.text === 'あ' && own(suffix.tokens).get(0) === 17, 'Token input slice lost token identity');
    const segmented = own(new native.common_chat_prompt('<start>'));
    segmented.append('<start>', true);
    segmented.replace_last('<start>', '<changed>');
    check(segmented.str() === '<changed><start>', 'Replacement changed a user input segment');
    const segments = own(segmented.parts);
    check(Number(segments.size()) === 2 && own(segments.get(1)).is_input, 'Prompt segments flattened');
    const standalone = new native.common_chat_session();
    let snapshot;
    try {
      snapshot = own(standalone.feed(input('hel')));
      check(own(standalone.finish(input('lo'))).content === 'hello', 'Plain session stream failed');
    } finally { standalone.delete(); }
    check(snapshot.content === 'hel', 'Session deletion invalidated an owned result');
    // Keep the public JSON + force_gbnf signature even when upstream adds
    // schema-document overloads. Both calls use the same owned common_json.
    const schema = json({ type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] });
    const schemaBefore = schema.dump(-1);
    for (const forceGbnf of [false, true]) {
      const grammar = native.json_schema_to_grammar(schema, forceGbnf);
      // This runtime disables LLGuidance, so both modes produce GBNF.
      check(typeof grammar === 'string' && grammar.includes('root ::='), `JSON schema grammar missing (force_gbnf=${forceGbnf})`);
      check(schema.dump(-1) === schemaBefore, 'JSON schema conversion changed the input');
    }
    const invalid = own(native.common_json.parse_no_throw('{'));
    check(invalid.is_discarded(), 'Upstream nonthrowing JSON parse changed');

    const trigger = own(new native.common_grammar_trigger());
    trigger.type = native.common_grammar_trigger_type.COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
    check(trigger.type === native.common_grammar_trigger_type.COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, 'Grammar trigger enum changed');
    const delimiter = own(new native.common_chat_msg_delimiter());
    delimiter.role = native.common_chat_role.COMMON_CHAT_ROLE_TOOL;
    check(delimiter.role === native.common_chat_role.COMMON_CHAT_ROLE_TOOL, 'Delimiter role enum changed');
    const zero = own(new native.system_clock_duration(0n));
    inputs.now = own(new native.system_clock_time_point(zero));
    check(own(own(inputs.now).time_since_epoch()).count() === 0n, 'Native clock value changed');
    check(native.system_clock_period_num() > 0n && native.system_clock_period_den() > 0n, 'Native clock period missing');

    const starts = own(new native.llama_token_sequences());
    const ends = own(new native.llama_token_sequences());
    const start = own(new native.llama_tokens()); start.push_back(1);
    const end = own(new native.llama_tokens()); end.push_back(2);
    starts.push_back(start); ends.push_back(end);
    const budget = native.common_reasoning_budget_init(0n, starts, ends, end, 4,
      native.common_reasoning_budget_state.REASONING_BUDGET_IDLE);
    let endCopy;
    try {
      check(Number(own(native.common_reasoning_budget_get_end_match_copy(budget)).size()) === 0, 'Unexpected reasoning end match');
      await native._lcb_llama_sampler_accept(budget, 1);
      check(native.common_reasoning_budget_get_state(budget) === native.common_reasoning_budget_state.REASONING_BUDGET_COUNTING, 'Reasoning counting transition failed');
      await native._lcb_llama_sampler_accept(budget, 2);
      check(native.common_reasoning_budget_get_state(budget) === native.common_reasoning_budget_state.REASONING_BUDGET_DONE, 'Reasoning completion transition failed');
      endCopy = own(native.common_reasoning_budget_get_end_match_copy(budget));
      check(Number(endCopy.size()) === 1 && endCopy.get(0) === 2, 'Reasoning end copy is unreadable');
      check(Array.from(endCopy).join(',') === '2', 'Reasoning end copy is not iterable');
      const disposable = native.common_reasoning_budget_get_end_match_copy(budget);
      disposable.delete();
      check(native.common_reasoning_budget_get_state(budget) === native.common_reasoning_budget_state.REASONING_BUDGET_DONE, 'Deleting the owned copy changed sampler state');
      endCopy.set(0, 3);
      check(own(native.common_reasoning_budget_get_end_match_copy(budget)).get(0) === 2, 'Reasoning end copy aliases sampler state');
      await native._lcb_llama_sampler_reset(budget);
      check(Number(own(native.common_reasoning_budget_get_end_match_copy(budget)).size()) === 0, 'Reasoning reset retained the end match');
      check(endCopy.get(0) === 3, 'Reasoning reset invalidated the owned copy');
      await native._lcb_llama_sampler_accept(budget, 1);
      check(native.common_reasoning_budget_force(budget), 'Reasoning force failed');
      check(native.common_reasoning_budget_get_state(budget) === native.common_reasoning_budget_state.REASONING_BUDGET_FORCING, 'Reasoning force transition failed');
    } finally { await native._lcb_llama_sampler_free(budget); }
    check(endCopy.get(0) === 3, 'Sampler release invalidated the owned copy');

    // Bulk data stays in linear memory through the generated C ABI.
    const rgb = native._lcb_malloc(3n);
    check(rgb !== 0n, 'RGB allocation failed');
    let bitmap = 0n;
    try {
      native.HEAPU8.set([13, 29, 47], Number(rgb));
      bitmap = await native._lcb_mtmd_bitmap_init(1, 1, rgb);
      check(bitmap !== 0n && (await native._lcb_mtmd_bitmap_get_n_bytes(bitmap)) === 3n, 'RGB bitmap failed');
      check(native.HEAPU8[Number(await native._lcb_mtmd_bitmap_get_data(bitmap)) + 2] === 47, 'RGB pixels changed');
    } finally {
      if (bitmap) await native._lcb_mtmd_bitmap_free(bitmap);
      native._lcb_free(rgb);
    }
    const pcm = native._lcb_malloc(8n);
    check(pcm !== 0n, 'PCM allocation failed');
    let audio = 0n;
    try {
      const view = new DataView(native.HEAPU8.buffer, Number(pcm), 8);
      view.setFloat32(0, 0.25, true); view.setFloat32(4, -0.5, true);
      audio = await native._lcb_mtmd_bitmap_init_from_audio(2n, pcm);
      check(audio !== 0n && (await native._lcb_mtmd_bitmap_is_audio(audio)) === 1, 'PCM bitmap failed');
    } finally {
      if (audio) await native._lcb_mtmd_bitmap_free(audio);
      native._lcb_free(pcm);
    }
    check((await native._lcb_mtmd_helper_support_video(0n)) === 0, 'Unexpected subprocess video support');
    return { directGeneratedModule: true, nativeChatToolsAndHistory: true, reusedParser: true,
      nativeEnumsAndDefaults: true, reasoningSampler: true, multimodalBitmaps: true,
      trainedModelTools: false, multimodalModelInference: false };
  } finally {
    for (const object of objects.reverse()) object.delete();
  }
}
