import { READ_ENV_BICORE_LLAMA_ABI_SCHEMA, READ_ENV_BICORE_SD_ABI_SCHEMA } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import test from 'node:test';
import { decodeAbiSchema, type AbiField, type AbiSchema, type AbiType } from '#dto/shared';
import { readJson } from '#tools/shared/json';
import { abiSchemaFingerprint } from '#tools/abi/schema';

const llamaSchema = READ_ENV_BICORE_LLAMA_ABI_SCHEMA();
const imageSchema = READ_ENV_BICORE_SD_ABI_SCHEMA();

function verifyRecordGraph({ schema }: { schema: AbiSchema }): void {
  const names = new Set(schema.records.map(record => record.name));
  function inspect({ type }: { type: AbiType }): void {
    switch (type.kind) {
    case 'record': assert.ok(type.record && names.has(type.record), `Unresolved record: ${type.record}`);
      break;
    case 'array':
      assert.ok(type.element, 'Array elements retain their ABI/public kinds');
      inspect({ type: type.element.abiType });
      break;
    case 'void': case 'pointer': case 'integer': case 'float': break;
    default: {
      const exhaustive: never = type.kind;
      throw new Error(String(exhaustive));
    }
    }
  }
  for (const record of schema.records) {
    for (const field of record.fields) {
      inspect({ type: field.abiType });
    }
  }
  for (const fn of schema.functions) {
    inspect({ type: fn.abiType });
    for (const parameter of fn.parameters) {
      inspect({ type: parameter.abiType });
    }
  }
}

test('actual llama metadata preserves current multimodal, batch, callback and record contracts', { skip: llamaSchema === undefined }, async () => {
  assert.ok(llamaSchema);
  const schema = await readJson({ path: llamaSchema, decode: decodeAbiSchema });
  assert.equal(schema.schemaSha256, abiSchemaFingerprint({ schema }));
  verifyRecordGraph({ schema });
  const functions = new Map(schema.functions.map(fn => [fn.name, fn]));
  for (const name of ['ggml_backend_tensor_copy_n', 'ggml_backend_tensor_get_n', 'ggml_backend_tensor_set_n', 'ggml_flash_attn_ext_rows', 'ggml_rope_set_truncate', 'mtmd_bitmap_is_voice', 'mtmd_gen_audio_get_voice', 'mtmd_gen_audio_voice_name', 'mtmd_gen_audio_read_tensor', 'mtmd_gen_state_set_ref', 'mtmd_helper_resampler_init', 'mtmd_helper_resampler_process', 'mtmd_helper_resampler_free', 'llama_batch_ext_add_embd_row', 'llama_batch_ext_set_embd_state_row', 'llama_get_ctx_other', 'llama_get_embeddings_layer_inp', 'llama_get_embeddings_nextn', 'llama_get_embeddings_nextn_ith', 'llama_get_embeddings_nextn_rows', 'llama_model_dflash_selector_top_k', 'llama_model_target_layer_ids', 'llama_model_target_layer_ids_n', 'llama_set_ctx_state_src', 'llama_set_embeddings_layer_inp', 'llama_set_embeddings_layer_inp_dev', 'llama_set_embeddings_nextn', 'llama_set_embeddings_nextn_dev', 'llama_set_embeddings_nextn_rows', 'llama_set_nextn_layer_offset', 'llama_supports_embd_row', 'mtmd_helper_support_video', 'mtmd_helper_model_can_chat']) {
    const fn = functions.get(name);
    assert.ok(fn, name);
    assert.ok(schema.exports.includes(fn.export), name);
  }
  const initializer = functions.get('mtmd_init_from_file_with_vision_bf16_storage');
  assert.ok(initializer);
  assert.equal(initializer.returnKind, 'pointer');
  assert.equal(initializer.returnType, 'struct mtmd_context *');
  assert.equal(initializer.parameters.length, 4);
  assert.equal(initializer.parameters[2]?.kind, 'record');
  assert.equal(initializer.parameters[3]?.abiType.bits, 32);
  assert.equal(initializer.parameters[3]?.abiType.signed, true);
  const getters = new Set(schema.callbackMetadata.getters.map(getter => getter.name));
  for (const name of ['ggml_op_desc', 'ggml_backend_buffer_name', 'ggml_backend_buffer_is_host', 'ggml_backend_dev_count', 'ggml_backend_dev_get', 'ggml_backend_dev_name', 'ggml_backend_dev_supports_op']) {
    assert.ok(getters.has(name), name);
  }
  for (const name of ['bicore_file_backed_rows_version', 'bicore_model_file_backed_shard']) {
    const fn = functions.get(name);
    assert.ok(fn, name);
    assert.equal(fn.export, '_' + name);
    assert.equal(fn.abiType.bits, 32);
    assert.ok(schema.callbackMetadata.getters.some(getter => getter.name === name));
  }
  assert.equal(functions.get('bicore_file_backed_rows_version')?.parameters.length, 0);
  assert.equal(functions.get('bicore_model_file_backed_shard')?.parameters[0]?.kind, 'pointer');
  for (const getter of schema.callbackMetadata.getters) {
    assert.equal(getter.export, '_' + getter.name);
    assert.ok(!schema.suspendingExports.includes(getter.export.slice(1)));
  }
  assert.ok(schema.excluded.some(entry => entry.name === 'ggml_threadpool_get_n_threads' && entry.reason.length > 0));
  for (const name of ['GGML_TYPE_F8_E4M3', 'GGML_FTYPE_MOSTLY_F8_E4M3', 'LLAMA_FTYPE_MOSTLY_F8_E4M3', 'MTMD_VIDEO_TEMPLATE_QWEN3VL', 'MTMD_GEN_AUDIO_TYPE_CHATTERBOX', 'MTMD_GEN_AUDIO_TYPE_DRAMA3']) {
    assert.ok(schema.constants.includes(name), name);
  }
  for (const name of ['GGUF_VERSION', 'GGUF_DEFAULT_ALIGNMENT']) {
    assert.ok(schema.constants.includes(name), name);
  }
  const records = new Map(schema.records.map(record => [record.name, record]));
  assert.ok(records.get('llama_context_params')?.fields.some(field => field.name === 'moe_cache_cpu_share' && field.cType === 'float'));
  assert.ok(records.get('llama_context_params')?.fields.some(field => field.name === 'moe_cache_size'));
  assert.ok(records.get('mtmd_gen_audio_info')?.fields.some(field => field.name === 'n_voices'));
  assert.ok(records.get('mtmd_helper_gen_audio_inp')?.fields.some(field => field.name === 'speaker_ref'));
  assert.ok(records.get('mtmd_helper_gen_audio_inp')?.fields.some(field => field.name === 'temp' && field.cType === 'float'));
  assert.ok(schema.extensions?.['model-graph-v1']);
  const declarations = schema.extensionDeclarations;
  assert.ok(declarations);
  assert.deepEqual(declarations.map(value => value.extensionId).sort(), ['model-graph-v1', 'webgpu-q8-diagnostic-v1']);
  const diagnostic = declarations.find(value => value.extensionId === 'webgpu-q8-diagnostic-v1');
  assert.ok(diagnostic);
  assert.equal(diagnostic.functions.find(fn => fn.name.endsWith('_version'))?.parameters[0]?.kind, 'u64');
  const run = diagnostic.functions.find(fn => fn.name.endsWith('_run_v1'));
  assert.ok(run);
  assert.deepEqual(run.parameters.map(parameter => parameter.kind), ['u64', 'u64', 'u64', 'u64', 'unsigned']);
  assert.ok(!schema.records.some(record => record.name.startsWith('ggml_webgpu_q8_diagnostic_')));
  assert.ok(!schema.constants.some(name => name.startsWith('GGML_WEBGPU_Q8_DIAGNOSTIC_')));
  assert.ok(!schema.callbacks.some(callback => callback.name.startsWith('ggml_backend_webgpu_q8_diagnostic_')));
  const implemented = schema.extensions['webgpu-q8-diagnostic-v1'];
  if (implemented) {
    assert.deepEqual(implemented.functions, diagnostic.functions);
    for (const fn of implemented.functions) {
      assert.ok(schema.exports.includes(fn.export));
    }
    assert.ok(schema.suspendingExports.includes(run.export.slice(1)));
  } else {
    assert.ok(!schema.exports.some(name => name.startsWith('_lcb_webgpu_q8_diagnostic_')));
    assert.ok(!schema.suspendingExports.some(name => name.startsWith('lcb_webgpu_q8_diagnostic_')));
  }
});

test('actual image metadata preserves generation results, progress width and public GGML records', { skip: imageSchema === undefined }, async () => {
  assert.ok(imageSchema);
  const schema = await readJson({ path: imageSchema, decode: decodeAbiSchema });
  assert.equal(schema.schemaSha256, abiSchemaFingerprint({ schema }));
  verifyRecordGraph({ schema });
  const functions = new Map(schema.functions.map(fn => [fn.name, fn]));
  for (const name of ['generate_image', 'generate_video', 'free_sd_image_generation_result', 'free_sd_video_generation_result', 'sd_browser_bf16_retention_version', 'sd_ctx_get_runtime_info', 'sd_ctx_get_memory_info', 'sd_set_generation_progress_callback', 'ggml_new_tensor_2d', 'ggml_graph_plan', 'ggml_backend_cpu_set_abort_callback', 'gguf_init_from_callback']) {
    assert.ok(functions.has(name), name);
  }
  assert.equal(functions.get('generate_image')?.parameters.length, 3);
  assert.equal(functions.get('generate_image')?.returnKind, 'boolean');
  assert.equal(functions.get('sd_get_preview_info')?.returnKind, 'record');
  assert.equal(functions.get('sd_get_preview_info')?.export, '_bicore_sd_get_preview_info');
  const records = new Map(schema.records.map(record => [record.name, record]));
  for (const [name, kind] of [['items', 'pointer'], ['count', 'signed']] as const) {
    assert.equal(records.get('sd_image_generation_result_t')?.fields.find(field => field.name === name)?.kind, kind, name);
  }
  const progress = records.get('sd_generation_progress_t');
  assert.ok(progress);
  for (const name of ['completed', 'total']) {
    const field: AbiField | undefined = progress.fields.find(value => value.name === name);
    assert.equal(field?.kind, 'i64');
    assert.equal(field?.size, 8);
    assert.equal(field?.abiType.signed, true);
  }
  assert.equal(records.get('ggml_bf16_t')?.fields[0]?.name, 'bits');
  assert.ok(records.has('ggml_cplan'));
  assert.ok(records.has('gguf_init_params'));
  assert.ok(records.has('ggml_tallocr'));
  for (const [name, kind] of [['mem_size', 'u64'], ['mem_buffer', 'pointer'], ['no_alloc', 'boolean']] as const) {
    assert.equal(records.get('ggml_init_params')?.fields.find(field => field.name === name)?.kind, kind, name);
  }
  for (const [name, kind] of [['image', 'record'], ['generation_info', 'pointer']] as const) {
    assert.equal(records.get('sd_image_result_t')?.fields.find(field => field.name === name)?.kind, kind, name);
  }
  assert.ok(schema.constants.includes('SD_TYPE_BF16'));
  for (const name of ['GGUF_VERSION', 'GGUF_DEFAULT_ALIGNMENT']) {
    assert.ok(schema.constants.includes(name), name);
  }
  const contextFields = new Set(records.get('sd_ctx_params_t')?.fields.map(field => field.name));
  for (const name of ['batched_cfg', 'flash_attn_mode', 'conv_direct_mode']) {
    assert.ok(contextFields.has(name), name);
  }
  const getters = new Set(schema.callbackMetadata.getters.map(getter => getter.name));
  for (const name of ['sd_cancel_generation', 'sd_set_preview_callback']) {
    assert.ok(getters.has(name), name);
  }
  for (const getter of schema.callbackMetadata.getters) {
    assert.ok(!schema.suspendingExports.includes(getter.export.slice(1)));
  }
});
