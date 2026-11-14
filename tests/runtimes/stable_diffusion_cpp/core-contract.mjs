// Shared by the real browser Worker and the Node real-Wasm contract probe.
// Keep runtime dependencies inside each function for explicit Worker transfer.

/** @param {{ module: import('browser-inference-core/stable-diffusion-cpp/host').CoreModule }} input */
export function assertModuleSurface({ module }) {
  if (module._bicore_abi_version() !== 3 || module._bicore_model_io_capabilities() !== 7 || ('_sdb_load' in module && module._sdb_load !== undefined)) {
    throw Error('Wrong public surface');
  }
}

/** @param {{ core: import('browser-inference-core/stable-diffusion-cpp/host').Core }} input */
export async function assertPublicGguf({ core }) {
  // These were public macro constants before the ABI transport change. Their
  // values remain compiler-derived, separately from the chosen fixture version.
  core.constant('GGUF_VERSION');
  core.constant('GGUF_DEFAULT_ALIGNMENT');
  if (!core.readUtf8(await core.api.ggml_version()) ||
      core.readUtf8(await core.api.ggml_type_name(core.constant('GGML_TYPE_F32'))) !== 'f32' ||
      await core.api.ggml_type_size(core.constant('GGML_TYPE_F32')) !== 4n) {
    throw Error('Public GGML type/version query failed');
  }
  const module = core.module;
  const bytes = core.alloc(24), params = core.allocRecord('gguf_init_params');
  // The fixture's GGUF v3 header is file-format input, not an ABI enum query.
  const version = 3;
  try {
    module.HEAPU8.fill(0, Number(bytes), Number(bytes) + 24);
    const header = new DataView(module.HEAPU8.buffer, Number(bytes), 24);
    header.setUint32(0, 0x46554747, true);
    header.setUint32(4, version, true);
    core.setField('gguf_init_params', params, 'no_alloc', 1);
    const metadata = await core.api.gguf_init_from_buffer(bytes, 24n, params);
    if (metadata === 0n) {
      throw Error('Public GGUF buffer constructor failed');
    }
    try {
      if (await core.api.gguf_get_version(metadata) !== version ||
          await core.api.gguf_get_n_kv(metadata) !== 0n || await core.api.gguf_get_n_tensors(metadata) !== 0n) {
        throw Error('Public GGUF metadata query failed');
      }
    } finally {
      await core.api.gguf_free(metadata);
    }
  } finally {
    core.free(params);
    core.free(bytes);
  }
}
