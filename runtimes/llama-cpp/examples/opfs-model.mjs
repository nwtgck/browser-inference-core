/** Dedicated Worker example; importing this file does not open or download a model. */
import { createCore, mountReadOnlyFile } from '../dist/package/host/index.mjs';

export async function loadStoredModel(fileHandle, { profile = 'cpu-wasm32', baseURL } = {}) {
  const storage = await fileHandle.createSyncAccessHandle();
  let core, mounted, model = 0n;
  try {
    core = await createCore({ profile, ...(baseURL ? { baseURL } : {}) });
    await core.api.llama_backend_init();
    mounted = mountReadOnlyFile(core, '/models/model.gguf', {
      size: storage.getSize(),
      read(destination, offset) { return storage.read(destination, { at: offset }); },
    });
    const params = core.allocRecord('llama_model_params');
    const path = core.utf8(mounted.path);
    try {
      await core.api.llama_model_default_params(params);
      core.setField('llama_model_params', params, 'n_gpu_layers', profile.startsWith('webgpu') ? -1 : 0);
      core.setField('llama_model_params', params, 'load_mode', core.constant('LLAMA_LOAD_MODE_NONE'));
      core.setField('llama_model_params', params, 'lazy_mode', core.constant('LLAMA_LAZY_MODE_OFF'));
      model = await core.api.llama_model_load_from_file(path, params);
      if (!model) throw new Error('Model loading failed');
    } finally { core.free(path); core.free(params); }
    let closed = false;
    return {
      core, model,
      // The caller must first release all contexts and objects that borrow this model.
      async close() {
        if (closed) return;
        closed = true;
        try {
          await core.api.llama_model_free(model);
          mounted.remove();
          await core.api.llama_backend_free();
        } finally { storage.close(); }
      },
    };
  } catch (error) {
    // An unexpected native abort can poison the module. Discard its owning Worker.
    storage.close();
    throw error;
  }
}
