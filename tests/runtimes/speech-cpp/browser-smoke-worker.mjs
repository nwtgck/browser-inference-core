import { checkSpeechStartup } from './startup-contract.mjs';

self.onmessage = async ({ data: { profile, pointerBytes } }) => {
  try {
    if (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function') {
      throw new Error('Speech startup requires JSPI support in the Worker');
    }
    const { attachCore, schema } = await import('/host/index.mjs');
    const { default: createSpeechCore } = await import(`/profiles/${profile}/browser/core.mjs`);
    const module = await createSpeechCore();
    const core = attachCore(module, schema, { suspension: 'jspi' });
    const evidence = await checkSpeechStartup({ core, pointerBytes });
    self.postMessage({ result: { profile, variant: 'browser', passed: true, ...evidence } });
  } catch (error) {
    self.postMessage({ error: String(error.stack || error) });
  }
};
