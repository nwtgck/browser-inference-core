import assert from 'node:assert/strict';
import test from 'node:test';

import { loadConfiguration, selectConfiguration, type Configuration } from '#runtimes/llama-cpp/tools/config';

test('runtime configuration keeps all five build profiles and both variants', async () => {
  const configuration = await loadConfiguration({ checkoutRoot: process.cwd() });
  const profiles = ['cpu-wasm32', 'webgpu-wasm32-asyncify', 'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi', 'webgpu-wasm64-jspi-gpu-profile'];
  for (const profile of profiles) {
    for (const variant of ['browser', 'test']) {
      const selected = selectConfiguration({ configuration, profile, variant });
      assert.equal(selected.profile.pthreads, false);
    }
  }
  assert.throws(() => selectConfiguration({ configuration, profile: '../cpu', variant: 'browser' }));
  assert.throws(() => selectConfiguration({ configuration, profile: 'missing', variant: 'browser' }));
  assert.equal('llamaCommit' in configuration.toolchain, false);
  const cpu = configuration.profiles['cpu-wasm32'];
  assert.ok(cpu);
  const extended: Configuration = { ...configuration, profiles: { ...configuration.profiles, future_mode: { ...cpu, pthreads: true, asyncify: true } } };
  assert.equal(selectConfiguration({ configuration: extended, profile: 'cpu-wasm32', variant: 'browser' }).profile, cpu);
});
