import assert from 'node:assert/strict';
import test from 'node:test';
import { READ_ENV_BIC_METRICS_MODE } from '#tools/shared/node-env';
import { stubEnvironment } from './node-env.test-support.js';

test('named environment readers stay lazy and preserve empty and absent values', () => {
  const original = READ_ENV_BIC_METRICS_MODE();
  {
    using first = stubEnvironment({ values: { BIC_METRICS_MODE: 'first' } });
    assert.equal(READ_ENV_BIC_METRICS_MODE(), 'first');
    {
      using second = stubEnvironment({ values: { BIC_METRICS_MODE: 'second' } });
      assert.equal(READ_ENV_BIC_METRICS_MODE(), 'second');
    }
    assert.equal(READ_ENV_BIC_METRICS_MODE(), 'first');
    {
      using empty = stubEnvironment({ values: { BIC_METRICS_MODE: '' } });
      assert.equal(READ_ENV_BIC_METRICS_MODE(), '');
    }
    {
      using absent = stubEnvironment({ values: { BIC_METRICS_MODE: undefined } });
      assert.equal(READ_ENV_BIC_METRICS_MODE(), undefined);
    }
  }
  assert.equal(READ_ENV_BIC_METRICS_MODE(), original);
});

test('environment test stubs restore keys after an operation throws', () => {
  const original = READ_ENV_BIC_METRICS_MODE();
  const failure = new Error('operation failed');
  assert.throws(() => {
    using environment = stubEnvironment({ values: { BIC_METRICS_MODE: 'temporary' } });
    assert.equal(READ_ENV_BIC_METRICS_MODE(), 'temporary');
    throw failure;
  }, (error) => error === failure);
  assert.equal(READ_ENV_BIC_METRICS_MODE(), original);
});
