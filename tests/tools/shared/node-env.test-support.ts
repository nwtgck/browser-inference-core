/** Test-only ambient mutation; callers receive cleanup, never an environment getter. */
export function stubEnvironment({ values }: { values: Readonly<Record<string, string | undefined>> }) {
  const previous = new Map<string, string | undefined>();
  const assign = ({ key, value }: { key: string; value: string | undefined }): void => {
    if (value === undefined) {
      // eslint-disable-next-line local-rules-node-env/node-env-policy -- Dedicated test stub restores absent keys without exposing ambient values.
      delete process.env[key];
    } else {
      // eslint-disable-next-line local-rules-node-env/node-env-policy -- Dedicated test stub is the only test environment writer.
      process.env[key] = value;
    }
  };
  const restore = (): void => {
    for (const [key, value] of previous) {
      assign({ key, value });
    }
  };
  try {
    for (const [key, value] of Object.entries(values)) {
      // eslint-disable-next-line local-rules-node-env/node-env-policy -- Save only explicitly stubbed keys for exact test cleanup; no raw map escapes.
      previous.set(key, process.env[key]);
      assign({ key, value });
    }
  } catch (error) {
    restore();
    throw error;
  }
  return { [Symbol.dispose]: restore };
}
