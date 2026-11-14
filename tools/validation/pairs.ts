export function requireResultPairs<Result extends { profile: string; variant: string; passed: boolean }>({ profiles, variants, results }: { profiles: readonly string[]; variants: readonly string[]; results: readonly Result[] }): void {
  const expected = new Set<string>();
  for (const profile of profiles) {
    for (const variant of variants) {
      expected.add(`${profile}/${variant}`);
    }
  }
  if (results.length !== expected.size) {
    throw new Error('Missing or extra browser test results');
  }
  const seen = new Set<string>();
  for (const result of results) {
    const key = `${result.profile}/${result.variant}`;
    if (!expected.has(key) || seen.has(key) || result.passed !== true) {
      throw new Error('Duplicate, unknown or failed browser test result');
    }
    seen.add(key);
  }
}
