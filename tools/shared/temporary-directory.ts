import { mkdtemp, realpath, rm } from 'node:fs/promises';

/** Own a newly created scratch directory until its lexical async scope ends. */
export async function temporaryDirectory({ prefix }: { prefix: string }) {
  const allocated = await mkdtemp(prefix);
  let root: string;
  try {
    root = await realpath(allocated);
  } catch (error) {
    try {
      await rm(allocated, { recursive: true, force: true });
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Temporary directory canonicalization and cleanup failed');
    }
    throw error;
  }
  return {
    root,
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
}
