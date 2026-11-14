import { chmod, lstat, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { regularFile } from '#tools/shared/files';

export type ManifestIo = { writeFile: typeof writeFile; rename: typeof rename };
export const manifestIo: ManifestIo = { writeFile, rename };

/** Only complete metadata is activated; a failed final verification restores it. */
export async function withValidationManifest<T>({ manifestPath, before, after, finish, io }: { manifestPath: string; before: Buffer; after: Buffer; finish: () => Promise<T>; io: ManifestIo }): Promise<T> {
  const file = await regularFile({ path: manifestPath });
  const mode = (await lstat(file)).mode & 0o777;
  const temporary = await mkdtemp(path.join(path.dirname(path.dirname(file)), '.bic-validation-'));
  const original = path.join(temporary, 'manifest.original.json');
  const candidate = path.join(temporary, 'manifest.next.json');
  let retain = false;
  try {
    await io.writeFile(original, before);
    await chmod(original, mode);
    await io.writeFile(candidate, after);
    await chmod(candidate, mode);
    if (!(await readFile(await regularFile({ path: file }))).equals(before)) {
      throw new Error('Manifest changed before final validation');
    }
    await io.rename(candidate, file);
    try {
      return await finish();
    } catch (failure) {
      try {
        await io.rename(original, file);
      } catch (recovery) {
        retain = true;
        throw new Error(`Could not restore validation manifest; original retained at ${original}`, { cause: recovery });
      }
      throw failure;
    }
  } finally {
    if (!retain) {
      await rm(temporary, { recursive: true, force: true });
    }
  }
}
