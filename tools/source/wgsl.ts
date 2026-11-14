import { copyFile, mkdir, readdir, stat } from 'node:fs/promises';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { directory, isContained, regularFile } from '#tools/shared/files';
import { replaceFileUnchanged } from '#tools/shared/output';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tools/shared/temporary-directory';

export async function wgslInputs({ sourceRoot, generator }: { sourceRoot: string; generator?: string }): Promise<readonly string[]> {
  sourceRoot = await directory({ path: sourceRoot });
  const program = await regularFile({ path: generator ?? join(sourceRoot, 'embed_wgsl.py') });
  if (basename(program) !== 'embed_wgsl.py') {
    throw new Error('Expected the upstream embed_wgsl.py generator');
  }
  const files = (await readdir(sourceRoot)).filter((name) => ['.wgsl', '.tmpl'].includes(extname(name))).map((name) => join(sourceRoot, name));
  if (!files.some((file) => extname(file) === '.wgsl')) {
    throw new Error('No canonical WGSL sources');
  }
  files.push(program);
  for (const path of files) {
    await regularFile({ path });
  }
  return files.sort();
}

/** Invoke the unmodified upstream generator after patching its private WGSL inputs. */
export async function generateWgsl({ sourceRoot, output, generator, patchedTemplate, python }: { sourceRoot: string; output: string; generator?: string; patchedTemplate?: string; python?: string }): Promise<void> {
  sourceRoot = await directory({ path: sourceRoot });
  generator = await regularFile({ path: generator ?? join(sourceRoot, 'embed_wgsl.py') });
  const inputs = await wgslInputs({ sourceRoot, generator });
  if (patchedTemplate !== undefined) {
    patchedTemplate = await regularFile({ path: patchedTemplate });
    if (isContained({ root: sourceRoot, path: patchedTemplate }) || extname(patchedTemplate) !== '.tmpl' || !inputs.includes(join(sourceRoot, basename(patchedTemplate)))) {
      throw new Error('Patched template must be a declared template outside upstream shaders');
    }
  }
  output = resolve(output);
  if (isContained({ root: sourceRoot, path: output }) || isContained({ root: dirname(generator), path: output }) || output === patchedTemplate) {
    throw new Error('Generated header must not replace an input');
  }
  await directory({ path: dirname(output), create: true });
  await using temporary = await temporaryDirectory({ prefix: join(dirname(output), 'wgsl-embed-') });
  const privateInputs = join(temporary.root, 'inputs');
  await mkdir(privateInputs);
  for (const input of inputs) {
    await copyFile(patchedTemplate !== undefined && basename(input) === basename(patchedTemplate) ? patchedTemplate : input, join(privateInputs, basename(input)));
  }
  const header = join(temporary.root, 'shaders.hpp');
  await runProcess({ command: python ?? 'python3', args: ['-I', '-S', join(privateInputs, 'embed_wgsl.py'), '--input_dir', privateInputs, '--output_file', header], cwd: temporary.root });
  if (!(await stat(await regularFile({ path: header }))).size) {
    throw new Error('Canonical shader generator produced no header');
  }
  await replaceFileUnchanged({ candidate: header, output });
}
