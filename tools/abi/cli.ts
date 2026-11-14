import { writeFileSync } from 'node:fs';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { generateBindings, type BindingPolicy } from './generate.js';

/** CMake is a real external generator boundary; the implementation remains a
 * normal typed call using the policy imported by this checkout's runtime entry. */
export async function runBindingsCli({ policy }: { policy: BindingPolicy }): Promise<void> {
  const values = await yargs(hideBin(process.argv))
    .option('compile-root', { type: 'string', demandOption: true })
    .option('compilation-database', { type: 'string', demandOption: true })
    .option('translation-unit', { type: 'string', demandOption: true })
    .option('bridge-unit', { type: 'string', demandOption: true })
    .option('output', { type: 'string', demandOption: true })
    .option('runtime-directory', { type: 'string' })
    .strict()
    .parseAsync();
  const schema = await generateBindings({
    policy,
    compileRoot: values['compile-root'],
    compilationDatabase: values['compilation-database'],
    translationUnit: values['translation-unit'],
    bridgeUnit: values['bridge-unit'],
    output: values.output,
    runtimeDirectory: values['runtime-directory'],
  });
  writeFileSync(1, JSON.stringify({ schemaSha256: schema.schemaSha256, functions: schema.functions.length, records: schema.records.length, pointerBytes: schema.pointerBytes }) + '\n');
}
