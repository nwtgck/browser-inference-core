/** Independent consumer of native encoder output. No native decoder is reused. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { decodeDelta } from '../runtime/delta.mjs';
import { predict } from '../runtime/predict.mjs';
const cases = JSON.parse(await readFile(process.argv[2], 'utf8'));
let verified = 0;
for (const { base, target, output, expected } of cases) {
    const a = new Uint8Array(await readFile(base));
    const b = new Uint8Array(await readFile(target));
    const report = JSON.parse(await readFile(`${output}/result.json`, 'utf8'));
    for (const candidate of report.candidates) {
        let dictionary = a;
        if (candidate.prediction) {
            const expectedDictionary = new Uint8Array(await readFile(`${output}/${candidate.dictionary}`));
            const rules = new Uint8Array(await readFile(`${output}/${candidate.prediction}`));
            dictionary = predict(a, rules, { expectedBytes: expectedDictionary.length });
            assert.deepEqual(dictionary, expectedDictionary);
        }
        const recipe = new Uint8Array(await readFile(`${output}/${candidate.recipe}`));
        const decoded = decodeDelta(dictionary, recipe, { expectedTargetBytes: b.length });
        assert.deepEqual(decoded, b);
        assert.equal(new WebAssembly.Instance(new WebAssembly.Module(decoded)).exports.f(), expected);
        verified++;
    }
}
console.log(JSON.stringify({ cases: cases.length, verifiedRepresentations: verified }));
