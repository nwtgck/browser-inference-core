#!/usr/bin/env node
import { writeSync } from 'node:fs';

if (process.argv[2] === 'compiler') {
  const mode = process.argv[3];
  switch (mode) {
  case 'valid':
    writeSync(1, JSON.stringify({ kind: 'FunctionDecl', name: 'a', value: '日本語 } \\"\n' }) + '\n' + JSON.stringify({ kind: 'FunctionDecl', name: 'b' }));
    break;
  case 'compiler_error':
    writeSync(1, JSON.stringify({ kind: 'FunctionDecl', name: 'partial' }));
    writeSync(2, 'actual compiler diagnostic\n');
    process.exitCode = 7;
    break;
  case 'json_eof': writeSync(1, '{"kind":');
    break;
  case 'utf8': writeSync(1, Buffer.from([0x7b, 0x22, 0xc3, 0x28]));
    break;
  case 'utf8_eof': writeSync(1, Buffer.from([0x7b, 0x22, 0xc3]));
    break;
  default: throw new Error(`Unknown compiler fixture mode: ${mode}`);
  }
} else {
  process.stdin.resume();
  process.stdin.on('end', () => {
    switch (process.env.BICORE_TEST_JQ_MODE) {
    case 'json_eof': writeSync(1, '{');
      break;
    case 'utf8': writeSync(1, Buffer.from([0xc3, 0x28]));
      break;
    case 'exit':
      writeSync(2, 'actual jq diagnostic\n');
      process.exitCode = 9;
      break;
    default: throw new Error('Unknown jq fixture mode');
    }
  });
}
