import { createInterface } from 'node:readline';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { ProcessError, startProcess } from '#tools/shared/process';

function utf8({ label }: { label: string }): Transform {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  return new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      try {
        decoder.decode(chunk, { stream: true });
        callback(undefined, chunk);
      } catch (cause) {
        callback(new Error(`Invalid UTF-8 in ${label}`, { cause }));
      }
    },
    flush(callback) {
      try {
        decoder.decode();
        callback();
      } catch (cause) {
        callback(new Error(`Incomplete UTF-8 at EOF in ${label}`, { cause }));
      }
    },
  });
}

/** Clang emits a sequence of JSON documents for a filtered dump. Existing jq
 * owns JSON syntax and compact JSONL output; no Clang pretty-print framing. */
export async function readClangJson({ command, args, cwd, onDocument }: {
  command: string; args: readonly string[]; cwd: string;
  onDocument: ({ input }: { input: unknown }) => void;
}): Promise<void> {
  const compiler = startProcess({ command, args, cwd });
  const jq = startProcess({ command: 'jq', args: ['-c', '.'], cwd });
  const compilerUtf8 = utf8({ label: 'Clang AST' });
  const jsonlUtf8 = utf8({ label: 'jq JSONL' });
  const lines = createInterface({ input: jsonlUtf8, crlfDelay: Infinity });
  compiler.stdin.end();
  const consume = async (): Promise<void> => {
    for await (const line of lines) {
      const input: unknown = JSON.parse(line);
      onDocument({ input });
    }
  };
  const pending = [
    compiler.completed,
    jq.completed,
    pipeline(compiler.stdout, compilerUtf8, jq.stdin),
    pipeline(jq.stdout, jsonlUtf8),
    consume(),
  ];
  try {
    await Promise.all(pending);
  } catch (error) {
    compiler.kill();
    jq.kill();
    lines.close();
    compilerUtf8.destroy();
    jsonlUtf8.destroy();
    const settled = await Promise.allSettled(pending);
    // A closed pipe may be observed before the producer's original diagnostic.
    // Preserve both, excluding secondary processes killed during cleanup.
    const failures: unknown[] = [error];
    for (const result of settled) {
      switch (result.status) {
      case 'rejected': {
        const reason: unknown = result.reason;
        if (reason instanceof ProcessError && reason.exitCode !== null && !failures.includes(reason)) {
          failures.push(reason);
        }
        break;
      }
      case 'fulfilled': break;
      default: {
        const exhaustive: never = result;
        throw new Error(String(exhaustive));
      }
      }
    }
    throw new AggregateError(failures, `Clang JSON extraction failed: ${failures.map(value => value instanceof Error ? value.message : String(value)).join('\n')}`);
  } finally {
    lines.close();
  }
}
