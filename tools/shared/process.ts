import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtemp, open, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Readable, Writable } from 'node:stream';

export type ProcessResult = { stdout: Buffer; stderr: Buffer; exitCode: number };

export class ProcessError extends Error {
  readonly command: string;
  readonly stderr: Buffer;
  readonly exitCode: number | null;
  stdout: Buffer = Buffer.alloc(0);

  constructor({ command, stderr, exitCode, reason }: { command: string; stderr: Buffer; exitCode: number | null; reason: string }) {
    super(`${command}: ${reason}${stderr.length ? `\n${stderr.toString('utf8').trimEnd()}` : ''}`);
    this.name = 'ProcessError';
    this.command = command;
    this.stderr = stderr;
    this.exitCode = exitCode;
  }
}

type ExitStatus = { code: number | null; signal: NodeJS.Signals | null; failure: Error | undefined };

function completion({ child, timeoutMs }: { child: ChildProcess; timeoutMs: number | undefined }): Promise<ExitStatus> {
  let failure: Error | undefined;
  child.on('error', (error) => {
    failure = error;
  });
  child.stdin?.on('error', (error) => {
    failure ??= error;
  });
  const timer = timeoutMs === undefined ? undefined : setTimeout(() => {
    failure = new Error(`timed out after ${timeoutMs}ms`);
    child.kill('SIGKILL');
  }, timeoutMs);
  timer?.unref();
  return new Promise((resolve) => {
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      resolve({ code, signal, failure });
    });
  });
}

function checkedResult({ command, stdout, stderr, status }: { command: string; stdout: Buffer; stderr: Buffer; status: ExitStatus }): ProcessResult {
  if (status.failure || status.signal || status.code !== 0) {
    const error = new ProcessError({ command, stderr, exitCode: status.code, reason: status.failure?.message ?? (status.signal ? `terminated by ${status.signal}` : `exited with ${String(status.code)}`) });
    error.stdout = stdout;
    if (stdout.length) {
      error.message += `\n${stdout.toString('utf8').trimEnd()}`;
    }
    throw error;
  }
  return { stdout, stderr, exitCode: status.code };
}

/** Streaming processes share the same error/exit handling as buffered commands. */
export function startProcess({ command, args, cwd, env, timeoutMs }: { command: string; args: readonly string[]; cwd: string; env?: NodeJS.ProcessEnv; timeoutMs?: number }) {
  const child = spawn(command, args, { cwd, env, stdio: ['pipe', 'pipe', 'pipe'] });
  const errors: Buffer[] = [];
  child.stderr.on('data', (chunk: Buffer) => errors.push(chunk));
  const completed = completion({ child, timeoutMs }).then((status) => {
    const { stderr, exitCode } = checkedResult({ command, stdout: Buffer.alloc(0), stderr: Buffer.concat(errors), status });
    return { stderr, exitCode };
  });
  // A streaming consumer may still be reading when a failed child closes.
  void completed.catch(() => undefined);
  return { stdin: child.stdin satisfies Writable, stdout: child.stdout satisfies Readable, completed, kill: () => child.kill() };
}

export async function runProcess({ command, args, cwd, env, input, timeoutMs, capture }: { command: string; args: readonly string[]; cwd: string; env?: NodeJS.ProcessEnv; input?: string | Uint8Array; timeoutMs?: number; capture?: 'pipe' | 'file' }): Promise<ProcessResult> {
  switch (capture) {
  case 'file': {
    const temporary = await mkdtemp(join(tmpdir(), 'bicore-process-'));
    try {
      const output = join(temporary, 'stdout');
      const errors = join(temporary, 'stderr');
      const stdout = await open(output, 'wx');
      try {
        const stderr = await open(errors, 'wx');
        try {
          const child = spawn(command, args, { cwd, env, stdio: ['pipe', stdout.fd, stderr.fd] });
          const completed = completion({ child, timeoutMs });
          child.stdin?.end(input);
          const status = await completed;
          return checkedResult({ command, stdout: await readFile(output), stderr: await readFile(errors), status });
        } finally {
          await stderr.close();
        }
      } finally {
        await stdout.close();
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  }
  case 'pipe':
  case undefined:
    break;
  default: {
    const _ex: never = capture;
    throw new Error(`Unhandled process capture mode: ${_ex}`);
  }
  }
  const child = startProcess({ command, args, cwd, ...(env === undefined ? {} : { env }), ...(timeoutMs === undefined ? {} : { timeoutMs }) });
  const chunks: Buffer[] = [];
  child.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
  child.stdin.end(input);
  try {
    const result = await child.completed;
    return { ...result, stdout: Buffer.concat(chunks) };
  } catch (error) {
    if (error instanceof ProcessError) {
      error.stdout = Buffer.concat(chunks);
      if (error.stdout.length) {
        error.message += `\n${error.stdout.toString('utf8').trimEnd()}`;
      }
    }
    throw error;
  }
}

/** Decode a compdb command without invoking a shell or accepting shell expansions. */
export function splitCommandLine({ command }: { command: string }): string[] {
  const result: string[] = [];
  let word = '';
  let active = false;
  let quote: 'single' | 'double' | undefined;
  for (let index = 0; index < command.length; index++) {
    const char = command[index];
    if (char === undefined || char === '\0' || char === '\n' || char === '\r') {
      throw new Error('Unsupported command character');
    }
    switch (quote) {
    case 'single':
      if (char === "'") {
        quote = undefined;
      } else {
        word += char;
      }
      continue;
    case 'double':
    case undefined:
      break;
    default: {
      const _ex: never = quote;
      throw new Error(`Unhandled command quote state: ${_ex}`);
    }
    }
    if (char === '\\') {
      const next = command[++index];
      if (next === undefined || next === '\n' || next === '\r' || next === '\0') {
        throw new Error('Invalid command escape');
      }
      // POSIX double quotes preserve backslashes except before these characters.
      switch (quote) {
      case 'double':
        if (!['$', '`', '"', '\\'].includes(next)) {
          word += '\\';
        }
        break;
      case undefined:
        break;
      default: {
        const _ex: never = quote;
        throw new Error(`Unhandled command quote state: ${_ex}`);
      }
      }
      word += next;
      active = true;
      continue;
    }
    switch (quote) {
    case 'double':
      if (char === '"') {
        quote = undefined;
      } else if (char === '$' || char === '`') {
        throw new Error('Shell expansion is not supported in a compile command');
      } else {
        word += char;
      }
      continue;
    case undefined:
      break;
    default: {
      const _ex: never = quote;
      throw new Error(`Unhandled command quote state: ${_ex}`);
    }
    }
    if (char === "'" || char === '"') {
      switch (char) {
      case "'":
        quote = 'single';
        break;
      case '"':
        quote = 'double';
        break;
      default: {
        const _ex: never = char;
        throw new Error(`Unhandled opening quote: ${_ex}`);
      }
      }
      active = true;
    } else if (/\s/u.test(char)) {
      if (active) {
        result.push(word);
        word = '';
        active = false;
      }
    } else {
      if ('|&;<>()$`*?[]{}~'.includes(char)) {
        throw new Error('Shell operators and expansions are not supported in a compile command');
      }
      word += char;
      active = true;
    }
  }
  if (quote) {
    throw new Error('Unterminated command quote');
  }
  if (active) {
    result.push(word);
  }
  if (!result.length || /^[A-Za-z_][A-Za-z_0-9]*=/u.test(result[0] ?? '')) {
    throw new Error('Expected an executable command');
  }
  return result;
}
