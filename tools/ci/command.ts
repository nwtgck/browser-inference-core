import { spawn, type ChildProcess } from 'node:child_process';
import { Metrics, type MetricResult } from '#tools/ci/metrics';

export type ObservedCommandResult = { exitCode: number | null; signal: NodeJS.Signals | null };

/** Run a real external command with inherited streams and bounded signal cleanup. */
export async function runObservedCommand({ metrics, phase, command, args, cwd, env }: {
  metrics: Metrics; phase: string; command: string; args: readonly string[]; cwd: string; env: NodeJS.ProcessEnv;
}): Promise<ObservedCommandResult> {
  const span = await metrics.begin({ phase, packageKind: undefined });
  let child: ChildProcess | undefined;
  let received: 'SIGINT' | 'SIGTERM' | undefined;
  let timer: NodeJS.Timeout | undefined;
  let outcome: ObservedCommandResult = { exitCode: 126, signal: null };
  const signalGroup = ({ signal }: { signal: NodeJS.Signals }): void => {
    if (!child?.pid) {
      return;
    }
    try {
      switch (process.platform) {
      case 'win32':
        child.kill(signal);
        break;
      default:
        process.kill(-child.pid, signal);
        break;
      }
    } catch (error) {
      if (!(error instanceof Error && 'code' in error && error.code === 'ESRCH')) {
        child.kill(signal);
      }
    }
  };
  const forward = ({ signal }: { signal: 'SIGINT' | 'SIGTERM' }): void => {
    received ??= signal;
    signalGroup({ signal });
    timer ??= setTimeout(() => signalGroup({ signal: 'SIGKILL' }), 2000);
    timer.unref();
  };
  const sigint = () => forward({ signal: 'SIGINT' });
  const sigterm = () => forward({ signal: 'SIGTERM' });
  process.on('SIGINT', sigint);
  process.on('SIGTERM', sigterm);
  try {
    const completed = await new Promise<ObservedCommandResult>((resolve) => {
      try {
        child = spawn(command, [...args], { cwd, env, stdio: 'inherit', detached: process.platform !== 'win32' });
        if (received) {
          signalGroup({ signal: received });
        }
        let spawnFailure: number | undefined;
        child.on('error', (error: NodeJS.ErrnoException) => {
          spawnFailure = error.code === 'ENOENT' ? 127 : 126;
        });
        child.on('close', (exitCode, signal) => resolve({ exitCode: spawnFailure ?? exitCode, signal }));
      } catch {
        resolve({ exitCode: 126, signal: null });
      }
    });
    if (received) {
      // A child can exit cleanly while an ignoring grandchild remains in its group.
      signalGroup({ signal: 'SIGKILL' });
      outcome = { exitCode: null, signal: received };
    } else {
      outcome = completed;
    }
    return outcome;
  } finally {
    clearTimeout(timer);
    process.removeListener('SIGINT', sigint);
    process.removeListener('SIGTERM', sigterm);
    const result: MetricResult = outcome.signal ? 'cancelled' : outcome.exitCode === 0 ? 'success' : 'failure';
    await metrics.end({ start: span, result, exitCode: outcome.exitCode });
  }
}
