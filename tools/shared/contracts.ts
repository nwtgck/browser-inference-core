import type { FileEntry } from '#dto/shared';
import { relativePath } from './files.js';

export const runtimeNames = ['llama-cpp', 'stable-diffusion-cpp'] as const;
export const artifactDirectories = {
  'llama-cpp': 'runtimes/llama-cpp-browser-core',
  'stable-diffusion-cpp': 'runtimes/stable-diffusion-cpp-browser-core',
} as const;
export const packageName = 'browser-inference-core';

/** Each path has one owner, including on case-insensitive filesystems. */
export function inventory({ entries }: { entries: readonly FileEntry[] }): Map<string, FileEntry> {
  const result = new Map<string, FileEntry>();
  const folded = new Set<string>();
  for (const entry of entries) {
    relativePath({ value: entry.path });
    if (entry.path.includes(':')) {
      throw new Error(`Unsafe artifact path: ${entry.path}`);
    }
    const key = entry.path.toLowerCase();
    if (folded.has(key)) {
      throw new Error(`Duplicate or case-colliding artifact path: ${entry.path}`);
    }
    folded.add(key);
    result.set(entry.path, { path: entry.path, bytes: entry.bytes, sha256: entry.sha256 });
  }
  return result;
}
