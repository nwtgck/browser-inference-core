import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { runProcess } from '#tools/shared/process';

test('file-backed rows checks the target pointer width', async () => {
  await using workspace = {
    path: await mkdtemp(join(tmpdir(), 'llama-file-rows-width-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  const companion = join(workspace.path, 'upstream/src/models');
  await mkdir(companion, { recursive: true });
  await writeFile(join(companion, 'file-backed-rows.h'), '// Prepared companion exists.\n');
  const overlay = join(process.cwd(), 'runtimes/llama-cpp/cmake/FileBackedRowsOverlay.cmake');
  await writeFile(join(workspace.path, 'CMakeLists.txt'), `\
cmake_minimum_required(VERSION 3.24)
project(file_rows_target_width NONE)
add_library(llama INTERFACE)
set(EMSCRIPTEN "\${TEST_EMSCRIPTEN}")
set(CMAKE_SIZEOF_VOID_P "\${TEST_POINTER_BYTES}")
set(LCB_LLAMA_SOURCE "\${CMAKE_CURRENT_SOURCE_DIR}/upstream")
include("${overlay}")
`);
  for (const input of [
    { name: 'wasm64-with-wasm32-probe', emscripten: true, memory64: true, pointerBytes: 4, accepted: true },
    { name: 'wasm32-with-wide-probe', emscripten: true, memory64: false, pointerBytes: 8, accepted: false },
    { name: 'native64', emscripten: false, memory64: false, pointerBytes: 8, accepted: true },
    { name: 'native32-with-memory64-option', emscripten: false, memory64: true, pointerBytes: 4, accepted: false },
  ]) {
    const configure = runProcess({
      command: 'cmake',
      args: ['-S', workspace.path, '-B', join(workspace.path, input.name),
        `-DTEST_EMSCRIPTEN=${input.emscripten ? 'ON' : 'OFF'}`, `-DLCB_MEMORY64=${input.memory64 ? 'ON' : 'OFF'}`,
        `-DTEST_POINTER_BYTES=${input.pointerBytes}`],
      cwd: workspace.path,
    });
    if (input.accepted) {
      await configure;
    } else {
      await assert.rejects(configure, /requires a 64-bit build/u);
    }
  }
});
