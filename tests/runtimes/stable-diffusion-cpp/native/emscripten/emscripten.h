#pragma once

// The backend's browser branch includes this header but calls no Emscripten API.
// Keep this boundary explicit: native contracts do not emulate JS or Wasm.
