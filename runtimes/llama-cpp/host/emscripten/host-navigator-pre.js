// The WebGPU port's unqualified navigator lookup stays local to this factory.
var navigator = Module['hostNavigator'] ?? globalThis.navigator;
