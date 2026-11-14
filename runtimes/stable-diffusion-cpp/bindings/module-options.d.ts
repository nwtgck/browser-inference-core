/** Public options accepted by the Emscripten module factory. */
export interface CoreOptions {
  wasmBinary?: Uint8Array<ArrayBuffer>;
  /** Call onSuccess to complete initialization. The hook's return value is ignored. */
  instantiateWasm?(imports: WebAssembly.Imports, onSuccess: (instance: WebAssembly.Instance, module?: WebAssembly.Module) => void): void;
  locateFile?(path: string): string;
  print?(text: string): void;
  printErr?(text: string): void;
  onAbort?(reason: unknown): void;
}
