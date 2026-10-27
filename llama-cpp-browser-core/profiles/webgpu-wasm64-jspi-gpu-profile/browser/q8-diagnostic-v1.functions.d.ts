/** Generated optional extension; no private native function pointers. */
export interface Q8DiagnosticFunctions {
  version(backend: bigint): number;
  runV1(backend: bigint, dst: bigint, request: bigint, report: bigint, reportCapacity: number): Promise<number>;
}
