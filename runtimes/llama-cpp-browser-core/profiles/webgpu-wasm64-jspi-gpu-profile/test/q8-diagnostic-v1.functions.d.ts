/** Derived from the compiled optional diagnostic declaration. */
export interface Q8DiagnosticFunctions {
  version(backend: bigint): number;
  runV1(backend: bigint, dst: bigint, request: bigint, report: bigint, report_capacity: number): Promise<number>;
}
