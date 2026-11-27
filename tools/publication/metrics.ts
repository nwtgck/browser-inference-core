export type MeasurePublication = <T>({ phase, operation }: { phase: string; operation: () => Promise<T> }) => Promise<T>;

export const unmeasuredPublication: MeasurePublication = ({ operation }) => operation();
