import contract from '#runtimes/stable-diffusion-cpp/config/browser-smoke-contract.json' with { type: 'json' };

/** One scope producer shared by the browser harness and artifact validation. */
export const imageSmokeContract = contract;

export function imageSmokeScope({ variant, testWebGpu }: { variant: string; testWebGpu: boolean }): string {
  if (typeof testWebGpu !== 'boolean') {
    throw new TypeError('testWebGpu must be boolean');
  }
  switch (variant) {
  case 'browser': return imageSmokeContract.browserScope;
  case 'test': return imageSmokeContract.scopePrefix + (testWebGpu ? 'CPU and WebGPU' : 'CPU (no GPU inference)') + imageSmokeContract.scopeSuffix;
  default: throw new TypeError(`Unknown image variant: ${variant}`);
  }
}
