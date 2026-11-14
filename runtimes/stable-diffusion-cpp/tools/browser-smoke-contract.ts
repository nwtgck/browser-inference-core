import contract from '#runtimes/stable-diffusion-cpp/config/browser-smoke-contract.json' with { type: 'json' };

/** One scope producer shared by the browser harness and artifact validation. */
export const imageSmokeContract = contract;

export function imageSmokeScope({ testWebGpu }: { testWebGpu: boolean }): string {
  if (typeof testWebGpu !== 'boolean') {
    throw new TypeError('testWebGpu must be boolean');
  }
  return imageSmokeContract.scopePrefix + (testWebGpu ? 'CPU and WebGPU' : 'CPU (no GPU inference)') + imageSmokeContract.scopeSuffix;
}
