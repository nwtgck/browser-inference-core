"""CLI for the accepted final WebGPU source recipe.

Historical stage builders and expected byte identities are test fixtures, not
production imports or upstream compatibility gates. The pinned Dawn package
integrity check remains distinct from the source recipe and type/behavior tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from build_tools.runtimes.llama_cpp.webgpu_plan import prepare
from build_tools.toolchain.dawn import verify as verify_dawn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dawn-package', type=Path, required=True)
    parser.add_argument('--gpu-profile', action='store_true')
    args = parser.parse_args()
    verify_dawn(args.dawn_package)
    prepare(args.source, args.output, gpu_profile=args.gpu_profile)


if __name__ == '__main__':
    main()
