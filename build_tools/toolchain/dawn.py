"""Integrity of the pinned Dawn headers, not an upstream semantic contract.

The archive's hash is verified during toolchain acquisition. These retained
header identities also protect direct source-CLI callers from a mixed SDK.
They do not replace C++ compilation, browser probes or numerical tests.
"""
from __future__ import annotations

from pathlib import Path

from build_tools.source.files import file_digest

PINNED_RELEASE = 'v20260908.214631'
PINNED_HEADERS = {
    'webgpu/include/webgpu/webgpu.h': 'da1bbcb1c6e0f65d917c4829b465454914b44ea16fa49c3146fc44e606ac989a',
    'webgpu_cpp/include/webgpu/webgpu_cpp.h': 'dbad68fcf60e9b9948331bf388cfbe0317e7628ff3bb28246c8b6bb1ce4e9701',
    'webgpu_cpp/include/webgpu/webgpu_cpp_chained_struct.h': '582e62d371f29391d78a7356c634c6b76e1d58b6257a6f5be51701d15797fce1',
    'webgpu_cpp/include/webgpu/webgpu_enum_class_bitmasks.h': 'fd436dc17d050e156ac073b6148d90d388e2585de924ae0274b6aed3067c2a96',
}


def verify(package: Path) -> None:
    for relative, expected in PINNED_HEADERS.items():
        if file_digest(package / relative) != expected:
            raise ValueError("Pinned Dawn header integrity mismatch: " + relative)
