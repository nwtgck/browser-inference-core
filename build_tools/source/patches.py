"""Apply a declared patch without inheriting a surrounding Git repository.

The declaration limits *which* files a patch may edit. Input/output hashes are
observations in the receipt, not hand-maintained claims of semantic compatibility.
Type, layout, arithmetic and lifetime contracts belong to their real tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable

from build_tools.source.files import directory, optional_file_digest, regular_file, relative_path


@dataclass(frozen=True)
class Patch:
    file: str
    target: str
    reason: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        relative_path(self.file)
        if not self.file.endswith(".patch"):
            raise ValueError("An upstream patch must be an explicit .patch file")
        if not isinstance(self.target, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", self.target):
            raise ValueError("Invalid patch target")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("A patch needs its purpose in the inventory")
        if not isinstance(self.files, tuple) or not self.files or any(not isinstance(name, str) for name in self.files) or len(set(self.files)) != len(self.files):
            raise ValueError("Empty or duplicate patch inventory")
        for name in self.files:
            relative_path(name)


@dataclass(frozen=True)
class Series:
    patches: tuple[Patch, ...]
    sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate inventory key: {name}")
        result[name] = value
    return result


def read_series(path: Path, *, targets: Iterable[str]) -> Series:
    """Read a small declarative inventory, not an executable rewrite language."""
    raw = regular_file(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"formatVersion", "patches"}:
        raise ValueError("Unknown patch inventory fields")
    if type(value["formatVersion"]) is not int or value["formatVersion"] != 2 or not isinstance(value["patches"], list):
        raise ValueError("Unknown patch inventory version")
    accepted = set(targets)
    result = []
    seen = set()
    for entry in value["patches"]:
        if not isinstance(entry, dict) or set(entry) != {"file", "target", "reason", "files"} or not isinstance(entry["files"], list):
            raise ValueError("Unknown patch declaration fields")
        patch = Patch(entry["file"], entry["target"], entry["reason"], tuple(entry["files"]))
        if patch.target not in accepted:
            raise ValueError(f"Unknown patch target: {patch.target}")
        if patch.file in seen:
            raise ValueError(f"Duplicate patch declaration: {patch.file}")
        seen.add(patch.file)
        result.append(patch)
    return Series(tuple(result), hashlib.sha256(raw).hexdigest())


def patch_environment(work: Path) -> dict[str, str]:
    # An inherited index/worktree or user apply.ignoreWhitespace setting must
    # not redirect a patch or turn changed code into an ignored difference.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CEILING_DIRECTORIES": str(work.parent),
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"})
    return env


def _git(work: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--no-index", *arguments],
                            cwd=work, env=patch_environment(work), capture_output=True)
    if result.returncode:
        raise ValueError("Patch application failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _changed_paths(work: Path, patch: Path) -> tuple[str, ...]:
    raw = _git(work, "--numstat", "-z", str(patch))
    paths = []
    for row in raw.split(b"\0"):
        if not row:
            continue
        fields = row.split(b"\t", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise ValueError("Expected a text-only source patch with explicit file paths")
        name = fields[2].decode("utf-8")
        relative_path(name)
        paths.append(name)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Empty or duplicate patch changes")
    return tuple(paths)


def apply_patch(work: Path, patches: Path, patch: Patch) -> dict:
    """Apply one patch to a private tree and report exactly the bytes used.

    The caller owns the transaction: failed work must not be published. A private
    immutable copy of the patch is used for inspection, checking and application.
    """
    work = directory(work)
    raw = regular_file(patches / relative_path(patch.file)).read_bytes()
    # Source patches may add/remove regular files and change their executable bit.
    # Reject link/gitlink/binary/rename representations instead of interpreting
    # a numstat destination as permission to modify an undeclared source path.
    text = raw.decode("utf-8")
    if re.search(r"^(?:rename |copy |GIT binary patch|Binary files |(?:old|new|deleted) (?:file )?mode (?!100(?:644|755)$))", text, re.MULTILINE):
        raise ValueError(f"Unsupported source patch representation: {patch.file}")
    with tempfile.TemporaryDirectory(prefix="bicore-patch-") as temporary:
        captured = Path(temporary) / "input.patch"
        captured.write_bytes(raw)
        changed = _changed_paths(work, captured)
        if set(changed) != set(patch.files):
            raise ValueError(f"Patch changes files outside its inventory: {patch.file}")
        # The declaration owns receipt ordering; numstat only verifies scope.
        # Keep records stable when a multi-file patch orders its hunks differently.
        changed = patch.files
        before = {name: optional_file_digest(work, name) for name in changed}
        _git(work, "--check", "--whitespace=error-all", str(captured))
        _git(work, "--whitespace=error-all", str(captured))
        after = {name: optional_file_digest(work, name) for name in changed}
    return {"file": patch.file, "target": patch.target, "reason": patch.reason,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "files": [{"path": name, "beforeSha256": before[name], "afterSha256": after[name]}
                      for name in changed]}
