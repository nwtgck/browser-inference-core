"""Pure GitHub identity validation; all remote operations remain in workflow YAML."""
from __future__ import annotations
import re


def full_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Expected a full lowercase Git commit SHA")
    return value


def repository_name(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value)
            or any(part in (".", "..") for part in value.split("/"))):
        raise ValueError("Expected owner/repository")
    return value


def positive_id(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Expected a positive GitHub identifier")
    return value
