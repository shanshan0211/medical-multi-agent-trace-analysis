"""Verify the public allowlist, file hashes, and obvious credential patterns.

This reports filenames/rule names only, never matched secret values. It is a
limited automated scan, not a guarantee about all possible sensitive content.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_ROOTS = {"local_inputs", "paper", "generated_figures", "dist"}
PATTERNS = {
    "api_key_literal": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "github_token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential_assignment": re.compile(r'''(?i)(?:api_key|access_token|password)\s*[:=]\s*["'][^"'\s]{12,}["']'''),
}


def public_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_file() and relative.parts[0] not in IGNORED_ROOTS and not any(
                p in {"__pycache__", ".git", ".venv", ".pytest_cache"} for p in relative.parts):
            yield path


def main():
    manifest_path = ROOT / "metadata/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {e["release_path"]: e for e in manifest["files"]}
    actual = {p.relative_to(ROOT).as_posix(): p for p in public_files() if p != manifest_path}
    if set(actual) != set(expected):
        raise RuntimeError(f"Allowlist mismatch: {sorted(set(actual) ^ set(expected))}")
    problems = []
    for name, path in actual.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]["release_sha256"]:
            problems.append((name, "manifest_hash_mismatch"))
        if path.suffix.lower() in {".jsonl", ".pem", ".key", ".pdf", ".png", ".zip"} or "aiapi" in path.name.lower():
            problems.append((name, "unexpected_public_file_type"))
        content = path.read_text(encoding="utf-8-sig")
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append((name, rule))
    if problems:
        raise RuntimeError(json.dumps(problems))
    print(json.dumps(dict(status="passed", public_files=len(actual) + 1,
        total_public_bytes=sum(p.stat().st_size for p in actual.values()) + manifest_path.stat().st_size,
        credential_patterns_found=0, raw_trace_files_in_allowlist=0)))


if __name__ == "__main__":
    main()
