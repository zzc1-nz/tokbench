"""Pre-publish audit: make sure nothing machine-specific or secret is committed.

Two classes of problem this catches, both of which are embarrassing to discover
after a repo goes public:

1. **Absolute local paths** — a README or config that hard-codes
   ``C:\\Users\\<name>\\...`` tells the world your username and is broken for
   everyone else.
2. **Credential-shaped strings** — API keys pasted into a config "just to test".

Incidental digit runs inside floats (``1.24657...``) must NOT trip the path check,
so the path pattern requires a real drive/root prefix.

    python scripts/audit_repo.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = (
    ".git",
    "__pycache__",
    "pytest-cache",
    ".pytemp",
    ".pytmp",
    ".pytest_tmp",
    "_scratch",
    "_tmp_render",
    ".ptmp",
    ".mypy_cache",
    ".ruff_cache",
)
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".html", ".txt", ".jsonl", ".cfg", ".ini"}

# A real absolute path, not an incidental number inside a float.
PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\Users\\[^\"'\s\\]+"), "Windows user path"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "macOS user path"),
    (re.compile(r"/home/(?!runner/)[A-Za-z0-9._-]+/"), "Linux user path"),
    (re.compile(r"default-workspace"), "sandbox workspace path"),
]

#: Files that legitimately contain path-like strings and must not be flagged:
#:
#: - this script (it defines the patterns)
#: - the task set: several tasks are *about* CI logs, so ``/home/runner/...`` is
#:   the realistic tool output the benchmark is supposed to measure, not a leak.
AUDIT_EXEMPT = {
    "scripts/audit_repo.py",
    "scripts/build_taskset.py",  # generates the task payloads above
    "tasks/coding_bench.jsonl",
}

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style key"),
    (re.compile(r"sk-ant-[A-Za-z0-9-]{20,}"), "Anthropic key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "GitHub personal access token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
]

findings: list[str] = []


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel for part in SKIP_PARTS):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    files = iter_files()
    print(f"auditing {len(files)} text files under {ROOT.name}/")

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in AUDIT_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            findings.append(f"{rel}: unreadable ({exc})")
            continue
        for pattern, label in PATH_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(f"{rel}: {label} -> {match.group(0)[:80]}")
        for pattern, label in SECRET_PATTERNS:
            for _ in pattern.finditer(text):
                findings.append(f"{rel}: possible {label}")

    # Also confirm the staged set (if this is a git repo) matches what we scanned.
    git_dir = ROOT / ".git"
    if git_dir.exists():
        try:
            out = subprocess.run(
                ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
            )
            tracked = [line for line in out.stdout.splitlines() if line.strip()]
            print(f"git reports {len(tracked)} tracked files")
            junk = [
                f
                for f in tracked
                if any(part in f for part in ("pytest-cache", ".pytemp", ".pytmp", ".pytest_tmp", "_scratch", "_tmp_render", ".ptmp"))
            ]
            for j in junk:
                findings.append(f"{j}: scratch artefact is tracked (should be gitignored)")
        except OSError:
            pass

    if findings:
        print(f"\n{len(findings)} finding(s):", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nCLEAN — no absolute paths, no credential-shaped strings, no tracked scratch files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
