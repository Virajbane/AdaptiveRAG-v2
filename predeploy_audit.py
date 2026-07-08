#!/usr/bin/env python3
"""
predeploy_audit.py

Run this before every deploy (or right after cloning the repo) to catch:
  1. Hardcoded localhost/127.0.0.1 URLs left in source instead of settings/env
  2. Hardcoded service URLs (mongodb://, redis://, qdrant, ollama) typed directly
     instead of read from settings
  3. API keys / secrets accidentally committed as literal strings
  4. .env keys that exist locally but are missing from .env.example
     (so a fresh clone knows every variable it needs to set)

Usage:
    python predeploy_audit.py

Exit code 0 = clean, 1 = issues found (so it can be wired into CI later if wanted).
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".next", "venv", ".venv",
    "dist", "build", ".vercel", ".render"
}

# File extensions worth scanning
SCAN_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".yml", ".yaml", ".env", ""}

# Patterns that indicate a hardcoded value that should come from settings/env instead
HARDCODE_PATTERNS = [
    (r"localhost:\d+", "hardcoded localhost URL"),
    (r"127\.0\.0\.1", "hardcoded loopback IP"),
    (r"mongodb(\+srv)?://[^\s\"'$]+", "hardcoded MongoDB connection string"),
    (r"rediss?://[^\s\"'$]+", "hardcoded Redis connection string"),
]

# Patterns that look like real secrets (rough heuristics, not perfect)
SECRET_PATTERNS = [
    (r"gsk_[A-Za-z0-9]{20,}", "Groq API key"),
    (r"tvly-[A-Za-z0-9-]{10,}", "Tavily API key"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API key"),
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "JWT-looking token / API key"),
    (r"pa-[A-Za-z0-9_-]{10,}", "Voyage API key"),
]

ENV_FILE_CANDIDATES = [".env", "backend/.env", "app/.env"]
ENV_EXAMPLE_CANDIDATES = [".env.example", "backend/.env.example", "app/.env.example"]


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext in SCAN_EXTS or fname in (".env", ".env.example"):
                yield os.path.join(dirpath, fname)


def scan_file(path):
    issues = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return issues

    is_settings_file = os.path.basename(path) == "settings.py"
    is_this_script = os.path.abspath(path) == os.path.abspath(__file__)
    if is_this_script:
        return issues

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        for pattern, label in HARDCODE_PATTERNS:
            if re.search(pattern, line):
                if is_settings_file and ("localhost" in line or "127.0.0.1" in line):
                    continue
                issues.append((path, lineno, "HARDCODE", label, stripped[:120]))

        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, line):
                issues.append((path, lineno, "SECRET", label, stripped[:60] + " ...[REDACTED REST]"))

    return issues


def check_env_example_coverage():
    issues = []
    env_path = None
    example_path = None
    for c in ENV_FILE_CANDIDATES:
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            env_path = p
            break
    for c in ENV_EXAMPLE_CANDIDATES:
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            example_path = p
            break

    if not env_path:
        return issues

    def get_keys(path):
        keys = set()
        if not path or not os.path.exists(path):
            return keys
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                keys.add(line.split("=", 1)[0].strip())
        return keys

    env_keys = get_keys(env_path)
    example_keys = get_keys(example_path)

    missing_from_example = env_keys - example_keys
    if missing_from_example:
        issues.append((
            example_path or "(missing .env.example)",
            0,
            "ENV_EXAMPLE_GAP",
            "keys in .env but missing from .env.example",
            ", ".join(sorted(missing_from_example)),
        ))

    return issues


def main():
    all_issues = []
    for filepath in iter_files():
        all_issues.extend(scan_file(filepath))

    all_issues.extend(check_env_example_coverage())

    hardcode_issues = [i for i in all_issues if i[2] == "HARDCODE"]
    secret_issues = [i for i in all_issues if i[2] == "SECRET"]
    env_gap_issues = [i for i in all_issues if i[2] == "ENV_EXAMPLE_GAP"]

    print("=" * 70)
    print("PRE-DEPLOY AUDIT REPORT")
    print("=" * 70)

    if hardcode_issues:
        print(f"\n[HARDCODED VALUES] {len(hardcode_issues)} found:\n")
        for path, lineno, _, label, snippet in hardcode_issues:
            rel = os.path.relpath(path, ROOT)
            print(f"  {rel}:{lineno}  ({label})\n      {snippet}\n")
    else:
        print("\n[HARDCODED VALUES] none found - OK")

    if secret_issues:
        print(f"\n[POSSIBLE SECRETS IN CODE] {len(secret_issues)} found:\n")
        for path, lineno, _, label, snippet in secret_issues:
            rel = os.path.relpath(path, ROOT)
            print(f"  {rel}:{lineno}  ({label})\n      {snippet}\n")
    else:
        print("\n[POSSIBLE SECRETS IN CODE] none found - OK")

    if env_gap_issues:
        print("\n[.env.example GAPS] found:\n")
        for path, _, _, label, snippet in env_gap_issues:
            print(f"  {label}: {snippet}")
            print("  -> add these keys (with blank/dummy values) to .env.example\n")
    else:
        print("\n[.env.example COVERAGE] ok (or no .env found to compare)")

    total = len(hardcode_issues) + len(secret_issues) + len(env_gap_issues)
    print("\n" + "=" * 70)
    if total == 0:
        print("RESULT: clean. Safe to deploy.")
        return 0
    else:
        print(f"RESULT: {total} issue(s) found. Review before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())