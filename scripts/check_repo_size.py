"""Pre-push safety check: list files git would add that are too big for GitHub.

GitHub rejects any file over 100 MB (and warns above 50 MB). Run BEFORE `git add .`:
    python scripts/check_repo_size.py
Needs git. Honors .gitignore (uses `git ls-files --cached --others --exclude-standard`).
"""
import os
import subprocess
import sys

WARN, BLOCK = 50 * 1024**2, 100 * 1024**2

out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                     capture_output=True, text=True, check=True).stdout.splitlines()
files = [(os.path.getsize(f), f) for f in out if os.path.isfile(f)]
total = sum(s for s, _ in files)
bad = [(s, f) for s, f in files if s >= WARN]

print(f"{len(files)} files would be pushed, {total / 1024**2:.1f} MB total")
for s, f in sorted(files, reverse=True)[:8]:
    print(f"  {s / 1024**2:8.2f} MB  {f}")
for s, f in bad:
    tag = "BLOCKED by GitHub (>100 MB)" if s >= BLOCK else "large (>50 MB)"
    print(f"!! {tag}: {f} ({s / 1024**2:.1f} MB)")
sys.exit(1 if any(s >= BLOCK for s, _ in bad) else 0)
