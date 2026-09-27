"""Mints a dev bearer token for the gateway. Run: python -m scripts.print_token [subject]
(plain `python scripts/print_token.py` fails — the script's own directory
lands on sys.path, not the repo root, so `from core.auth import ...` can't
resolve; -m runs it as a module from the repo root instead)."""

import sys

from core.auth import issue_token

if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "dev-user"
    print(issue_token(subject))
