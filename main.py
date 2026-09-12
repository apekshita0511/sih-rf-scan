"""Convenience wrapper so the CLI can be run as ``python main.py ...``.

The installed entry point ``rfscan`` (see pyproject.toml) is equivalent.
"""

from rfscan.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
