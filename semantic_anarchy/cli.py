"""Thin dispatcher: ``python -m semantic_anarchy.cli <command> [args...]``.

Routes to the standalone scripts in ``scripts/`` so the package has one entry
point. Each subcommand forwards its remaining argv straight to that script's
``main``::

    python -m semantic_anarchy.cli demo
    python -m semantic_anarchy.cli mine --prompts prompts.txt
    python -m semantic_anarchy.cli generate --dist outputs/dist --n 8
    python -m semantic_anarchy.cli evolve --scorer aesthetic
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_COMMANDS = {
    "demo": "demo_no_sd.py",
    "mine": "mine_distribution.py",
    "generate": "generate.py",
    "evolve": "evolve.py",
}


def _load(script: str):
    path = _SCRIPTS / script
    spec = importlib.util.spec_from_file_location(f"_sa_{script}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in _COMMANDS:
        print("usage: python -m semantic_anarchy.cli <command> [args...]")
        print("commands: " + " | ".join(_COMMANDS))
        print("  demo      run the no-SD synthetic demo (no torch needed)")
        print("  mine      encode prompts.txt and fit the distribution (needs SD)")
        print("  generate  sample embeddings and decode them (needs SD)")
        print("  evolve    grow an evolutionary branch by aesthetic selection")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module = _load(_COMMANDS[argv[0]])
    return int(module.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
