"""The v2 user CLI accepts only versioned v2 inputs."""

import argparse
from pathlib import Path
import sys

from .runner import EXIT_FAILED, execute
from .storage import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ez2rc", description="EZ2Re:Chart v2")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("extract", "collect video evidence and reconstruct a chart"),
                              ("reconstruct", "reuse observation evidence without decoding video")):
        command = commands.add_parser(name, help=description)
        command.add_argument("config", type=Path, help="v2 song TOML")
        command.add_argument("--output", type=Path, default=Path("out"))
        if name == "extract":
            command.add_argument("--cache-root", type=Path)
        else:
            command.add_argument("--observations", type=Path, required=True,
                                 help="observation cache directory, or run/artifacts/observations")
    args = parser.parse_args(argv)
    try:
        code, root = execute(args.config, output_root=args.output,
                             cache_root=getattr(args, "cache_root", None),
                             observation_cache=getattr(args, "observations", None))
        run = read_json(root / "run.json", "ez2rc.run")
        print(f"{run['status']}: {root}")
        if run["failure"]:
            print(f"error [{run['failure']['stage']}]: {run['failure']['message']}", file=sys.stderr)
        elif run["status"] == "partial":
            pending = ", ".join(name for name, stage in run["stages"].items()
                                if stage["status"] == "pending")
            print(f"Partial results saved; pending stages: {pending}.")
        return code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
