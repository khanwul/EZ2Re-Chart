"""The v2 user CLI accepts only versioned v2 inputs."""

import argparse
from pathlib import Path
import sys

from .runner import EXIT_FAILED, execute
from .storage import read_json
from .draft import read_draft, save_draft, change_note, parse_beat, describe


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ez2rc", description="EZ2Re:Chart v2")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser('init', help='prepare a project for the supported recording profile')
    init.add_argument('directory', type=Path)
    init.add_argument('--video', type=Path, required=True)
    init.add_argument('--bpm', required=True)
    init.add_argument('--templates', type=Path, required=True, help='directory containing the six user-supplied template PNGs')
    init.add_argument('--title', default='Untitled')
    init.add_argument('--difficulty', choices=('EZ', 'NM', 'HD', 'SHD'), default='SHD')
    review = commands.add_parser('review', help='write a local video review table')
    review.add_argument('draft', type=Path)
    review.add_argument('--output', type=Path, required=True)
    frame = commands.add_parser('frame', help='save a video witness or region as PNG')
    frame.add_argument('draft', type=Path)
    selection = frame.add_mutually_exclusive_group(required=True)
    selection.add_argument('--id')
    selection.add_argument('--time', type=float, help='seconds from the start of the recording')
    frame.add_argument('--side', choices=('start', 'end'), default='start')
    frame.add_argument('--video', type=Path, help='original video in a new location; hash must match')
    frame.add_argument('--output', type=Path, required=True)
    open_command = commands.add_parser('open', help='reopen and inspect an editable draft')
    open_command.add_argument('draft', type=Path)
    open_command.add_argument('--id')
    open_command.add_argument('--unresolved', action='store_true')
    open_command.add_argument('--limit', type=int, default=20)
    for name in ('add', 'edit', 'delete', 'adopt'):
        command = commands.add_parser(name, help=f'{name} a draft note and save the revision')
        command.add_argument('draft', type=Path)
        if name != 'add':
            command.add_argument('--id', required=True)
        if name != 'delete':
            command.add_argument('--lane', type=int, help='lane number, starting at 1', required=name == 'add')
            command.add_argument('--type', choices=('tap', 'longnote'), required=name == 'add')
            command.add_argument('--start', help='beat as decimal or fraction; none for unknown', required=name == 'add')
            command.add_argument('--end', help='beat as decimal or fraction; none for tap or unknown')
            command.add_argument('--confirm', action='store_true', help='mark this complete note as reviewed by you')
        command.add_argument('--output', type=Path, help='save a new file instead of updating the draft')
    for name, description in (("extract", "collect video evidence and reconstruct a chart"),
                              ("reconstruct", "reuse observation evidence without decoding video")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--no-progress", action="store_true", help="hide progress messages")
        command.add_argument("config", type=Path, help="v2 song TOML")
        command.add_argument("--output", type=Path, default=Path("out"))
        if name == "extract":
            command.add_argument("--cache-root", type=Path)
        else:
            command.add_argument("--observations", type=Path, required=True,
                                 help="observation cache directory, or run/artifacts/observations")
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            from .setup import initialize
            song = initialize(args.directory, video=args.video, bpm=args.bpm, templates=args.templates,
                              title=args.title, difficulty=args.difficulty)
            print(f'Project created: {song}')
            print('Check the documented skin, speed and panel settings, then run:')
            print(f'ez2rc extract "{song}" --output out')
            return 0
        if args.command == 'review':
            from .review import write_review
            if args.output.exists():
                raise ValueError('review output already exists; choose a new filename')
            write_review(args.output, read_draft(args.draft))
            print(f'Visual review saved: {args.output}')
            return 0
        if args.command == 'frame':
            from .review import export_frame
            from math import isfinite
            if args.time is not None and not isfinite(args.time):
                raise ValueError('--time must be finite')
            stamp = export_frame(read_draft(args.draft), args.output, identity=args.id, side=args.side,
                                 seconds=args.time, video=args.video)
            print(f'Frame {stamp.frame_id} at {stamp.timestamp_ms/1000:.3f}s: {args.output}')
            return 0
        if args.command == 'open':
            if args.limit < 1:
                raise ValueError('--limit must be positive')
            describe(read_draft(args.draft), identity=args.id, unresolved=args.unresolved, limit=args.limit)
            return 0
        if args.command in ('add', 'edit', 'delete', 'adopt'):
            value = read_draft(args.draft)
            fields = {}
            for flag, key in (('lane', 'lane'), ('type', 'type'), ('start', 'start_beat'), ('end', 'end_beat')):
                option = getattr(args, flag, None)
                if option is not None:
                    fields[key] = option-1 if flag == 'lane' else parse_beat(option) if flag in ('start', 'end') else option
            edited = change_note(value, args.command, identity=getattr(args, 'id', None), fields=fields,
                                 confirm=getattr(args, 'confirm', False))
            destination = args.output or args.draft
            same = destination.resolve() == args.draft.resolve()
            save_draft(destination, edited, expected_sha=value['document_sha256'] if same else None)
            print(f"Saved draft revision {edited['revision']}: {destination}")
            print(f"{args.command}: {edited['history'][-1]['id']}")
            return 0
        from .progress import ProgressDisplay
        progress = ProgressDisplay(sys.stderr)
        if not args.no_progress:
            progress.start()
        try:
            code, root = execute(args.config, output_root=args.output,
                                 cache_root=getattr(args, "cache_root", None),
                                 observation_cache=getattr(args, "observations", None),
                                 progress=None if args.no_progress else progress)
        finally:
            progress.finish()
        run = read_json(root / "run.json", "ez2rc.run", schema_versions=(1, 2))
        print(f"{run['status']}: {root}")
        if 'draft' in run:
            print(f"Editable draft saved: {root / 'draft.json'} ({run['draft']['notes']} notes; video review required).")
        for warning in run.get("warnings", []):
            print(f"warning: {warning['irregular_interval_count']} irregular frame interval(s). "
                  f"{warning['message']}", file=sys.stderr)
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
