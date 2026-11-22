"""The v2 user CLI accepts only versioned v2 inputs."""

import argparse
from pathlib import Path
import sys
import shlex

from .progress import progress_display
from .runner import EXIT_FAILED, execute
from .storage import read_json
from .draft import draft_lock, read_draft, save_draft, change_note, parse_beat, describe


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
    chart = commands.add_parser('chart', help='refresh the PNG from a saved run')
    photos = commands.add_parser('photos', help='save numbered review photos from the original video')
    photos.add_argument('--output', type=Path, help='new directory for the photos')
    photos.add_argument('--video', type=Path, help='original recording in a new location')
    overlay = commands.add_parser('overlay', help='draw the current draft over the original recording')
    overlay.add_argument('--no-progress', action='store_true', help='hide progress messages')
    overlay.add_argument('--output', type=Path, help='new MP4 file (default: overlay.mp4)')
    overlay.add_argument('--video', type=Path, help='original recording in a new location')
    choose = commands.add_parser('choose', help='apply a numbered choice from the PNG')
    for command in (chart, choose, photos, overlay):
        command.add_argument('--run', type=Path, default=Path('.'), help='result directory (default: here)')
    choose.add_argument('number', type=int)
    choose.add_argument('letter', nargs='?', help='letter in the picture, or X to remove')
    choose.add_argument('--start', help='supply a start beat')
    choose.add_argument('--end', help='supply a long-note end beat')
    choose.add_argument('--frame', type=Path, metavar='PNG', help='save the original frame for this option')
    choose.add_argument('--side', choices=('start', 'end'))
    choose.add_argument('--video', type=Path, help='original recording in a new location')
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
        if args.command in ('chart', 'choose', 'photos', 'overlay'):
            return picture_command(args)
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
                                 seconds=args.time, video=args.video, analysis_dir=args.draft.parent)
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
        with progress_display(sys.stderr, enabled=not args.no_progress) as progress:
            code, root = execute(args.config, output_root=args.output,
                                 cache_root=getattr(args, "cache_root", None),
                                 observation_cache=getattr(args, "observations", None),
                                 progress=progress, picture_output=True)
        run = read_json(root / "run.json", "ez2rc.run", schema_versions=(1, 2))
        result_dir = root.parent if root.name == '.ez2rc' else root
        draft_issues = run.get('draft', {}).get('timing_issues')
        if draft_issues is not None and not run['failure']:
            print(f"Draft {run['status']}: {result_dir}")
        else:
            print(f"{run['status']}: {result_dir}")
        if 'picture' in run:
            from .picture import IMAGE_NAME
            print(f"Chart image: {result_dir / IMAGE_NAME}")
            count = run['picture']['pending_choices']
            print(f"Numbered review items: {count}")
            if run['picture'].get('photos'):
                print(f"{run['picture']['photos']} source video photos: {result_dir / run['picture']['folder']}")
            if run['picture'].get('missing_items'):
                print(f"Items without observed frames: {len(run['picture']['missing_items'])}")
            if 'photo_error' in run['picture']:
                print(f"Photo generation failed: {run['picture']['photo_error']}", file=sys.stderr)
            if count:
                print(f"cd {shlex.quote(str(result_dir))}")
                print("ez2rc choose 1     # show choices for the first item")
        elif 'draft' in run:
            print(f"Editable draft saved: {root / 'draft.json'} ({run['draft']['notes']} notes; video review required).")
        for warning in run.get("warnings", []):
            print(f"warning: {warning['irregular_interval_count']} irregular frame interval(s). "
                  f"{warning['message']}", file=sys.stderr)
        if run["failure"]:
            print(f"error [{run['failure']['stage']}]: {run['failure']['message']}", file=sys.stderr)
        elif run["status"] == "partial":
            if draft_issues is None:
                pending = ", ".join(name for name, stage in run["stages"].items()
                                    if stage["status"] == "pending")
                print(f"Partial results saved; pending stages: {pending}.")
            else:
                for issue in draft_issues:
                    print(issue)
        return code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


def picture_command(args):
    from .picture import IMAGE_NAME, TapPlacementItem, beat_text, decide, pending_items, witness_time, write_picture
    directory = args.run
    path = directory / '.ez2rc' / 'draft.json'
    if not path.is_file():
        path = directory / 'draft.json'
    if not path.is_file():
        raise ValueError('Run this command in the result directory or specify it with --run.')
    image = directory / IMAGE_NAME
    if args.command == 'chart':
        with draft_lock(path):
            write_picture(image, read_draft(path))
        print(image)
        return 0
    if args.command == 'overlay':
        from .overlay import write_overlay
        destination = args.output or directory / 'overlay.mp4'
        with progress_display(sys.stderr, enabled=not args.no_progress) as progress:
            if progress:
                progress('Overlay: reading chart draft')
            placed, total, omitted = write_overlay(destination, read_draft(path), path.parent,
                                                  video=args.video, progress=progress)
        print(f'Overlay video: {destination} ({placed}/{total} frames with position data)')
        if omitted:
            print(f'Notes omitted due to missing beat positions: {omitted}')
        return 0
    draft = read_draft(path)
    if args.command == 'photos':
        from .photos import photo_folder, write_photos
        destination = args.output or directory / photo_folder(draft)
        count, missing = write_photos(destination, draft, video=args.video, analysis_dir=path.parent)
        if count:
            print(f'{count} source video photos: {destination}')
        else:
            print('No observed frames are available for photos.')
        if missing:
            print('Items without observed frames: ' + ', '.join(map(str, missing)))
        return 0
    item = pending_items(draft).get(args.number)
    if item is None:
        raise ValueError('No pending choice has this number. Check the current chart image.')
    letter = (args.letter or 'A').upper()
    if args.frame:
        from .review import export_frame
        note = item.frame_note(letter, args.side)
        if note is None:
            raise ValueError('Enter a choice letter shown in the chart image.')
        evidence = note['evidence']
        side = args.side or next((s for s in ('start', 'end') if (evidence.get(s) or {}).get('witnesses')), 'start')
        if not (evidence.get(side) or {}).get('witnesses'):
            raise ValueError('There is no saved source frame for this position.')
        export_frame(draft, args.frame, identity=note['id'], side=side, video=args.video, analysis_dir=path.parent)
        print(args.frame)
        return 0
    if args.side or args.video:
        raise ValueError('Use --side and --video with --frame.')
    if not args.letter:
        from .photos import review_question
        if isinstance(item, TapPlacementItem):
            print(*item.lines(), sep='\n')
            print('Choose A or B to place all taps. Leave the item open to keep their starts unknown.')
            frame_letter = 'A' if witness_time(item.frame_note('A', 'start')) is not None else None
        else:
            print(f'Item {item.number}: {review_question(item.options)}')
            for key, note in item.options.items():
                start = beat_text(note['start_beat'])
                if note['type'] == 'tap':
                    action = f'Tap at beat {start}'
                else:
                    action = f"Hold from beat {start} to beat {beat_text(note['end_beat'])}"
                seconds = witness_time(note)
                when = f' · video {seconds:.3f}s' if seconds is not None else ''
                print(f'{key}  {action}{when}')
            print('X  Remove the note')
            frame_letter = next((key for key, row in item.options.items() if witness_time(row) is not None), None)
        if frame_letter:
            print(f'Save the source frame: ez2rc choose {item.number} {frame_letter} --frame check.png')
        return 0
    fields = {side + '_beat': parse_beat(getattr(args, side)) for side in ('start', 'end') if getattr(args, side) is not None}
    edited = decide(draft, item, letter, fields)
    save_draft(path, edited, expected_sha=draft['document_sha256'], picture_path=image)
    print(f'Saved {item.number}{letter}. Remaining items: {len(pending_items(edited))}. Chart: {image}')
    return 0
