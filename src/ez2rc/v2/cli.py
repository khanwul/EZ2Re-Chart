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
    overlay.add_argument('--output', type=Path, help='new MP4 file (default: 대조.mp4)')
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
        print(f"{run['status']}: {result_dir}")
        if 'picture' in run:
            from .picture import IMAGE_NAME
            print(f"채보 그림: {result_dir / IMAGE_NAME}")
            count = run['picture']['pending_choices']
            print(f"번호가 붙은 확인 항목 {count}개")
            if run['picture'].get('photos'):
                print(f"원영상 사진 {run['picture']['photos']}장: {result_dir / run['picture']['folder']}")
            if run['picture'].get('missing_items'):
                print(f"관측 프레임이 없는 항목 {len(run['picture']['missing_items'])}개")
            if 'photo_error' in run['picture']:
                print(f"사진 생성 실패: {run['picture']['photo_error']}", file=sys.stderr)
            if count:
                print(f"cd {shlex.quote(str(result_dir))}")
                print("ez2rc choose 1     # 첫 항목의 선택지 보기")
        elif 'draft' in run:
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


def picture_command(args):
    from .picture import IMAGE_NAME, beat_text, decide, pending_items, witness_time, write_picture
    directory = args.run
    path = directory / '.ez2rc' / 'draft.json'
    if not path.is_file():
        path = directory / 'draft.json'
    if not path.is_file():
        raise ValueError('결과 폴더에서 실행하거나 --run 옵션으로 결과 폴더를 지정하세요.')
    image = directory / IMAGE_NAME
    if args.command == 'chart':
        with draft_lock(path):
            write_picture(image, read_draft(path))
        print(image)
        return 0
    if args.command == 'overlay':
        from .overlay import write_overlay
        destination = args.output or directory / '대조.mp4'
        with progress_display(sys.stderr, enabled=not args.no_progress) as progress:
            if progress:
                progress('대조 영상 · 채보 초안 읽는 중')
            placed, total, omitted = write_overlay(destination, read_draft(path), path.parent,
                                                  video=args.video, progress=progress)
        print(f'대조 영상: {destination} · 위치 대조 가능 {placed}/{total} 프레임')
        if omitted:
            print(f'박자 정보가 부족해 표시할 수 없는 노트 {omitted}개')
        return 0
    draft = read_draft(path)
    if args.command == 'photos':
        from .photos import photo_folder, write_photos
        destination = args.output or directory / photo_folder(draft)
        count, missing = write_photos(destination, draft, video=args.video, analysis_dir=path.parent)
        if count:
            print(f'원영상 사진 {count}장: {destination}')
        else:
            print('사진으로 표시할 관측 프레임이 없습니다.')
        if missing:
            print('관측 프레임이 없는 항목: ' + ', '.join(map(str, missing)))
        return 0
    item = pending_items(draft).get(args.number)
    if item is None:
        raise ValueError('이 번호로 선택할 항목이 없습니다. 현재 채보 그림을 확인하세요.')
    letter = (args.letter or 'A').upper()
    if args.frame:
        from .review import export_frame
        note = item.options.get(letter)
        if note is None:
            raise ValueError('그림에 표시된 후보 문자를 입력하세요.')
        evidence = note['evidence']
        side = args.side or next((s for s in ('start', 'end') if (evidence.get(s) or {}).get('witnesses')), 'start')
        if not (evidence.get(side) or {}).get('witnesses'):
            raise ValueError('이 위치에는 저장된 원본 프레임 근거가 없습니다.')
        export_frame(draft, args.frame, identity=note['id'], side=side, video=args.video, analysis_dir=path.parent)
        print(args.frame)
        return 0
    if args.side or args.video:
        raise ValueError('--side와 --video는 --frame과 함께 사용하세요.')
    if not args.letter:
        from .photos import review_question
        print(f'{item.number}번 · {review_question(item.options)}')
        for key, note in item.options.items():
            start = beat_text(note['start_beat'])
            if note['type'] == 'tap':
                action = f'일반 노트 · {start}박'
            else:
                action = f"롱노트 · 시작 {start}박 · 끝 {beat_text(note['end_beat'])}박"
            seconds = witness_time(note)
            when = f' · 원영상 {seconds:.3f}초' if seconds is not None else ''
            print(f'{key}  {action}{when}')
        print('X  노트 삭제')
        frame_letter = next((key for key, row in item.options.items() if witness_time(row) is not None), None)
        if frame_letter:
            print(f'원본 프레임 저장: ez2rc choose {item.number} {frame_letter} --frame 확인.png')
        return 0
    fields = {side + '_beat': parse_beat(getattr(args, side)) for side in ('start', 'end') if getattr(args, side) is not None}
    edited = decide(draft, item, letter, fields)
    save_draft(path, edited, expected_sha=draft['document_sha256'], picture_path=image)
    print(f'{item.number}{letter} 저장 · 남은 항목 {len(pending_items(edited))}개 · {image}')
    return 0
