"""Create a portable project using the packaged profile and user templates."""
from importlib.resources import files
import json
from pathlib import Path
import shutil
import tempfile

import av
import cv2

from .config import load_profile, load_song
from .contracts import decimal_bpm


def check_video(path, resolution, fps=None):
    if not Path(path).is_file():
        raise ValueError(f'video not found: {path}')
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError('input contains no video stream')
            stream = container.streams.video[0]
            if (stream.width, stream.height) != tuple(resolution):
                raise ValueError(f'video resolution must be {resolution[0]}x{resolution[1]} for this profile')
            if fps is not None and (stream.average_rate is None or abs(float(stream.average_rate)-fps) > .01):
                raise ValueError(f'this profile requires {fps:g} FPS; received {stream.average_rate}')
    except av.FFmpegError as exc:
        raise ValueError(f'cannot inspect video: {exc}') from exc


def initialize(directory, *, video, bpm, templates, title='Untitled', difficulty='SHD'):
    decimal_bpm(bpm)
    directory, video, templates = Path(directory).resolve(), Path(video).resolve(), Path(templates).resolve()
    check_video(video, (1920, 1080), 60)
    if directory.exists():
        raise ValueError(f'project directory already exists: {directory}')
    profile = files('ez2rc.v2').joinpath('resources/ez2on_5k_1080p_speed8.toml').read_text()
    names = [f'note_{color}{suffix}.png' for color in ('white', 'cyan') for suffix in ('', '_lnhead', '_lntail')]
    for name in names:
        path = templates / name
        if not path.is_file():
            raise ValueError(f'missing template asset: {path}')
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (22, 82) or image.std() == 0:
            raise ValueError(f'template must be a nonuniform 82x22 PNG: {path}')
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ez2rc-project-', dir=directory.parent))
    try:
        (temporary / 'templates').mkdir()
        for name in names:
            shutil.copyfile(templates / name, temporary / 'templates' / name)
        profile = profile.replace('../../skins/ez2on/5k/', 'templates/')
        (temporary / 'profile.toml').write_text(profile, encoding='utf-8')
        load_profile(temporary / 'profile.toml')
        # JSON strings are valid TOML basic strings for these paths and labels.
        quote = lambda value: json.dumps(str(value), ensure_ascii=False)
        song = (f'schema_version = 1\n[input]\nvideo = {quote(video)}\nprofile = "profile.toml"\n'
                f'[metadata]\ntitle = {quote(title)}\ndifficulty = {quote(difficulty)}\n'
                f'[timing]\nmode = "fixed"\nbpm = {quote(bpm)}\nmeasure_beats = [4, 1]\n')
        (temporary / 'song.toml').write_text(song, encoding='utf-8')
        load_song(temporary / 'song.toml')
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory / 'song.toml'
