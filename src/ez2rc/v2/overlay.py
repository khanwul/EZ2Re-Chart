"""Render the edited draft in recording coordinates, preserving source frame times."""
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import os
import time
from tempfile import TemporaryDirectory

import av
import cv2
import numpy as np

from .capture import TimestampClock
from .contracts import read_ratio
from .motion import STATUS, open_motion, timeline
from .observation import template_shapes
from .progress import frame_progress
from .photos import _korean_font
from .storage import digest, file_hash, read_json
from .window import in_interval
from .witness_box import OUTLINE_PINK


def frame_origins(clock, grid, scale):
    """Y of beat zero, anchored locally on each connected scroll component."""
    if grid['status'] != 'supported' or len(grid['hypotheses']) != 1:
        return {}
    beats = {a['marker_id']: float(read_ratio(a['beat']))
             for a in grid['hypotheses'][0]['assignments']}
    direct = defaultdict(list)
    for track in grid['tracks']:
        if track['id'] in beats:
            for row in track['observations']:
                direct[row['frame_id']].append(row['y_px'] + beats[track['id']] * scale)

    roots = {}
    anchors = defaultdict(list)
    scroll = dict(zip(map(int, clock['frame_id']), clock['scroll_px']))
    for fid, state, parent, group in zip(clock['frame_id'], clock['status'],
                                          clock['reference_frame'], clock['repeat_group']):
        root = fid
        if state == STATUS['observed']:
            root = roots.get(parent, fid)
        elif state == STATUS['repeated']:
            root = roots.get(group, fid)
        roots[fid] = root
        if fid in direct:
            anchors[root].append((fid, float(np.median(direct[fid])) - scroll[fid]))

    origins = {}
    for fid, root in roots.items():
        points = anchors[root]
        if points:
            i = bisect_left(points, (fid,))
            _, offset = min(points[max(0, i-1):i+1], key=lambda p: abs(p[0]-fid))
            origins[fid] = scroll[fid] + offset
    return origins


class Outlines:
    def __init__(self, draft, profile, shapes, alignment, scale):
        dx, dy = alignment
        self.rows = []
        self.omitted = 0
        for note in draft['notes']:
            lane = note['lane']
            head = 'head' if note['type'] == 'longnote' else 'tap'
            if note['start_beat'] is None or head == 'head' and note['end_beat'] is None:
                self.omitted += 1
                continue

            def distance(key, kind):
                calibration = profile['calibration'][kind]
                offset = calibration['offset_px'] if calibration['status'] == 'verified' else 0
                offset = profile.get('draft_offsets_px', {}).get(kind, offset)
                return float(read_ratio(note[key])) * scale + offset

            # Both beat coordinates describe template tops after undoing the
            # endpoint offset. The hold spans the tail top to the head bottom.
            start = distance('start_beat', head)
            end = distance('end_beat', 'tail') if head == 'head' else start
            height, width = shapes[lane][head]
            left, top, right, bottom = profile['lanes'][lane]['roi']
            center = (left + right - 1) / 2 + dx
            x = round(center - (width - 1) / 2)
            self.rows.append((x, x + width - 1, top + dy, bottom + dy,
                              dy - end, dy - start + height - 1))
        rows = np.asarray(self.rows).reshape(-1, 6)
        self.first_visible = rows[:, 2] - rows[:, 5]
        self.last_visible = rows[:, 3] - rows[:, 4]

    def draw(self, image, origin):
        for i in np.flatnonzero((self.first_visible <= origin) & (origin < self.last_visible)):
            left, right, top, bottom, upper, lower = self.rows[i]
            # Crop the drawing surface so a hold crossing the viewport edge
            # stays open there instead of acquiring a false head or tail.
            top, bottom = max(0, top), min(image.shape[0], bottom)
            if bottom <= top:
                continue
            roi = image[top:bottom]
            y1 = max(-4, round(origin + upper) - top)
            y2 = min(bottom - top + 4, round(origin + lower) - top)
            for color, width in (((28, 14, 30), 4), (OUTLINE_PINK, 2)):
                cv2.rectangle(roi, (left, y1), (right, y2), color, width)


def _banner(text, fallback, width, height):
    font = _korean_font()
    strip = np.full((36, 540, 3), 28, dtype=np.uint8)
    if font:
        cv2.putText(strip, text, (10, 26), (250, 250, 250), font, 20)
    else:
        cv2.putText(strip, fallback, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (250, 250, 250), 1)
    factor = min(1, width / 1080, height / 600)
    return cv2.resize(strip, (max(1, round(540*factor)), max(1, round(36*factor))))


def _encode(source, destination, outlines, origins, capture, progress, *, total_frames=None):
    frame_count = placed = 0
    with av.open(str(source)) as recording, av.open(str(destination), 'w', options={'movflags': '+faststart'}) as output:
        video = recording.streams.video[0]
        encoded = output.add_stream('libx264', rate=video.average_rate or 30)
        encoded.width, encoded.height = video.width, video.height
        encoded.pix_fmt = 'yuv420p' if not (video.width % 2 or video.height % 2) else 'yuv444p'
        encoded.codec_context.time_base = video.time_base
        encoded.codec_context.thread_count = 4
        encoded.options = {'preset': 'veryfast', 'crf': '18'}
        clock = TimestampClock(video.average_rate, capture.get('timestamp_policy', 'strict'))
        audios = {}
        transcode = set()
        for stream in recording.streams.audio:
            if stream.codec_context.name in output.supported_codecs:
                audios[stream.index] = output.add_stream_from_template(stream)
            else:
                audio = output.add_stream('aac', rate=48000)
                layout = stream.codec_context.layout
                if layout.name.endswith(' channels'):
                    audio.layout = 'mono' if layout.nb_channels == 1 else 'stereo'
                else:
                    audio.layout = layout
                audios[stream.index] = audio
                transcode.add(stream.index)
        active = _banner('분홍 테두리 · 현재 초안의 예상 위치', 'Pink outlines: draft positions', video.width, video.height)
        unknown = _banner('위치 대조 불가', 'Position unavailable', video.width, video.height)
        duration_ms = None
        if video.duration:
            duration_ms = float(video.duration * video.time_base * 1000)
        elif recording.duration:
            duration_ms = recording.duration / 1000

        def report_frames(timestamp_ms, *, complete=False):
            progress(frame_progress(frame_count, timestamp_ms, 0, duration_ms, complete=complete,
                                    label='대조 영상 · 프레임 처리', total_frames=frame_count if complete else total_frames))

        if progress:
            report_frames(0)
            next_update = time.monotonic() + 2
        for packet in recording.demux(video, *recording.streams.audio):
            if packet.stream.index in audios:
                audio = audios[packet.stream.index]
                if packet.stream.index in transcode:
                    for samples in packet.decode():
                        for compressed in audio.encode(samples):
                            output.mux(compressed)
                elif packet.size:
                    packet.stream = audio
                    output.mux(packet)
                continue
            for frame in packet.decode():
                if frame.is_corrupt:
                    raise ValueError(f'원영상 프레임 {frame_count}이 손상되었습니다.')
                stamp = clock.stamp(frame_count, frame.pts, frame.time_base, frame.duration)
                if frame_count == 0:
                    first_pts = frame.pts if frame.pts is not None else (video.start_time or 0)
                pts = frame.pts
                if stamp.source == 'fps_fallback':
                    pts = first_pts + round(stamp.timestamp_ms / (1000 * video.time_base))
                image = frame.to_ndarray(format='bgr24')
                position = origins.get(frame_count)
                banner = unknown
                if position is not None:
                    ms, origin = position
                    if abs(stamp.timestamp_ms - ms) > .001:
                        raise ValueError('원영상과 분석 프레임의 시각이 일치하지 않습니다.')
                    outlines.draw(image, origin)
                    banner = active
                    placed += 1
                h, w = banner.shape[:2]
                image[-h:, :w] = banner
                rendered = av.VideoFrame.from_ndarray(image, format='bgr24')
                rendered.pts, rendered.time_base = pts, frame.time_base or video.time_base
                rendered.duration = frame.duration
                for compressed in encoded.encode(rendered):
                    output.mux(compressed)
                frame_count += 1
                if progress and time.monotonic() >= next_update:
                    report_frames(stamp.timestamp_ms)
                    next_update = time.monotonic() + 2
        if not frame_count or frame_count < capture.get('decoded_frames', 0):
            raise ValueError('원영상 프레임이 부족합니다.')
        if progress:
            report_frames(stamp.timestamp_ms, complete=True)
            progress('대조 영상 · 오디오와 영상 마무리 중')
        for index in transcode:
            for compressed in audios[index].encode(None):
                output.mux(compressed)
        for compressed in encoded.encode(None):
            output.mux(compressed)
    return placed, frame_count


def write_overlay(destination, draft, analysis_dir, *, video=None, progress=None):
    destination = Path(destination)
    if destination.suffix.lower() != '.mp4' or destination.exists():
        raise ValueError('저장할 새 MP4 경로를 --output으로 지정하세요.')
    source = draft['source']
    recording = Path(video or source['video'])
    if progress:
        progress('대조 영상 · 원본 영상 확인 중')
    if not recording.is_file() or file_hash(recording) != source['video_sha256']:
        raise ValueError('초안과 같은 원본 영상을 --video로 지정하세요.')
    if progress:
        progress('대조 영상 · 분석 자료 확인 및 노트 위치 계산 중')
    artifacts = analysis_dir / 'artifacts'
    obs = read_json(artifacts/'observations/manifest.json', 'ez2rc.stage-cache')
    motion = open_motion(artifacts/'motion')
    grid = read_json(artifacts/'barline-grid.json', 'ez2rc.barline-grid')
    if (obs['manifest_sha256'] != digest({k: v for k, v in obs.items() if k != 'manifest_sha256'})
            or obs['cache_id'] != source['observation_cache_id']
            or motion.id != source['motion_cache_id']
            or motion.manifest['key']['parents'] != [obs['cache_id']]
            or grid['observation_cache_id'] != obs['cache_id'] or grid['motion_cache_id'] != motion.id):
        raise ValueError('초안과 위치 분석 자료가 일치하지 않습니다.')
    meta = obs['metadata']
    shapes = template_shapes(obs['key']['parameters'])
    projection = draft['projection']
    bounds = projection.get('scale_px_per_beat')
    if not bounds or not shapes or meta.get('alignment_offset_px') is None:
        raise ValueError('영상 대조에 필요한 박자 간격 또는 레인 위치 정보가 없습니다.')
    scale = sum(float(read_ratio(v)) for v in bounds) / 2
    clock = timeline(motion)
    positions = frame_origins(clock, grid, scale)
    window = source.get('analysis_window', {}).get('requested', {'start_ms': 0, 'end_ms': None})
    included = in_interval(clock['timestamp_ms'], window)
    origins = {fid: (ms, positions[fid]) for fid, ms in zip(clock['frame_id'][included], clock['timestamp_ms'][included])
               if fid in positions}
    if not origins:
        raise ValueError('마디선과 스크롤 위치를 연결할 수 있는 프레임이 없습니다.')
    outlines = Outlines(draft, meta['profile'], shapes, meta['alignment_offset_px'], scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temp = Path(directory)/'overlay.mp4'
        try:
            # A bounded extraction may have stopped decoding before the movie ended.
            observation = source.get('analysis_window', {}).get('observation', {})
            total_frames = meta['capture'].get('decoded_frames') if (
                'end_ms' in observation and observation['end_ms'] is None) else None
            placed, total = _encode(recording, temp, outlines, origins, meta['capture'], progress,
                                    total_frames=total_frames)
        except av.FFmpegError as exc:
            raise ValueError(f'대조 영상을 인코딩할 수 없습니다: {exc}') from exc
        os.link(temp, destination)
    return placed, total, outlines.omitted
