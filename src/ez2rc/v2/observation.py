"""Unpaired, score-preserving visual observations with bounded chunk memory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .capture import VideoReader, VERSION as CAPTURE_VERSION, decoder_parameters, validate_timeline
from .config import AlgorithmConfig, Profile, SongConfig
from .storage import Cache, CacheError, CacheWriter, digest, file_hash, stage_key

VERSION = "visual-observations-2"
KINDS = ("tap", "head", "tail")


def channel(bgr, name):
    if name == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if name == "max_bg":
        return np.maximum(bgr[:, :, 0], bgr[:, :, 1])
    return bgr[:, :, {"blue": 0, "green": 1, "red": 2}[name]]


def runs(mask):
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def crop(image, roi, offset=(0, 0)):
    x1, y1, x2, y2 = roi
    dx, dy = offset
    if x1 + dx < 0 or y1 + dy < 0 or x2 + dx > image.shape[1] or y2 + dy > image.shape[0]:
        raise ValueError("aligned ROI falls outside the recording")
    return image[y1 + dy:y2 + dy, x1 + dx:x2 + dx]


def alignment_offset(image, profile: Profile):
    if (image.shape[1], image.shape[0]) != profile.resolution:
        raise ValueError("video resolution does not match recording profile")
    left, top, right, bottom = profile.alignment_band
    limit = profile.alignment_max_shift
    x1, y1 = max(0, left - limit), max(0, top - limit)
    x2, y2 = min(image.shape[1], right + limit), min(image.shape[0], bottom + limit)
    blue, green, red = [c.astype(np.int16) for c in cv2.split(image[y1:y2, x1:x2])]
    mask = (blue > 100) & (blue > green + 20) & (blue > red + 40)

    def locate(values, fraction, length, expected):
        candidates = [(int(a), int(b)) for a, b in runs(values >= fraction)
                      if abs((b - a) - length) <= 2]
        if not candidates:
            raise ValueError("calibrated judgment band not found or size mismatch")
        return min(candidates, key=lambda r: (abs(r[0] - expected), r[0]))

    row = locate(mask.sum(axis=1), .8 * (right - left), bottom - top, top - y1)
    col = locate(mask[row[0]:row[1]].sum(axis=0), .8 * (row[1] - row[0]), right - left, left - x1)
    dx, dy = x1 + col[0] - left, y1 + row[0] - top
    # The band is antialiased by up to two pixels, as in the geometric probe.
    dx, dy = (0 if abs(dx) <= 2 else dx), (0 if abs(dy) <= 2 else dy)
    if abs(dx) > limit or abs(dy) > limit:
        raise ValueError("panel translation exceeds profile limit")
    for roi in [profile.beat_roi, *[r for lane in profile.lanes for r in (lane.roi, lane.match_roi)]]:
        crop(image, roi, (dx, dy))
    return dx, dy


def probe_alignment(song: SongConfig, profile: Profile, algo: AlgorithmConfig):
    # Probe from the original origin even when processing a later subrange.
    reader = VideoReader(song.video, end_ms=algo.alignment_probe_ms,
                         timestamp_policy=algo.timestamp_policy)
    error = None
    iterator = iter(reader)
    try:
        for frame in iterator:
            if (frame.bgr.shape[1], frame.bgr.shape[0]) != profile.resolution:
                raise ValueError("video resolution does not match recording profile")
            try:
                return alignment_offset(frame.bgr, profile)
            except ValueError as exc:
                error = exc
    finally:
        iterator.close()
    raise ValueError(f"recording alignment failed: {error}")


def candidate_seeds(projection, edge, height, algo):
    """Union of brightness boundaries and edge peaks; no winning-type gate."""
    seeds = set()
    for start, end in runs(projection >= algo.brightness_threshold):
        seeds.add(int(start))
        seeds.add(max(0, int(end) - height))
    peaks = ((edge >= algo.edge_threshold)
             & (edge >= np.r_[0, edge[:-1]]) & (edge > np.r_[edge[1:], 0]))
    for y in np.flatnonzero(peaks):
        seeds.add(int(y))
        seeds.add(max(0, int(y) - height + 1))
    # Merge overlapping seed neighborhoods deterministically. All four score
    # channels and full brightness/edge profiles remain available in the cache.
    merged = []
    for seed in sorted(seeds):
        if not merged or seed - merged[-1] > max(1, algo.search_radius_px):
            merged.append(seed)
    return merged


def observe_frame(frame, profile, algo, offset):
    projections, edges, indices, scores, positions, flags, barlines = [], [], [], [], [], [], []
    signature = hashlib.sha256()
    dx, dy = offset
    for lane_id, lane in enumerate(profile.lanes):
        image = crop(frame.bgr, lane.roi, offset)
        signature.update(image.tobytes())
        gray = channel(image, lane.channel).astype(np.float32)
        projection = gray.mean(axis=1)
        edge = np.r_[np.zeros(1, np.float32), np.abs(np.diff(gray, axis=0)).mean(axis=1)]
        # A top-censored brightness run can match an endpoint template well
        # inside the search window. Only call it continuation while BOTH the
        # brightness and the lack of vertical edges persist from the viewport
        # top through the matched patch and its following row. A nearby real
        # endpoint with an observed edge must remain usable. Use the existing
        # seed thresholds, not a new similarity or note-length cutoff.
        breaks = np.flatnonzero((projection < algo.brightness_threshold)
                                | (edge >= algo.edge_threshold))
        top_continuation_end = int(breaks[0]) if len(breaks) else len(projection)
        projections.append(projection)
        edges.append(edge)
        matching = np.ascontiguousarray(crop(frame.bgr, lane.match_roi, offset))
        th = min(t.shape[0] for t in lane.templates.values())
        for seed in candidate_seeds(projection, edge, th, algo):
            candidate_scores, xy, candidate_flags = [], [], []
            for kind in KINDS:
                template = lane.templates[kind]
                h, w = template.shape[:2]
                lo = max(0, min(seed - algo.search_radius_px, matching.shape[0] - h))
                hi = min(matching.shape[0] - h, seed + algo.search_radius_px)
                score_map = cv2.matchTemplate(matching[lo:hi + h], template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(score_map)
                x, y = lane.match_roi[0] + location[0], lane.roi[1] + lo + location[1]
                candidate_scores.append(score)
                xy.append((x, y))
                top_continuation = (seed < algo.search_radius_px
                                    and y - lane.roi[1] + h < top_continuation_end)
                # bit 0: clipped search/template; bit 1: judgment-effect zone;
                # bit 2: visibility/occlusion unknown (not assumed unobscured).
                candidate_flags.append(int(y <= lane.roi[1] or y + h >= lane.roi[3]
                                           or top_continuation)
                                       | (2 if y + h >= profile.effect_top else 0) | 4)
            body = float(projection[seed:min(len(projection), seed + th)].mean()) / 255.0
            indices.append((frame.stamp.frame_id, lane_id, seed))
            scores.append((*candidate_scores, body))
            positions.append(xy)
            flags.append(candidate_flags)
    projections = np.asarray(projections)
    common = np.median(projections, axis=0)
    possible = ((common >= algo.barline_min_brightness) & (common <= algo.barline_max_brightness)
                & ((projections >= algo.barline_min_brightness).sum(axis=0) >= max(1, profile.lane_count - 1)))
    for start, end in runs(possible):
        if end - start <= algo.barline_max_thickness:
            barlines.append((frame.stamp.frame_id, profile.lanes[0].roi[1] + (start + end - 1) / 2,
                             float(common[start:end].mean())))
    return {
        "projection": projections.astype(np.float16),
        "edge": np.asarray(edges, dtype=np.float16),
        "candidate_index": np.asarray(indices, dtype=np.int64).reshape(-1, 3),
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1, 4),
        "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 3, 2),
        "flags": np.asarray(flags, dtype=np.uint8).reshape(-1, 3),
        "barlines": np.asarray(barlines, dtype=np.float64).reshape(-1, 3),
        "beat_brightness": float(crop(frame.bgr, profile.beat_roi, offset)[:, :, 1].mean()),
        "signature": signature.digest(),
    }


def cache_keys(song, profile, algo, video_hash):
    capture = stage_key("capture", CAPTURE_VERSION, {
        "video_sha256": video_hash, "start_ms": song.start_ms, "end_ms": song.end_ms,
        "timestamp_policy": algo.timestamp_policy, **decoder_parameters()})
    observation = stage_key("observation", VERSION, {
        "profile": profile.snapshot, "asset_sha256": profile.asset_hashes,
        "algorithm": algo.observation_parameters(), "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }, (digest(capture),))
    return capture, observation


def collect(song: SongConfig, profile: Profile, algo: AlgorithmConfig, root: Path):
    video_hash = file_hash(song.video)
    capture_key, observation_key = cache_keys(song, profile, algo, video_hash)
    capture_path = root / "capture" / digest(capture_key)
    observation_path = root / "observation" / digest(observation_key)
    try:
        capture_cache = Cache(capture_path, capture_key)
        observations = open_observations(observation_path)
        if observations.manifest["key"] != observation_key:
            raise CacheError("observation key mismatch")
        return capture_cache, observations, True
    except CacheError:
        pass
    offset = probe_alignment(song, profile, algo)
    reader = VideoReader(song.video, start_ms=song.start_ms, end_ms=song.end_ms,
                         timestamp_policy=algo.timestamp_policy)
    buffer = []
    previous_signature, repeat_group = None, None
    count = candidate_count = repeated = 0
    with CacheWriter(root, capture_key) as capture_writer, CacheWriter(root, observation_key) as writer:
        def flush():
            stamps = [f.stamp for f, _, _ in buffer]
            timeline = {
                "frame_id": np.array([s.frame_id for s in stamps], np.int64),
                "pts": np.array([s.pts if s.pts is not None else 0 for s in stamps], np.int64),
                "pts_valid": np.array([s.pts is not None for s in stamps], np.bool_),
                "time_base": np.array([[s.time_base.numerator, s.time_base.denominator] for s in stamps], np.int64),
                "timestamp_ms": np.array([s.timestamp_ms for s in stamps], np.float64),
                "duration_ms": np.array([s.duration_ms if s.duration_ms is not None else -1 for s in stamps], np.float64),
                "fps_fallback": np.array([s.source == "fps_fallback" for s in stamps], np.bool_),
            }
            capture_writer.append(**timeline)
            arrays = {k: np.concatenate([o[k] for _, o, _ in buffer]) for k in (
                "candidate_index", "scores", "positions", "flags", "barlines")}
            arrays.update({k: np.stack([o[k] for _, o, _ in buffer]) for k in ("projection", "edge")})
            arrays.update(timeline)
            arrays["beat_brightness"] = np.array([o["beat_brightness"] for _, o, _ in buffer], np.float32)
            arrays["repeat_candidate_group"] = np.array([g for _, _, g in buffer], np.int64)
            writer.append(**arrays)
            buffer.clear()

        for frame in reader:
            if (frame.bgr.shape[1], frame.bgr.shape[0]) != profile.resolution:
                raise ValueError("video resolution changed during decode")
            observation = observe_frame(frame, profile, algo, offset)
            if observation["signature"] != previous_signature:
                repeat_group = frame.stamp.frame_id
            else:
                repeated += 1
            previous_signature = observation["signature"]
            count += 1
            candidate_count += len(observation["scores"])
            # Keep only the timestamp alongside features, never full frames.
            frame.bgr = np.empty((0,), dtype=np.uint8)
            buffer.append((frame, observation, repeat_group))
            if len(buffer) >= algo.chunk_frames:
                flush()
        if buffer:
            flush()
        # Detect external input mutation before publishing derived evidence.
        if file_hash(song.video) != video_hash:
            raise ValueError("video changed during analysis")
        capture_cache = capture_writer.commit(reader.metadata)
        metadata = {
            "video_sha256": video_hash, "capture": reader.metadata,
            "capture_cache_id": capture_cache.id, "alignment_offset_px": list(offset),
            "profile": profile.snapshot, "profile_path": str(song.profile),
            "profile_sha256": file_hash(song.profile), "asset_sha256": profile.asset_hashes,
            "frame_count": count, "candidate_count": candidate_count,
            "exact_repeated_images": repeated,
            "repeat_policy": "provisional exact lane-image equality; motion confirmation pending",
            "score_columns": ["tap_similarity", "head_similarity", "tail_similarity", "body_brightness_fraction"],
            "position_kinds": list(KINDS), "position_unit": "aligned pixels, template top-left",
            "position_error_px": None, "position_error_reason": "not independently calibrated",
            "candidate_id": "f{frame_id}:l{lane}:y{seed_row}",
            "flag_bits": {"clipped": 1, "judgment_effect": 2, "occlusion_unknown": 4},
            "clipping_policy": "template touches viewport edge, or top-truncated seed search "
                               "matches an edgeless bright continuation through the following row",
            "barline_columns": ["frame_id", "aligned_y_px", "brightness"],
            "duration_missing_sentinel": -1, "time_unit": "ms since original first frame PTS",
        }
        observations = writer.commit(metadata, validate=open_observations)
    return capture_cache, observations, False


def _open_observations(path: Path) -> Cache:
    cache = Cache(path)
    key = cache.manifest["key"]
    if key["stage"] != "observation" or key["algorithm_version"] != VERSION:
        raise CacheError("unsupported observation algorithm version")
    previous_id, previous_ms = -1, -1.0
    count = 0
    metadata = cache.manifest["metadata"]
    lane_count = len(metadata["profile"]["lanes"])
    roi = metadata["profile"]["lanes"][0]["roi"]
    height = roi[3] - roi[1]
    candidates_count = 0
    for chunk in cache.chunks():
        ids, timestamps = chunk["frame_id"], chunk["timestamp_ms"]
        n = len(ids)
        if (n == 0 or ids.ndim != 1 or timestamps.shape != (n,)
                or ids[0] < 0 or previous_id >= 0 and ids[0] != previous_id + 1
                or timestamps[0] < previous_ms
                or np.any(np.diff(ids) != 1) or np.any(np.diff(timestamps) < 0)
                or not np.isfinite(timestamps).all()):
            raise CacheError("invalid observation frame timeline")
        try:
            validate_timeline(chunk, metadata["capture"])
        except (ValueError, KeyError, TypeError) as exc:
            raise CacheError(f"invalid observation timeline: {exc}") from exc
        candidates = chunk["candidate_index"]
        m = len(candidates)
        if (candidates.shape != (m, 3) or chunk["scores"].shape != (m, 4)
                or chunk["positions"].shape != (m, 3, 2) or chunk["flags"].shape != (m, 3)
                or not np.isfinite(chunk["scores"]).all() or not np.isfinite(chunk["positions"]).all()
                or candidates.dtype.kind not in "iu" or np.any(~np.isin(candidates[:, 0], ids))
                or np.any(candidates[:, 1] < 0) or np.any(candidates[:, 1] >= lane_count)
                or np.any(candidates[:, 2] < 0) or np.any(candidates[:, 2] >= height)
                or len(np.unique(candidates, axis=0)) != m):
            raise CacheError("invalid observation candidates")
        if (chunk["projection"].shape != (n, lane_count, height)
                or chunk["edge"].shape != (n, lane_count, height)
                or chunk["beat_brightness"].shape != (n,)
                or chunk["repeat_candidate_group"].shape != (n,)
                or np.any(chunk["repeat_candidate_group"] > ids)
                or np.any(chunk["repeat_candidate_group"] < 0)
                or any(not np.isfinite(chunk[name]).all() for name in ("projection", "edge", "beat_brightness", "barlines"))):
            raise CacheError("invalid observation signals")
        previous_id, previous_ms = int(ids[-1]), float(timestamps[-1])
        count += n
        candidates_count += m
    if count != metadata["frame_count"] or candidates_count != metadata["candidate_count"]:
        raise CacheError("observation frame count mismatch")
    return cache


def open_observations(path: Path) -> Cache:
    try:
        return _open_observations(path)
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        raise CacheError(f"invalid observations {path}: {exc}") from exc
