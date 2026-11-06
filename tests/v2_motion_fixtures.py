"""Procedural feature observations with exact scroll and long-body geometry."""
from pathlib import Path
import numpy as np
from ez2rc.v2.storage import CacheWriter, stage_key


def observations(root: Path, scroll, *, chunk_frames=4, blank=(), one_lane=(),
                 duplicate_seeds=False, flags=0, static=False, times=None,
                 close=False, calibration=None, near_repeat=False):
    scroll = np.array(scroll, dtype=float)
    times = np.arange(len(scroll)) * 16. if times is None else np.asarray(times, float)
    profile = {"judge_y": 150., "effect_top": 140.,
               "lanes": [{"roi": [i*20, 0, i*20+20, 160]} for i in range(3)],
               "calibration": {k: {"status": "unverified", "detection_reference": "template_top", "version": "test"}
                               for k in ("tap", "head", "tail")}}
    taps = ([42, 101, 183, 267, 336, 439, 524, 611], [23, 119, 172, 295, 397, 478, 579, 662])
    if calibration:
        profile['calibration'] = calibration
    all_frames = []
    group = 100
    for i, s in enumerate(scroll):
        fid = 100 + i
        if i == 0 or s != scroll[i-1]:
            group = fid
        index, scores, pos, marks = [], [], [], []
        projection = np.zeros((3, 160), np.float16)
        def endpoint(lane, y, kind, fl=flags, strength=.99):
            if not 1 < y < 159:
                return
            row = int(y)
            kinds = [.15, .2, .1, .4]
            kinds[kind] = strength
            index.append((fid, lane, row))
            scores.append(kinds)
            pos.append([[lane*20, y] for _ in range(3)])
            marks.append([fl]*3)
            projection[lane, row:min(row+3, 160)] = 220
            if duplicate_seeds:
                index.append((fid, lane, row+1))
                scores.append(kinds)
                pos.append(pos[-1])
                marks.append([fl]*3)
        if i not in blank:
            for lane in range(2):
                if i in one_lane and lane:
                    continue
                for q in taps[lane]:
                    endpoint(lane, s+150-q, 0)
            if i not in one_lane:
                q = s+150-np.arange(160)
                projection[2, (q > 90) & (q < 490)] = 100
                endpoint(2, s+150-90, 1)
                endpoint(2, s+150-490, 2)
            if close:
                endpoint(0, s+150-107, 0, strength=.45)
            # Fixed judgment head and static UI must not vote for scroll.
            for lane in range(3):
                endpoint(lane, 145, 1, fl=2)
                if static:
                    endpoint(lane, 12, 0)
                    endpoint(lane, 32, 0)
        edge = np.abs(np.diff(projection, axis=1, prepend=0)).astype(np.float16)
        if near_repeat:
            projection[:, 0] += i % 2
        all_frames.append({"frame_id": np.array([fid]), "timestamp_ms": np.array([times[i]]),
                           "repeat_candidate_group": np.array([fid if near_repeat else group]),
                           "projection": projection[None], "edge": edge[None],
                           "candidate_index": np.array(index, np.int64).reshape(-1, 3),
                           "scores": np.array(scores, np.float32).reshape(-1, 4),
                           "positions": np.array(pos, np.float32).reshape(-1, 3, 2),
                           "flags": np.array(marks, np.uint8).reshape(-1, 3)})
    key = stage_key("observation", "procedural-motion-test", {"scroll": scroll.tolist(),
                    "chunk_frames": chunk_frames, "blank": list(blank), "one_lane": list(one_lane),
                    "duplicates": duplicate_seeds, "flags": flags, "static": static, "times": times.tolist(),
                    "close": close, "calibration": calibration, "near_repeat": near_repeat})
    with CacheWriter(root, key) as writer:
        for start in range(0, len(all_frames), chunk_frames):
            batch = all_frames[start:start+chunk_frames]
            writer.append(**{k: np.concatenate([f[k] for f in batch]) for k in batch[0]})
        return writer.commit({"profile": profile, "frame_count": len(scroll)})


def evidence_arrays(cache):
    chunks = list(cache.chunks())
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    order = np.lexsort((data["index"][:, 1], data["index"][:, 0]))
    for k in ("index", "score_sum", "weight_sum", "visibility_weight", "group_count", "error_px", "frame_min", "frame_max"):
        data[k] = data[k][order]
    return data
