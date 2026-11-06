"""Robust common displacement from moving, ordered endpoint observations.

The coordinate origin is the first observed frame, not a musical/time origin.
Errors are conservative algorithmic envelopes, not calibrated confidence levels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import MotionConfig
from .storage import Cache, CacheError, CacheWriter, digest, stage_key

VERSION = "common-scroll-3"
STATUS = {"origin": 0, "observed": 1, "single_lane": 2, "ambiguous": 3,
          "extrapolated": 4, "unresolved": 5, "repeated": 6}


@dataclass(frozen=True)
class ScrollPoint:
    frame_id: int
    timestamp_ms: float
    scroll_px: float
    error_px: float
    repeated_group: int | None
    observation_ids: tuple[str, ...]


def frames(observations):
    """Stream one feature frame; original IDs survive storage chunk changes."""
    for chunk in observations.chunks():
        indices = chunk["candidate_index"]
        for i, fid in enumerate(chunk["frame_id"]):
            selected = indices[:, 0] == fid
            yield {"id": int(fid), "ms": float(chunk["timestamp_ms"][i]),
                   "group": int(chunk["repeat_candidate_group"][i]),
                   "projection": chunk["projection"][i], "edge": chunk["edge"][i],
                   **{k: chunk[k][selected] for k in ("candidate_index", "scores", "positions", "flags")}}


def landmarks(frame, profile, config):
    result = []
    for lane in range(len(profile["lanes"])):
        for kind in range(3):
            valid = ((frame["candidate_index"][:, 1] == lane)
                     & (frame["scores"][:, kind] >= config.min_similarity)
                     & ((frame["flags"][:, kind] & 11) == 0)
                     & (frame["positions"][:, kind, 1] < profile["effect_top"]))
            indices = np.flatnonzero(valid)
            kept = []
            for i in sorted(indices, key=lambda i: (-frame["scores"][i, kind], int(i))):
                y = float(frame["positions"][i, kind, 1])
                if all(abs(y - p[3]) > config.match_tolerance_px for p in kept):
                    kept.append((lane, kind, int(frame["candidate_index"][i, 2]), y,
                                 float(frame["scores"][i, kind])))
            result.extend(sorted(kept, key=lambda p: p[3]))
    return np.asarray(result, np.float64).reshape(-1, 5)


def match(previous, current, limit, config):
    """Alternate displacement hypotheses and one-to-one ordered associations.

    Fixed landmarks cannot vote for moving scroll. Each lane has at most one
    unit of influence, regardless of its candidate count or template types.
    Similar-cost displacement alternatives are retained by the caller.
    """
    pairs = []
    tol = config.match_tolerance_px
    for lane, kind in sorted({tuple(p[:2]) for p in previous}):
        a = previous[(previous[:, 0] == lane) & (previous[:, 1] == kind)]
        b = current[(current[:, 0] == lane) & (current[:, 1] == kind)]
        if not len(b):
            continue
        stationary = np.abs(b[:, 3, None] - a[None, :, 3]) <= min(.75, tol / 2)
        # Exclude UI/static effects outside the judgment band too.
        moving_a, moving_b = ~stationary.any(axis=0), ~stationary.any(axis=1)
        for pa in a[moving_a]:
            for pb in b[moving_b]:
                d = pb[3] - pa[3]
                if .75 < d <= limit:
                    pairs.append((lane, kind, pa[2], pb[2], d, min(pa[4], pb[4]), pa[3], pb[3]))
    if not pairs:
        return []
    pairs = np.asarray(pairs)
    # Broad votes propose shifts; robust, ordered links below score each mode.
    bins = np.rint(pairs[:, 4] / tol).astype(int)
    votes = {}
    for b in np.unique(bins):
        close = np.abs(bins - b) <= 1
        votes[int(b)] = sum(float(pairs[close & (pairs[:, 0] == lane), 5].max())
                            for lane in np.unique(pairs[close, 0]))
    proposals = sorted(votes, key=lambda b: (-votes[b], b))[:16]
    models = []
    for proposal in proposals:
        shift = proposal * tol
        selected = []
        for _ in range(3):
            near = pairs[np.abs(pairs[:, 4] - shift) <= tol]
            selected = []
            for key in sorted({tuple(p[:2]) for p in near}):
                group = near[(near[:, 0] == key[0]) & (near[:, 1] == key[1])]
                links = []
                for p in sorted(group, key=lambda p: (abs(p[4] - shift), -p[5], p[6], p[7])):
                    if any(p[2] == q[2] or p[3] == q[3]
                           or (p[6] - q[6]) * (p[7] - q[7]) <= 0 for q in links):
                        continue
                    links.append(p)
                selected.extend(links)
            if not selected:
                break
            # Median within lanes then across lanes bounds dense-lane influence.
            links = np.asarray(selected)
            shift = float(np.median([np.median(links[links[:, 0] == lane, 4])
                                     for lane in np.unique(links[:, 0])]))
        if not selected or any(abs(shift - m["delta"]) <= tol for m in models):
            continue
        links = np.asarray(selected)
        residual = np.abs(links[:, 4] - shift)
        lanes = len(np.unique(links[:, 0]))
        quality = sum(float(np.max(links[links[:, 0] == lane, 5]
                                   * (1 - .25 * np.minimum(residual[links[:, 0] == lane] / tol, 1))))
                      for lane in np.unique(links[:, 0]))
        models.append({"delta": shift, "quality": quality, "lanes": lanes,
                       "error": max(.5, float(np.median(residual))), "links": links})
    return sorted(models, key=lambda m: (-m["quality"], -m["lanes"], m["delta"]))


def cache_key(observations, config):
    return stage_key("motion", VERSION, asdict(config), (observations.id,))


def estimate(observations: Cache, config: MotionConfig, root: Path):
    key = cache_key(observations, config)
    try:
        return open_motion(root / "motion" / digest(key), key), True
    except CacheError:
        pass
    profile = observations.manifest["metadata"]["profile"]
    top = profile["lanes"][0]["roi"][1]
    height = max(0, int(profile["effect_top"] - top))
    history = deque(maxlen=config.history_frames)
    rows, links, alternatives = [], [], []
    velocity = None
    previous = None
    repeat_edges = []
    for frame in frames(observations):
        points = landmarks(frame, profile, config)
        fid, ms = frame["id"], frame["ms"]
        row = {"frame_id": fid, "timestamp_ms": ms, "scroll_px": 0., "error_px": 0.,
               "step_error_px": 0., "status": STATUS["origin"], "support_lanes": 0,
               "repeat_group": fid, "repeat_confirmed": False, "reference_frame": -1}
        if previous is not None:
            dt = ms - previous["ms"]
            last = rows[-1]
            similar = (frame["group"] == previous["group"] or
                       (np.mean(np.abs(frame["projection"][:, :height].astype(float)
                                       - previous["projection"][:, :height])) <= config.repeat_difference
                        and np.mean(np.abs(frame["edge"][:, :height].astype(float)
                                           - previous["edge"][:, :height])) <= config.repeat_difference))
            zero_lanes = {int(p[0]) for p in points if any(
                p[0] == q[0] and p[1] == q[1] and abs(p[3] - q[3]) <= .5
                for q in previous["points"])}
            repeat = similar and len(zero_lanes) >= config.min_lanes
            repeat_edges.append(repeat)
            models = []
            if not repeat:
                # Nearest and older anchors reduce accumulated pixel rounding.
                refs = list(history)
                refs = [refs[i] for i in sorted({0, len(refs) // 2, len(refs) - 1})]
                for ref in refs:
                    elapsed = ms - ref["ms"]
                    # After viewport overlap is lost, similar new notes cannot
                    # establish an old-to-new association. Keep the PTS bridge
                    # and expanded uncertainty until fresh links are available.
                    if velocity is not None and velocity * elapsed >= height:
                        continue
                    limit = min(height, config.max_speed_px_per_ms * elapsed + config.displacement_slack_px)
                    found = match(ref["points"], points, limit, config)
                    for m in found:
                        m = {**m, "ref": ref, "scroll": ref["row"]["scroll_px"] + m["delta"]}
                        if m["scroll"] + config.match_tolerance_px >= last["scroll_px"]:
                            models.append(m)
            if models:
                # Prefer strong independent lanes, then the longest anchor.
                models.sort(key=lambda m: (-m["quality"], m["ref"]["id"], m["scroll"]))
                best = models[0]
                viable = [m for m in models if m["quality"] >= best["quality"] * config.ambiguity_ratio]
                predicted = last["scroll_px"] + (velocity or 0) * dt
                best = min(viable, key=lambda m: (abs(m["scroll"] - predicted), -m["quality"], m["ref"]["id"])) if velocity is not None else best
                spread = max(abs(m["scroll"] - best["scroll"]) for m in viable)
                ambiguous = spread > 2 * config.match_tolerance_px
                # Refine with consistent anchors without collapsing other modes.
                consistent = [m for m in viable if abs(m["scroll"] - best["scroll"]) <= config.match_tolerance_px]
                scroll = max(last["scroll_px"], float(np.median([m["scroll"] for m in consistent])))
                step_error = max(best["error"], spread)
                error = max(m["ref"]["row"]["error_px"] + m["error"] for m in consistent) + spread
                row.update(scroll_px=scroll, error_px=error, step_error_px=step_error,
                           status=STATUS["ambiguous" if ambiguous else "observed" if best["lanes"] >= config.min_lanes else "single_lane"],
                           support_lanes=best["lanes"], reference_frame=best["ref"]["id"])
                for m in viable:
                    alternatives.append((fid, m["ref"]["id"], m["scroll"], m["quality"], m["lanes"]))
                for p in best["links"]:
                    links.append((fid, best["ref"]["id"], *p[:6], p[4] - best["delta"]))
                elapsed = ms - best["ref"]["ms"]
                if elapsed > 0 and row["status"] == STATUS["observed"]:
                    measured = best["delta"] / elapsed
                    velocity = measured if velocity is None else .8 * velocity + .2 * measured
            elif repeat:
                row.update(scroll_px=last["scroll_px"], error_px=last["error_px"],
                           status=STATUS["unresolved"], reference_frame=previous["id"])
            else:
                delta = (velocity or 0) * dt
                error = config.max_speed_px_per_ms * dt + config.displacement_slack_px
                row.update(scroll_px=last["scroll_px"] + delta, error_px=last["error_px"] + error,
                           step_error_px=error, status=STATUS["extrapolated" if velocity is not None else "unresolved"])
        frame = {"id": fid, "ms": ms, "group": frame["group"], "points": points,
                 "row": row, "projection": frame["projection"], "edge": frame["edge"]}
        rows.append(row)
        if not repeat_edges or not repeat_edges[-1]:
            history.append(frame)
        else:
            # The display may be held for arbitrarily many PTS intervals. Its
            # last visible instance is still a current geometric anchor.
            history[-1] = frame
        previous = frame
    if not rows:
        raise ValueError("motion requires observations")
    # Confirm zero displacement only in a group connected to moving multi-lane
    # landmarks. A static title screen cannot establish a repeated scroll frame.
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) - 1 and repeat_edges[end]:
            end += 1
        outgoing = end + 1 < len(rows) and rows[end + 1]["status"] == STATUS["observed"] and rows[end + 1]["reference_frame"] >= rows[start]["frame_id"]
        confirmed = end > start and (rows[start]["status"] == STATUS["observed"] or outgoing)
        if confirmed:
            for i in range(start, end + 1):
                rows[i]["repeat_group"] = rows[start]["frame_id"]
                rows[i]["repeat_confirmed"] = True
                if i > start:
                    rows[i]["status"] = STATUS["repeated"]
        start = end + 1
    # An unobserved zero step has an interval, even when the displayed image is
    # static. No velocity or beat origin is invented for an empty recording.
    for i, row in enumerate(rows):
        if i and row["status"] == STATUS["unresolved"] and row["step_error_px"] == 0:
            row["step_error_px"] = config.max_speed_px_per_ms * (row["timestamp_ms"] - rows[i-1]["timestamp_ms"]) + config.displacement_slack_px
            row["error_px"] = rows[i-1]["error_px"] + row["step_error_px"]
    integers = {"frame_id", "status", "support_lanes", "repeat_group", "reference_frame"}
    arrays = {k: np.array([r[k] for r in rows], dtype=np.bool_ if k == "repeat_confirmed" else np.int64 if k in integers else np.float64) for k in rows[0]}
    link_array = np.asarray(links, np.float64).reshape(-1, 9)
    alternative_array = np.asarray(alternatives, np.float64).reshape(-1, 5)
    metadata = {"observation_cache_id": observations.id, "frame_count": len(rows),
                "lane_count": len(profile["lanes"]), "status_codes": STATUS,
                "coordinate": "relative scroll pixels; first observed frame S=0; not beat zero",
                "error_semantics": "uncalibrated conservative algorithmic envelope; shared drift does not shrink with observation count",
                "link_columns": ["frame_id", "reference_frame", "lane", "kind", "reference_seed", "seed", "delta_px", "similarity_weight", "residual_px"],
                "alternative_columns": ["frame_id", "reference_frame", "scroll_px", "support_score", "lanes"],
                "repeat_policy": "near/exact image agreement plus zero landmark motion and adjacent moving multi-lane support",
                "status_counts": {name: sum(r["status"] == code for r in rows) for name, code in STATUS.items()},
                "confirmed_repeat_frames": sum(r["repeat_confirmed"] for r in rows),
                "boundary_policy": "continuous rolling references across observation chunks; no note endpoints"}
    with CacheWriter(root, key) as writer:
        for start in range(0, len(rows), 120):
            end = min(start + 120, len(rows))
            lo, hi = rows[start]["frame_id"], rows[end-1]["frame_id"]
            writer.append(**{k: v[start:end] for k, v in arrays.items()},
                          links=link_array[(link_array[:, 0] >= lo) & (link_array[:, 0] <= hi)],
                          alternatives=alternative_array[(alternative_array[:, 0] >= lo) & (alternative_array[:, 0] <= hi)])
        return writer.commit(metadata, validate=open_motion), False


def open_motion(path, expected_key=None):
    try:
        cache = Cache(path, expected_key)
        key = cache.manifest["key"]
        if key["stage"] != "motion" or key["algorithm_version"] != VERSION or len(key["parents"]) != 1:
            raise CacheError("unsupported motion cache")
        n, last_id, last_ms, last_scroll = 0, -1, -1., -1.
        for c in cache.chunks():
            ids = c["frame_id"]
            size = len(ids)
            if not size or ids.ndim != 1 or np.any(np.diff(ids) != 1) or last_id >= 0 and ids[0] != last_id + 1:
                raise CacheError("invalid motion frame IDs")
            for name in ("frame_id", "status", "support_lanes", "repeat_group", "reference_frame"):
                if c[name].dtype.kind not in "iu":
                    raise CacheError("motion IDs/states must be integers")
            for name in ("timestamp_ms", "scroll_px", "error_px", "step_error_px", "status", "support_lanes", "repeat_group", "repeat_confirmed", "reference_frame"):
                if c[name].shape != (size,) or not np.isfinite(c[name]).all():
                    raise CacheError("invalid motion arrays")
            if (c["timestamp_ms"][0] < last_ms or np.any(np.diff(c["timestamp_ms"]) < 0)
                    or c["scroll_px"][0] < last_scroll or np.any(np.diff(c["scroll_px"]) < 0)
                    or np.any(c["error_px"] < 0) or np.any(c["step_error_px"] < 0)
                    or not np.isin(c["status"], list(STATUS.values())).all()
                    or np.any(c["repeat_group"] > ids) or np.any(c["repeat_group"] < 0)
                    or np.any(c["reference_frame"] >= ids) or np.any(c["reference_frame"] < -1)
                    or np.any(c["support_lanes"] < 0)
                    or np.any(c["support_lanes"] > cache.manifest["metadata"]["lane_count"])):
                raise CacheError("invalid motion coordinates/status")
            for name, width in (("links", 9), ("alternatives", 5)):
                if c[name].ndim != 2 or c[name].shape[1] != width or not np.isfinite(c[name]).all() or not np.isin(c[name][:, 0], ids).all():
                    raise CacheError("invalid motion evidence")
            n += size
            last_id, last_ms, last_scroll = int(ids[-1]), c["timestamp_ms"][-1], c["scroll_px"][-1]
        if n != cache.manifest["metadata"]["frame_count"] or key["parents"] != [cache.manifest["metadata"]["observation_cache_id"]]:
            raise CacheError("motion lineage/count mismatch")
        return cache
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        raise CacheError(f"invalid motion: {exc}") from exc


def timeline(cache):
    chunks = list(cache.chunks())
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0] if k not in {"links", "alternatives"}}
