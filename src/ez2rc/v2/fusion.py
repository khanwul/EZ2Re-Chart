"""Kind-specific visual evidence on a continuous scroll-coordinate lattice.

No note intervals or probabilities are produced. The bounded active tiles span
storage boundaries; finalized tiles remain contiguous evidence for a decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import FusionConfig
from .motion import STATUS, frames, timeline
from .storage import Cache, CacheError, CacheWriter, digest, stage_key

VERSION = "scroll-evidence-3"
CATEGORIES = ("visible", "unknown", "clipped", "judgment_effect", "occluded")


@dataclass(frozen=True)
class Evidence:
    id: str
    lane: int
    coordinate_px: float
    # None means endpoint calibration is unverified, not zero error.
    position_error_px: float | None
    scores: tuple[float, float, float, float]
    visible_weight: float
    occluded_weight: float
    observation_ids: tuple[str, ...]


def cache_key(observations, motion, config):
    return stage_key("fusion", VERSION, asdict(config), (observations.id, motion.id))


def visibility(flags, config):
    category = np.zeros(flags.shape, np.int64)
    category[(flags & 4) != 0] = 1
    category[(flags & 8) != 0] = 4
    category[(flags & 1) != 0] = 2
    category[(flags & 2) != 0] = 3
    weights = np.array([1., config.unknown_weight, config.clipped_weight,
                        config.effect_weight, config.occluded_weight])
    return category, weights[category]


def unique_contributions(values):
    """At most one contribution per lane/bin/kind in a displayed-frame group.

    Prefer visibility weight, then similarity. Adjacent seeds finding the same
    template cannot inflate support; all seed observations remain in the ledger.
    """
    if not len(values):
        return values
    order = np.lexsort((-values[:, 4], -values[:, 3], values[:, 2], values[:, 1], values[:, 0]))
    values = values[order]
    first = np.r_[True, np.any(values[1:, :3] != values[:-1, :3], axis=1)]
    return values[first]


def frame_contributions(deposits):
    """Interpolate distinct dense rows before capping repeated observations.

    Typed peaks are alternative seeds and retain their maximum contribution.
    Dense rows are adjacent samples: their split weights and weighted scores
    add within a frame. Keep category masses so clipping at a boundary cannot
    silently turn mixed visibility into entirely usable evidence.
    """
    values = np.zeros((len(deposits), 13))
    values[:, :8] = deposits
    values[np.arange(len(values)), 8 + deposits[:, 6].astype(int)] = deposits[:, 3]
    endpoints = unique_contributions(values[values[:, 2] != 3])
    body = values[values[:, 2] == 3]
    if not len(body):
        return endpoints
    order = np.lexsort((body[:, 1], body[:, 0]))
    body = body[order]
    starts = np.flatnonzero(np.r_[True, np.any(body[1:, :2] != body[:-1, :2], axis=1)])
    merged = body[starts].copy()
    merged[:, 3] = np.add.reduceat(body[:, 3], starts)
    merged[:, 4] = np.add.reduceat(body[:, 3] * body[:, 4], starts) / merged[:, 3]
    merged[:, 5] = np.maximum.reduceat(body[:, 5], starts)
    merged[:, 8:] = np.add.reduceat(body[:, 8:], starts, axis=0)
    # Column6 is only an exclusive input category. The interpolated row uses
    # the five mass columns, including when its two source rows differ.
    merged[:, 6] = -1
    return np.concatenate((endpoints, merged))


def accumulate(observations: Cache, motion: Cache, config: FusionConfig, root: Path):
    if motion.manifest["key"]["parents"] != [observations.id]:
        raise ValueError("fusion motion/observation lineage mismatch")
    key = cache_key(observations, motion, config)
    try:
        return open_fusion(root / "fusion" / digest(key), key), True
    except CacheError:
        pass
    meta = observations.manifest["metadata"]
    profile = meta["profile"]
    lanes, judge = len(profile["lanes"]), profile["judge_y"]
    calibrations = [profile["calibration"][k] for k in ("tap", "head", "tail")]
    offsets = np.array([c.get("offset_px", 0.) if c["status"] == "verified" else 0. for c in calibrations])
    calibration_errors = np.array([c.get("error_px", 0.) if c["status"] == "verified" else 0. for c in calibrations])
    clock = timeline(motion)
    tiles = {}
    group, pending = None, np.empty((0, 13), np.float64)
    ledger = []
    count = max_tiles = emitted_bins = 0
    previous_scroll = 0.

    with CacheWriter(root, key) as writer:
        def add_group():
            nonlocal pending, max_tiles
            if not len(pending):
                return
            # Cap applies independently to each spatial evidence cell/channel.
            capped = np.minimum(pending[:, 3], config.repeat_group_cap)
            pending[:, 8:] *= (capped / pending[:, 3])[:, None]
            pending[:, 3] = capped
            for tid in np.unique(np.floor_divide(pending[:, 1], config.tile_bins).astype(np.int64)):
                v = pending[np.floor_divide(pending[:, 1], config.tile_bins) == tid]
                if int(tid) not in tiles:
                    shape = (lanes, config.tile_bins, 4)
                    tiles[int(tid)] = {"score_sum": np.zeros(shape), "weight_sum": np.zeros(shape),
                                       "visibility_weight": np.zeros((*shape, 5)),
                                       "group_count": np.zeros(shape, np.int64), "error_px": np.zeros(shape),
                                       "frame_min": np.full(shape, -1, np.int64), "frame_max": np.full(shape, -1, np.int64)}
                tile = tiles[int(tid)]
                ix = (v[:, 0].astype(int), (v[:, 1] - tid * config.tile_bins).astype(int), v[:, 2].astype(int))
                tile["score_sum"][ix] += v[:, 3] * v[:, 4]
                tile["weight_sum"][ix] += v[:, 3]
                tile["visibility_weight"][ix] += v[:, 8:]
                tile["group_count"][ix] += 1
                tile["error_px"][ix] = np.maximum(tile["error_px"][ix], v[:, 5] + config.grid_px)
                old = tile["frame_min"][ix]
                tile["frame_min"][ix] = np.where(old < 0, v[:, 7], np.minimum(old, v[:, 7])).astype(int)
                tile["frame_max"][ix] = np.maximum(tile["frame_max"][ix], v[:, 7]).astype(int)
            max_tiles = max(max_tiles, len(tiles))
            pending = np.empty((0, 13), np.float64)

        def flush(before=None):
            nonlocal emitted_bins
            ready = sorted(t for t in tiles if before is None or (t + 1) * config.tile_bins <= before)
            indices, outputs = [], {k: [] for k in ("score_sum", "weight_sum", "visibility_weight", "group_count", "error_px", "frame_min", "frame_max")}
            for tid in ready:
                tile = tiles.pop(tid)
                lane, cell = np.nonzero(np.any(tile["weight_sum"] > 0, axis=2))
                indices.append(np.column_stack((lane, tid * config.tile_bins + cell)))
                for k in outputs:
                    outputs[k].append(tile[k][lane, cell])
            if not ready and not ledger:
                return
            index = np.concatenate(indices) if indices else np.empty((0, 2), np.int64)
            emitted_bins += len(index)
            arrays = {}
            for k, values in outputs.items():
                dtype = np.int64 if k in {"group_count", "frame_min", "frame_max"} else np.float64
                shape = (0, 4, 5) if k == "visibility_weight" else (0, 4)
                arrays[k] = np.concatenate(values) if values else np.empty(shape, dtype)
            rows = np.concatenate(ledger) if ledger else np.empty((0, 13), np.float64)
            writer.append(index=index, **arrays, endpoint_index=rows[:, :3].astype(np.int64),
                          endpoint_q=rows[:, 3:6], endpoint_weight=rows[:, 6:9],
                          endpoint_error_px=rows[:, 9:12], endpoint_group=rows[:, 12].astype(np.int64))
            ledger.clear()

        for i, frame in enumerate(frames(observations)):
            if i >= len(clock["frame_id"]) or frame["id"] != clock["frame_id"][i] or frame["ms"] != clock["timestamp_ms"][i]:
                raise ValueError("fusion frame timeline differs from motion")
            scroll, error = clock["scroll_px"][i], clock["error_px"][i]
            status = clock["status"][i]
            scale = (0. if status == STATUS["unresolved"] else config.inferred_motion_weight
                     if status in {STATUS["single_lane"], STATUS["ambiguous"], STATUS["extrapolated"]} else 1.)
            # A provisional equality group is conservatively capped too, but its
            # status is never promoted to a confirmed repeat in motion.
            current_group = int(clock["repeat_group"][i] if clock["repeat_confirmed"][i] else frame["group"])
            if group is not None and current_group != group:
                add_group()
            group = current_group
            coords = scroll + judge - frame["positions"][:, :, 1] - offsets
            categories, weights = visibility(frame["flags"], config)
            weights *= scale
            errors = np.broadcast_to(error + calibration_errors, coords.shape)
            ledger.append(np.column_stack((frame["candidate_index"], coords, weights, errors,
                                           np.full(len(coords), group))))
            # Dense brightness includes observed darkness; missing coverage has
            # zero weight and is never interpreted as absent longnote body.
            height = frame["projection"].shape[1]
            ys = np.arange(height) + profile["lanes"][0]["roi"][1]
            body_flags = np.full((lanes, height), 4, np.uint8)
            body_flags[:, [0, -1]] |= 1
            body_flags[:, ys >= profile["effect_top"]] |= 2
            body_cat, body_weight = visibility(body_flags, config)
            body_weight *= scale
            lane_ids = np.r_[np.repeat(frame["candidate_index"][:, 1], 3), np.repeat(np.arange(lanes), height)]
            kinds = np.r_[np.tile(np.arange(3), len(coords)), np.full(lanes * height, 3)]
            q = np.r_[coords.ravel(), np.tile(scroll + judge - ys, lanes)] / config.grid_px
            values = np.r_[frame["scores"][:, :3].ravel(), frame["projection"].astype(float).ravel() / 255.]
            weight = np.r_[weights.ravel(), body_weight.ravel()]
            errs = np.r_[errors.ravel(), np.full(lanes * height, error)]
            cats = np.r_[categories.ravel(), body_cat.ravel()]
            # Linear splatting preserves subpixel positions; raw coordinates
            # and candidate IDs remain available even for closely spaced peaks.
            floor = np.floor(q)
            fraction = q - floor
            deposits = []
            for bins, part in ((floor, 1 - fraction), (floor + 1, fraction)):
                w = weight * part
                use = w > 0
                deposits.append(np.column_stack((lane_ids[use], bins[use], kinds[use], w[use],
                                                  values[use], errs[use], cats[use], np.full(use.sum(), frame["id"]))))
            contribution = frame_contributions(np.concatenate(deposits))
            pending = unique_contributions(np.concatenate((pending, contribution))) if len(pending) else contribution
            count += len(coords)
            previous_scroll = scroll
            # Pending group may span the observation/output boundary. Keep its
            # cells until closed; never finalize a physical endpoint. Retire
            # tiles as the viewport passes them, independently of frame chunks.
            minimum = min(scroll + judge - profile["lanes"][0]["roi"][3] - max(0., float(offsets.max())),
                          float(pending[:, 1].min() * config.grid_px) if len(pending) else float("inf"))
            before = int(np.floor(minimum / config.grid_px)) - 1
            if (i + 1) % 120 == 0 or any((t + 1) * config.tile_bins <= before for t in tiles):
                flush(before)
        if i + 1 != len(clock["frame_id"]):
            raise ValueError("fusion motion frame count differs")
        add_group()
        flush()
        metadata = {"observation_cache_id": observations.id, "motion_cache_id": motion.id,
                    "frame_count": i + 1, "endpoint_count": count, "bin_count": emitted_bins,
                    "lane_count": lanes, "max_active_tiles": max_tiles,
                    "coordinate": "q = S + judge_y - (template_top_y + verified_offset); body uses aligned row y",
                    "grid_px": config.grid_px, "kind_columns": ["tap_similarity", "head_similarity", "tail_similarity", "body_brightness_fraction"],
                    "score_semantics": "weighted raw similarities/brightness, not probabilities; score_sum / weight_sum only where weight > 0",
                    "visibility_categories": list(CATEGORIES), "visibility_policy": "exclusive per sample: judgment_effect > clipped > occluded > unknown > visible; dense interpolation preserves category mass",
                    "repeat_policy": "dense rows interpolate additively within each frame; typed seeds and displayed groups use maximum weight then similarity per lane/bin/kind, capped",
                    "calibration": {k: profile["calibration"][k] for k in ("tap", "head", "tail")},
                    "endpoint_timing_error_px": [c.get("error_px") if c["status"] == "verified" else None for c in calibrations],
                    "observation_position_error_px": meta.get("position_error_px"),
                    "error_semantics": "shared scroll envelope plus verified calibration error (tiles add one grid spacing); unverified localization/calibration errors remain unknown, not zero",
                    "endpoint_id": "f{frame_id}:l{lane}:y{seed_row}; lookup original scores/flags in observation_cache_id",
                    "evidence_id": "l{lane}:q{bin_index}:k{kind}; stable across storage chunks",
                    "boundary": {"scroll_px": float(previous_scroll), "open_candidate_ids": [],
                                 "reason": "no interval inference; coordinate tiles continue across chunks without note creation"}}
        return writer.commit(metadata, validate=open_fusion), False


def open_fusion(path, expected_key=None):
    try:
        cache = Cache(path, expected_key)
        key, meta = cache.manifest["key"], cache.manifest["metadata"]
        if key["stage"] != "fusion" or key["algorithm_version"] != VERSION or key["parents"] != [meta["observation_cache_id"], meta["motion_cache_id"]]:
            raise CacheError("unsupported fusion cache/lineage")
        seen, bins, endpoints = set(), 0, 0
        for c in cache.chunks():
            index, endpoint = c["index"], c["endpoint_index"]
            n, m = len(index), len(endpoint)
            if index.shape != (n, 2) or endpoint.shape != (m, 3) or index.dtype.kind not in "iu" or endpoint.dtype.kind not in "iu":
                raise CacheError("invalid fusion indices")
            if np.any(index[:, 0] < 0) or np.any(index[:, 0] >= meta["lane_count"]):
                raise CacheError("invalid fusion lane")
            # Tile IDs cannot recur after finalization; this catches duplicated
            # boundary evidence without retaining all per-cell IDs in memory.
            tids = set(np.floor_divide(index[:, 1], key["parameters"]["tile_bins"]).tolist())
            if seen & tids or len(np.unique(index, axis=0)) != n:
                raise CacheError("duplicate fusion cells/tiles")
            seen |= tids
            for k in ("score_sum", "weight_sum", "group_count", "error_px", "frame_min", "frame_max", "visibility_weight"):
                shape = (n, 4, 5) if k == "visibility_weight" else (n, 4)
                if c[k].shape != shape or not np.isfinite(c[k]).all():
                    raise CacheError("invalid fusion arrays")
            if (np.any(c["weight_sum"] < 0) or np.any(c["error_px"] < 0)
                    or not np.allclose(c["visibility_weight"].sum(axis=2), c["weight_sum"], rtol=1e-12, atol=1e-12)):
                raise CacheError("invalid fusion weights")
            for k in ("endpoint_q", "endpoint_weight", "endpoint_error_px", "endpoint_group"):
                if c[k].shape != ((m,) if k == "endpoint_group" else (m, 3)) or not np.isfinite(c[k]).all():
                    raise CacheError("invalid endpoint evidence")
            if np.any(c["endpoint_weight"] < 0) or np.any(c["endpoint_error_px"] < 0):
                raise CacheError("invalid endpoint weights/errors")
            bins += n
            endpoints += m
        if bins != meta["bin_count"] or endpoints != meta["endpoint_count"]:
            raise CacheError("fusion counts mismatch")
        return cache
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        raise CacheError(f"invalid fusion: {exc}") from exc
