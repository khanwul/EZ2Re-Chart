"""Recover missing draft landmarks from shared pixels and the recorded beat indicator."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import groupby
from typing import NamedTuple
import warnings

import numpy as np

from .barline_grid import _unbranched_rows
from .contracts import exact_ms, ratio, read_ratio
from .draft_placements import pixel_separation, translated_placements
from .draft_start_grid import intersect_scale
from .endpoint_ranges import closed
from .rhythm import DRAFT_GRID, grid_candidate_ticks


def _track(points, error):
    """Keep the longest unambiguous run of downward-moving terminal edges."""
    clean = sorted((f, float(np.median(tuple(ys)))) for f, ys in points.items()
                   if max(ys)-min(ys) <= 2*error)
    runs = [list(group) for _, group in groupby(enumerate(clean), key=lambda p: p[1][0] - p[0])]
    runs.sort(key=len, reverse=True)
    if not runs or len(runs[0]) < 8 or len(runs) > 1 and len(runs[1]) == len(runs[0]):
        return None
    coords = np.asarray([value for _, value in runs[0]])
    if np.any(np.diff(coords[:, 1]) <= 0):
        return None
    return coords


def _read_tracks(observations, requests, endpoints, shapes, error):
    profile = observations.manifest['metadata']['profile']
    params = observations.manifest['key']['parameters']['algorithm']
    threshold = params['brightness_threshold']
    edges = defaultdict(lambda: defaultdict(set))
    lights, times = {}, {}
    for chunk in observations.chunks():
        frame_ids = chunk['frame_id']
        local = {int(f): j for j, f in enumerate(frame_ids)}
        lights.update(zip(local, map(float, chunk['beat_brightness'])))
        times.update(zip(local, map(float, chunk['timestamp_ms'])))
        for j, (f, lane, seed) in enumerate(chunk['candidate_index']):
            lane = int(lane)
            for kind, name in enumerate(('tap', 'head', 'tail')):
                ids = requests.get((f, lane, seed, kind), ())
                if not ids or chunk['flags'][j, kind] & 3 or chunk['scores'][j, kind] < 0.7:
                    continue
                roi = profile['lanes'][lane]['roi']
                top = int(chunk['positions'][j, kind, 1])
                height = shapes[lane][name][0]
                y = top if name == 'tail' else top + height - 1
                local_y = y - roi[1]
                field = chunk['projection'][local[f], lane]
                if local_y < max(4, 2*height) or local_y + 4 >= len(field) or y >= profile['effect_top']-2*height:
                    continue
                inward = 1 if name == 'tail' else -1
                # A strong terminal edge must actually be present at the proposed landmark.
                if (min(field[local_y+2*inward], field[local_y+3*inward]) < max(128, threshold)
                        or max(field[local_y-2*inward], field[local_y-3*inward]) >= threshold):
                    continue
                for eid in ids:
                    if eid in endpoints:
                        edges[eid][int(f)].add(y)
    tracks = {eid: track for eid, samples in edges.items() if (track := _track(samples, error)) is not None}
    return tracks, lights, times


def _median(array, axis=None):
    # A missing brightness bin has no supporting samples in that pulse.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        return np.nanmedian(array, axis=axis)


class _Bar(NamedTuple):
    first: int
    last: int
    beat: float
    phase_by_brightness: np.ndarray
    clock_offset: float


def _pooled_phase(tracks, light, used, cycles, first, scale):
    # The same falling pulse is also observed across the recording. Remove the
    # integral cycle offset before pooling its intensity-to-phase measurements.
    phases = np.concatenate([beat + rows[:, 1]/scale - cycles[rows[:, 0].astype(int)-first]
                             for beat, rows in tracks])
    measured = light[used]
    low, high = np.percentile(measured, [1, 99])
    rough_phase = (high-measured)/(high-low)
    offsets = phases - rough_phase
    center = np.angle(np.mean(np.exp(2j*np.pi*offsets)))/(2*np.pi)
    normalized = phases - np.rint(offsets-center)
    boundaries = np.unique(np.quantile(measured, np.linspace(0, 1, 51)))
    bins = np.searchsorted(boundaries[:-1], measured, side='right')
    groups = [bins == index for index in np.unique(bins)]
    intensities = [np.median(measured[g]) for g in groups]
    waveform = np.interp(light, intensities, [np.median(normalized[g]) for g in groups])
    phase = cycles + waveform
    reference_cycles = {beat: round(np.median(beat + rows[:, 1]/scale - phase[rows[:, 0].astype(int)-first]))
                        for beat, rows in tracks}
    return phase, reference_cycles


def _phase_estimators(barlines, lights, times, scale):
    hypotheses = barlines.get('hypotheses', ())
    if len(hypotheses) != 1 or not lights:
        return {}
    labels = {a['marker_id']: float(read_ratio(a['beat'])) for a in hypotheses[0]['assignments']}
    tracks = []
    for t in barlines['tracks']:
        if t['id'] in labels:
            rows = [(o['frame_id'], o['y_px']) for o in _unbranched_rows(t) if o['frame_id'] in lights]
            if len(rows) >= 6:
                tracks.append((labels[t['id']], np.asarray(rows)))
    if len(tracks) < 4:
        return {}
    first = min(lights)
    ids = sorted(lights)
    if ids[-1]-first+1 != len(ids):
        return {}
    light = np.array([lights[f] for f in ids])
    if not np.all(np.isfinite(light) & (light >= 0) & (light <= 255)):
        return {}
    used = np.concatenate([rows[:, 0].astype(int)-first for _, rows in tracks])
    height = np.ptp(np.percentile(light[used], [3, 97]))
    if height < 8:
        return {}
    cycles = np.cumsum(np.diff(light, prepend=light[0]) > height*0.45)
    bpm = float(barlines['bpm'])
    period = 60000/bpm
    bars = []
    levels = np.arange(256, dtype=float)[:, None]
    for beat, rows in tracks:
        frames = rows[:, 0].astype(int)
        x = light[frames-first]
        c = cycles[frames-first]
        y = rows[:, 1]
        # Interpolate only on the falling portion between successive recorded frames.
        valid = (np.diff(frames) == 1) & (np.diff(c) == 0) & (np.diff(x) < -1) & (np.diff(y) > 0)
        if np.count_nonzero(valid) < 3:
            continue
        left, right = x[:-1][valid], x[1:][valid]
        phase = beat + y/scale - c
        lower, upper = phase[:-1][valid], phase[1:][valid]
        values = lower + (levels-left)/(right-left)*(upper-lower)
        lookup = _median(np.where((right <= levels) & (levels <= left), values, np.nan), axis=1)
        clock = np.median([times[f]/period - p/scale - beat for f, p in zip(frames, y)])
        bars.append(_Bar(frames[0], frames[-1], beat, lookup, clock))
    bars.sort(key=lambda bar: bar.first)
    if len(bars) < 4 or any(a.last >= b.last or a.beat >= b.beat for a, b in zip(bars, bars[1:])):
        return {}
    starts = [b.first for b in bars]
    ends = [b.last for b in bars]
    spacing = np.median(np.diff([b.beat for b in bars]))

    pooled_phase, reference_cycles = _pooled_phase(tracks, light, used, cycles, first, scale)

    def estimate(track, pooled=False):
        f, y = track[:, 0].astype(int), track[:, 1]
        left = bisect_left(ends, f[0])
        right = bisect_right(starts, f[-1])
        if left == 0 or right == len(bars) or bars[right].beat-bars[left-1].beat > 3*spacing:
            return None
        if pooled:
            offset = reference_cycles[bars[left-1].beat]
            if offset != reference_cycles[bars[right].beat]:
                return None
            result = np.median(pooled_phase[f-first] + offset - y/scale)
        else:
            # Match neighboring pulses at the nearest measured intensity bin.
            brightness = np.rint(light[f-first]).astype(int)
            delta = cycles[f-first]-y/scale
            sides = []
            for neighbors in (bars[max(0, left-4):left], bars[right:right+4]):
                phase = _median([bar.phase_by_brightness[brightness] for bar in neighbors], axis=0)
                sides.append(phase+delta)
            valid = np.isfinite(sides[0]) & np.isfinite(sides[1]) & (np.abs(sides[0]-sides[1]) < 0.125)
            if np.count_nonzero(valid) < max(8, len(f)//2):
                return None
            result = float(np.median(((sides[0]+sides[1])/2)[valid]))
        coarse = np.median([times[i]/period - p/scale for i, p in track])
        if any(abs(coarse-bar.clock_offset-result) > 0.25 for bar in (bars[left-1], bars[right])):
            return None
        return float(result)
    return {'nearby_pulses': estimate, 'pooled_pulse': lambda track: estimate(track, pooled=True)}


def _indicator_ranges(estimators, tracks, controls, missing, margin):
    ranges = defaultdict(dict)
    for method, estimate in estimators.items():
        residuals = [v - float(beat) for eid, beat in controls.items()
                     if eid in tracks and (v := estimate(tracks[eid])) is not None]
        if len(residuals) < 32 or max(map(abs, residuals)) >= 1/32:
            continue
        lower, upper = min(residuals) - margin, max(residuals) + margin
        for eid in missing.keys() & tracks.keys():
            value = estimate(tracks[eid])
            if value is not None:
                ranges[eid][method] = {'range': closed(exact_ms(value - upper), exact_ms(value - lower)),
                                       'calibration_taps': len(residuals)}
    return ranges


def _beat_separation(bounds, scale):
    if bounds is None or bounds[0] > bounds[1]:
        return None
    low, high = bounds
    return min(low/s for s in scale), max(high/s for s in scale)


def _relative_bounds(track, other, scale, error):
    return _beat_separation(pixel_separation(track, other, error), scale)


def _placement_domain(ticks, fixed, scale):
    domain = {}
    for tick in ticks:
        beat = Fraction(tick, DRAFT_GRID.ticks_per_beat)
        fit = scale
        for anchor_beat, bounds in fixed:
            fit = intersect_scale(fit, *bounds, beat - anchor_beat)
            if fit is None:
                break
        if fit is not None:
            domain[beat] = fit
    return domain


def _conflicting_starts(starts, tracks, scale, error):
    ordered = sorted(starts, key=lambda eid: tracks[eid][0, 0])
    conflicts = set()
    for index, eid in enumerate(ordered):
        for j in range(index + 1, len(ordered)):
            other = ordered[j]
            if tracks[other][0, 0] > tracks[eid][-1, 0]:
                break
            bounds = _relative_bounds(tracks[eid], tracks[other], scale, error)
            difference = starts[eid][0] - starts[other][0]
            if bounds is not None and not bounds[0] <= difference <= bounds[1]:
                conflicts.update((eid, other))
    return conflicts


def recover_positions(notes, candidates, observations, requests, shapes, barlines, scale_data, error,
                      *, placement_scale=None):
    """Only fill absent editable values. Physical ambiguity stays visible in the draft."""
    if not shapes or not scale_data:
        return
    if all(n['start_beat'] is not None and (n['type'] == 'tap' or n['end_beat'] is not None)
           for n in notes):
        return
    scale = tuple(map(read_ratio, scale_data))
    if min(scale) <= 0:
        return
    error = exact_ms(error)
    ids = {c['id']: c for c in candidates if c['selected']}
    event_counts = Counter(c.get('event_id', c['id']) for c in candidates)
    points = {}
    missing = {}
    anchors = {}
    controls = {}
    recoverable_reasons = {'no_observed_path_to_numbered_barline', 'constant_spatial_scale_extension'}
    for n in notes:
        if n['evidence'].get('hold_resolution') or 'nonpositive_measured_duration' in n['issues']:
            continue
        for side in ('start', 'end'):
            point = n['evidence'].get(side)
            if not point:
                continue
            eid = ids[n['id']][side]
            kind = 'tail' if side == 'end' else ('head' if n['type'] == 'longnote' else 'tap')
            expected = 0 if side == 'end' else shapes[n['lane']][kind][0]-1
            if read_ratio(point['offset_px']) != expected:
                continue
            points[eid] = (n, side)
            if n[side+'_beat'] is None:
                if set(point['reasons']) <= recoverable_reasons:
                    missing[eid] = point
            elif side == 'start' and n['type'] == 'tap' and len(point.get('candidate_ticks') or ()) == 1:
                anchors[eid] = read_ratio(n['start_beat'])
                if n['evidence']['pixel_status'] == 'resolved':
                    original_ticks = grid_candidate_ticks(point['beat_range'], DRAFT_GRID)
                    if len(original_ticks) == 1:
                        controls[eid] = anchors[eid]
    if not missing:
        return
    tracks, lights, times = _read_tracks(observations, requests, set(anchors) | set(missing), shapes, error)
    estimators = _phase_estimators(barlines, lights, times, float(sum(scale)/2))
    optical = _indicator_ranges(estimators, tracks, controls, missing, float(error/min(scale)))
    anchors = {e: b for e, b in anchors.items() if e in tracks}
    windows = {eid: (track[0, 0], track[-1, 0]) for eid, track in tracks.items()}
    adopted = set()
    while True:
        new = {}
        domains = {}
        for eid, point in missing.items():
            if eid not in tracks or eid in adopted:
                continue
            note, side = points[eid]
            limits = [(read_ratio(v['range']['lower']), read_ratio(v['range']['upper']))
                      for v in optical.get(eid, {}).values()]
            spatial = point['beat_range']
            if spatial:
                limits.append((read_ratio(spatial['lower']), read_ratio(spatial['upper'])))
            links = []
            fixed = []
            start, stop = windows[eid]
            for anchor, beat in anchors.items():
                first, last = windows[anchor]
                if last < start or stop < first:
                    continue
                pixels = pixel_separation(tracks[eid], tracks[anchor], error)
                if pixels is not None:
                    fixed.append((beat, pixels))
                bound = _beat_separation(pixels, scale)
                if bound is not None:
                    links.append(anchor)
                    limits.append((beat+bound[0], beat+bound[1]))
            if not limits or (side == 'end' and not links):
                continue
            lo = max(lower for lower, _ in limits)
            hi = min(upper for _, upper in limits)
            if lo > hi:
                continue
            domain = closed(lo, hi)
            if side == 'start':
                ticks = set(grid_candidate_ticks(domain, DRAFT_GRID))
                if point.get('shared_candidate_ticks') is not None:
                    ticks.intersection_update(point['shared_candidate_ticks'])
                if point.get('candidate_ticks') is not None:
                    ticks.intersection_update(point['candidate_ticks'])
                if note['end_beat'] is not None:
                    ticks = {t for t in ticks if Fraction(t, DRAFT_GRID.ticks_per_beat) < read_ratio(note['end_beat'])}
                if len(ticks) != 1:
                    if (note['type'] == 'tap' and note['evidence']['pixel_status'] == 'resolved'
                            and event_counts[ids[note['id']].get('event_id', note['id'])] == 1
                            and set(note['issues']) <= {'start_unavailable', 'start_grid_ambiguous',
                                                        'start_grid_no_candidate'}):
                        domains[eid] = _placement_domain(ticks, fixed, placement_scale or scale)
                    continue
                value = Fraction(ticks.pop(), DRAFT_GRID.ticks_per_beat)
            else:
                # Preserve the midpoint of the visible tail's co-observed geometric range.
                if note['start_beat'] is None or lo <= read_ratio(note['start_beat']):
                    continue
                value = (lo+hi)/2
            new[eid] = (value, domain, links)
        if not new:
            break
        # Settle the grid starts before measuring continuous ends against them.
        starts = {eid: result for eid, result in new.items() if points[eid][1] == 'start'}
        if starts:
            conflicts = _conflicting_starts(starts, tracks, scale, error)
            for eid in conflicts:
                del starts[eid], new[eid]
            new = starts or new
            if not new:
                break
        for eid, (value, domain, links) in new.items():
            note, side = points[eid]
            point = note['evidence'][side]
            point['recovery'] = {'indicator_models': optical.get(eid, {}),
                                 'shared_endpoints': sorted(links), 'frames': [int(i) for i in windows[eid]],
                                 'prior_range': point['beat_range'], 'prior_reasons': point['reasons']}
            point['reasons'] = ['shared_screen_landmarks']
            point['beat_range'] = domain
            point['estimate'] = ratio((read_ratio(domain['lower'])+read_ratio(domain['upper']))/2)
            note[side+'_beat'] = ratio(value)
            removed = {'end_unavailable'}
            if side == 'start':
                point['candidate_ticks'] = [int(value*DRAFT_GRID.ticks_per_beat)]
                anchors[eid] = value
                removed = {'start_unavailable', 'start_grid_ambiguous', 'start_grid_no_candidate'}
            note['issues'] = [reason for reason in note['issues'] if reason not in removed]
            note['status'] = 'unresolved' if note['issues'] else 'draft'
        adopted.update(new)
    placements = translated_placements(domains, tracks, error, placement_scale or scale)
    if placements:
        return {points[eid][0]['id']: [ratio(early), ratio(late)]
                for eid, (early, late) in placements.items()}
