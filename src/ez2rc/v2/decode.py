"""Pixel interval inference. No video time, BPM, or chart-reference input.

Endpoint sites retain all three interpretations of every observation. Their
coordinate footprints are exclusive resources in an ordered interval DAG.
Skip edges mean noise; forward/backward costs retain *every* near-optimal edge,
without a beam or a duration cutoff. Storage tiles are only ingestion batches.
"""
from __future__ import annotations

import codecs
from dataclasses import asdict
import heapq
import json
import math
from pathlib import Path

import numpy as np

from .config import DecodeConfig, read_position_calibration
from .contracts import Event
from .motion import STATUS, timeline
from .observation import template_shapes
from .storage import Cache, CacheError, CacheWriter, canonical, digest, stage_key

VERSION = "lane-interval-dp-10"
KINDS = ("tap", "head", "tail")


def cache_key(observations, motion, fusion, config):
    return stage_key("decode", VERSION, asdict(config), (observations.id, motion.id, fusion.id))


def ordered_dp(size, candidates, margin, *, verify=False):
    """Exact shortest paths in a DAG with zero-cost noise edges.

    Candidate [lo, hi] consumes inclusive resource tokens. Adjacent, disjoint
    resources may touch at the same q. Marginals are whole-lane costs, not
    probabilities. Replacement costs use edges crossing the selected edge's
    first cut, so they also include the all-noise alternative.
    Verification compares derived fields without changing or copying candidates.
    """
    outgoing = [[] for _ in range(size)]
    for i, c in enumerate(candidates):
        if not 0 <= c['lo'] <= c['hi'] < size or not np.isfinite(c['cost']):
            raise ValueError("invalid interval edge")
        outgoing[c['lo']].append(i)
    forward = np.zeros(size + 1)
    previous = [None] * (size + 1)
    forward[1:] = np.inf
    for u in range(size):
        if forward[u] <= forward[u + 1]:
            forward[u + 1], previous[u + 1] = forward[u], (u, None)
        for i in outgoing[u]:
            c = candidates[i]; v = c['hi'] + 1
            value = forward[u] + c['cost']
            if value < forward[v]:
                forward[v], previous[v] = value, (u, i)
    backward = np.zeros(size + 1)
    for u in range(size - 1, -1, -1):
        backward[u] = min([backward[u + 1], *[
            candidates[i]['cost'] + backward[candidates[i]['hi'] + 1] for i in outgoing[u]]])
    selected, cursor = set(), size
    while cursor:
        cursor, edge = previous[cursor]
        if edge is not None:
            selected.add(edge)
    best = float(forward[-1])
    heap = []
    for u in range(size):
        for i in outgoing[u]:
            c = candidates[i]
            path_cost = float(forward[u] + c['cost'] + backward[c['hi'] + 1])
            excess_cost = max(0., path_cost - best)
            chosen = i in selected
            competitive = excess_cost <= margin + 1e-10
            if verify:
                if (c['path_cost'] != path_cost or c['excess_cost'] != excess_cost
                        or c['selected'] != chosen or c['competitive'] != competitive
                        or not chosen and c['exclusion_cost'] is not None):
                    raise ValueError('invalid path or alternative cost')
            else:
                c.update(path_cost=path_cost, excess_cost=excess_cost, selected=chosen,
                         competitive=competitive, exclusion_cost=None)
            heapq.heappush(heap, (path_cost, i, c['hi']))
        for i in outgoing[u]:
            if i not in selected:
                continue
            held = []
            while heap and (heap[0][2] < u or heap[0][1] == i):
                entry = heapq.heappop(heap)
                if entry[1] == i:
                    held.append(entry)
            alternative = float(forward[u] + backward[u + 1])
            if heap:
                alternative = min(alternative, heap[0][0])
            if verify:
                if candidates[i]['exclusion_cost'] != alternative:
                    raise ValueError('invalid path or alternative cost')
            else:
                candidates[i]['exclusion_cost'] = alternative
            for entry in held:
                heapq.heappush(heap, entry)
    return best


class BodyRunScale:
    """Share a bounded brightness reward across fragments of a continuous run."""
    def __init__(self, runs):
        self.runs = np.asarray(runs, dtype=float).reshape(-1, 2)
        self.lengths = self.runs[:, 1] - self.runs[:, 0]

    def scale(self, start, end, minimum):
        if start >= end:
            return minimum
        first = int(np.searchsorted(self.runs[:, 1], start, side='right'))
        stop = int(np.searchsorted(self.runs[:, 0], end, side='left'))
        return max(minimum, float(self.lengths[first:stop].max()) if first < stop else 0.)


class BodyEvidence:
    """Sparse cells interpreted as piecewise constant, with prefix integrals.

    Positive weight and dark brightness is absence. Occluded/clipped/effect
    contributions and holes are unavailable, with a separate, smaller cost.
    No duration is rewarded without bound: support reward is a fraction.
    Brightness and observation support remain separate: a bright cell with
    little usable weight is neither sustained evidence nor observed darkness.
    """
    def __init__(self, cells, grid, config, clock=None, endpoint_footprints=None):
        self.grid, self.config = grid, config
        self.q = cells['index'][:, 1].astype(float) * grid
        self.errors = cells.get('error_px', np.zeros((len(self.q), 4)))[:, 3]
        self.frame_min = cells.get('frame_min', np.full((len(self.q), 4), -1))[:, 3]
        self.frame_max = cells.get('frame_max', np.full((len(self.q), 4), -1))[:, 3]
        self.ambiguous_frames = (clock['frame_id'][clock['status'] == STATUS['ambiguous']]
                                 if clock is not None else np.array([], np.int64))
        w = cells['weight_sum'][:, 3]
        mean = np.divide(cells['score_sum'][:, 3], w, out=np.zeros_like(w), where=w > 0)
        visible = cells['visibility_weight'][:, 3, :2].sum(axis=1)
        trust = np.divide(visible, w, out=np.zeros_like(w), where=w > 0)
        bright = np.clip((mean - config.body_dark) / (config.body_bright - config.body_dark), 0, 1) * trust
        dark = np.clip((config.body_bright - mean) / (config.body_bright - config.body_dark), 0, 1) * trust
        # Fusion has already capped repeated frames and duplicate seeds. Use
        # its usable weight on the same support scale as typed endpoints, not
        # raw frame count. Keep the raw bright integral and run boundaries.
        supported = bright * np.minimum(1., visible / config.support_weight)
        self.values = np.column_stack((bright, dark, trust, supported))
        self.prefix = np.vstack((np.zeros(4), np.cumsum(self.values * grid, axis=0)))
        self.endpoint_footprints = union_ranges(endpoint_footprints) if endpoint_footprints is not None else None
        # Integrate only the supported brightness shared with endpoint patches.
        # Darkness, coverage and raw run boundaries retain their original meaning.
        segments = []
        for a, b in self.endpoint_footprints or ():
            first = int(np.searchsorted(self.q, a - grid, side='right'))
            last = int(np.searchsorted(self.q, b, side='left'))
            segments.extend((max(a, q), min(b, q + grid), supported[i])
                            for i in range(first, last) if (q := self.q[i]) + grid > a and q < b)
        self.overlap = np.array(segments, dtype=float).reshape(-1, 3)
        self.overlap_prefix = np.r_[0., np.cumsum((self.overlap[:, 1] - self.overlap[:, 0]) * self.overlap[:, 2])]
        # Contiguous support may pass through unobservable cells, but observed
        # darkness ends a missing-endpoint hypothesis. No tile edge is consulted.
        strong = bright > .5
        self.runs = []
        start = last = None
        for i, q in enumerate(self.q):
            if dark[i] > .5:
                if start is not None:
                    self.runs.append((start, last + grid))
                start = last = None
            elif strong[i]:
                if start is None:
                    start = q
                last = q
        if start is not None:
            self.runs.append((start, last + grid))
        self.reward_scale = BodyRunScale(self.runs)

    def integral(self, x):
        j = int(np.searchsorted(self.q, x, side='right'))
        value = self.prefix[j].copy()
        if j and x < self.q[j - 1] + self.grid:
            value -= self.values[j - 1] * (self.q[j - 1] + self.grid - x)
        return value

    def summary(self, start, end):
        length = max(0., end - start)
        bright, dark, covered, supported = self.integral(end) - self.integral(start)
        bright = max(0., float(bright))
        supported = float(np.clip(supported, 0., bright))
        unavailable = max(0., length - covered)
        left = int(np.searchsorted(self.q, start - self.grid, side='right'))
        right = int(np.searchsorted(self.q, end, side='left'))
        frame_min = self.frame_min[left:right]
        first = int(frame_min[frame_min >= 0].min()) if np.any(frame_min >= 0) else None
        last = int(self.frame_max[left:right].max()) if right > left else None
        uncertain = (first is not None and bool(np.any(
            (self.ambiguous_frames >= first) & (self.ambiguous_frames <= last))))
        scale = max(length, self.grid, self.config.body_support_px)
        support, supported_fraction = bright / scale, supported / scale
        result = {'range_px': [float(start), float(end)], 'bright_px': bright,
                'supported_bright_px': supported, 'supported_fraction': supported_fraction,
                'absent_px': max(0., float(dark)), 'unavailable_px': unavailable,
                'support_fraction': support, 'support_scale_px': scale,
                'reward_scale_px': self.reward_scale.scale(start, end, scale),
                'fusion_error_px': float(self.errors[left:right].max()) if right > left else None,
                'frame_bounds': [first, last], 'motion_ambiguous': uncertain,
                'cost': float(self.config.absence_cost_per_px * max(0., dark)
                              + self.config.unavailable_cost_per_px * unavailable
                              - self.config.body_reward * supported_fraction),
                'evidence_ref': {'kind': 'body', 'bin_start': int(np.floor(start / self.grid)),
                                 'bin_end_exclusive': int(np.ceil(end / self.grid))}}
        if self.endpoint_footprints is not None:
            def integral(x):
                j = int(np.searchsorted(self.overlap[:, 0], x, side='right'))
                total = self.overlap_prefix[j]
                if j and x < self.overlap[j - 1, 1]:
                    total -= self.overlap[j - 1, 2] * (self.overlap[j - 1, 1] - x)
                return total
            result['endpoint_overlap_supported_px'] = float(np.clip(integral(end) - integral(start), 0., supported))
        return result

    def extent(self, q, direction, radius):
        possible = [(a, b) for a, b in self.runs if a - radius <= q <= b + radius]
        if not possible:
            return q
        return float(min(a for a, _ in possible) if direction < 0 else max(b for _, b in possible))


def _clusters(q, radius):
    """Bounded diameter prevents transitive merging across a dense passage."""
    order = np.argsort(q, kind='stable')
    ordered = q[order]
    at = 0
    while at < len(order):
        end = int(np.searchsorted(ordered, ordered[at] + 2 * radius, side='right'))
        yield order[at:end]
        at = end


def _associated_clusters(anchor, primary, scores, frames, usable, config, interior=None):
    """Anchor sites on usable peaks before attaching weak observations.

    Weak rows cannot set the cut between two strong observations. A site's
    diameter is still bounded, and distinct strong same-kind peaks observed in
    one frame cannot merge. Exact same-frame maxima from duplicate seeds can.
    This is a spatial association heuristic, not a calibrated tracking model.
    """
    strength = scores[np.arange(len(scores)), primary]
    strong = usable & (strength >= config.association_similarity)
    order = sorted(np.flatnonzero(strong), key=lambda i: (-strength[i], int(i)))
    spatial = np.argsort(anchor, kind='stable')
    coordinates = anchor[spatial]
    assigned = np.zeros(len(anchor), bool)
    diameter = 2 * config.endpoint_radius_px
    groups = []
    for seed in order:
        if assigned[seed]:
            continue
        lo = np.searchsorted(coordinates, anchor[seed] - diameter, side='left')
        hi = np.searchsorted(coordinates, anchor[seed] + diameter, side='right')
        nearby = [i for i in spatial[lo:hi] if strong[i] and not assigned[i]]
        nearby.sort(key=lambda i: (abs(anchor[i] - anchor[seed]), -strength[i], int(i)))
        rows, peaks = [], {}
        lower = upper = anchor[seed]
        for i in nearby:
            a, b = min(lower, anchor[i]), max(upper, anchor[i])
            key = (int(frames[i]), int(primary[i]))
            if b - a > diameter or key in peaks and abs(peaks[key] - anchor[i]) > 1e-7:
                continue
            rows.append(int(i)); peaks[key] = anchor[i]; assigned[i] = True
            lower, upper = a, b
        groups.append({'rows': rows, 'peaks': peaks, 'center': float(np.median(list(peaks.values()))),
                       'lower': lower, 'upper': upper})
    groups.sort(key=lambda g: g['center'])
    centers = np.array([g['center'] for g in groups])
    # Fixed strong centers prevent weak attachment from moving an anchor.
    for i in sorted(np.flatnonzero(~strong), key=lambda i: (-strength[i], int(i))):
        lo = np.searchsorted(centers, anchor[i] - diameter, side='left')
        hi = np.searchsorted(centers, anchor[i] + diameter, side='right')
        choices = [j for j in range(lo, hi) if
                   max(groups[j]['upper'], anchor[i]) - min(groups[j]['lower'], anchor[i]) <= diameter]
        # A censored search response can be associated using a same-frame
        # interior maximum even when accumulated motion spreads the group's
        # other frames. Such a response does not widen its anchor support.
        witnessed = []
        if interior is not None and not interior[i, primary[i]]:
            key = (int(frames[i]), int(primary[i]))
            witnessed = [j for j in range(lo, hi) if key in groups[j]['peaks']
                         and abs(groups[j]['peaks'][key]-anchor[i]) <= diameter]
        if len(witnessed) == 1:
            g = groups[witnessed[0]]
            g['rows'].append(int(i)); assigned[i] = True
            continue
        if choices:
            j = min(choices, key=lambda j: (abs(centers[j] - anchor[i]), j))
            g = groups[j]
            g['rows'].append(int(i)); assigned[i] = True
            g['lower'], g['upper'] = min(g['lower'], anchor[i]), max(g['upper'], anchor[i])
    result = [(np.array(g['rows']), g['center'], 'supported_peak') for g in groups]
    remaining = np.flatnonzero(~assigned)
    for cluster in _clusters(anchor[remaining], config.endpoint_radius_px):
        rows = remaining[cluster]
        result.append((rows, float(np.median(anchor[rows])), 'weak_only'))
    return sorted(result, key=lambda item: (item[1], int(item[0].min())))


def endpoint_foreground(chunk, lane, heights, top, threshold):
    """Amplitude evidence at each matched patch, independent of correlation.

    Normalized correlation is invariant to amplitude and can match nearly
    black compression texture. Retained row projections supply a separate
    bounded support factor, using the observation's existing brightness scale.
    Average over the matched height: one bright scanline does not establish
    foreground support for an entire template patch. No row or kind hypothesis
    is discarded.
    """
    indices = np.flatnonzero(chunk['candidate_index'][:, 1] == lane)
    result = np.ones((len(indices), 3))
    if threshold <= 0:
        return result
    frames = {int(f): i for i, f in enumerate(chunk['frame_id'])}
    for row, index in enumerate(indices):
        projection = chunk['projection'][frames[int(chunk['candidate_index'][index, 0])], lane]
        for kind, height in enumerate(heights):
            y = int(chunk['positions'][index, kind, 1])-top
            patch = projection[max(0, y):min(len(projection), y+height)]
            result[row, kind] = min(1., max(0., float(patch.mean()))/threshold) if len(patch) else 0.
    return result


def _kind_modes(rows, kind, ledger, scores, flags, frames, interior, config):
    """Apply the same peak association to each retained kind interpretation.

    A weak attachment must not move a mode boundary and split the strong
    localizations it was attached to. Distinct co-observed interior maxima
    still form separate modes, including for a non-winning kind.
    """
    usable = (((flags[rows, kind] & 11) == 0)
              & (ledger['endpoint_weight'][rows, kind] > 0) & interior[rows, kind])
    groups = _associated_clusters(ledger['endpoint_q'][rows, kind], np.full(len(rows), kind),
                                  scores[rows], frames[rows], usable, config, interior[rows])
    for indices, _, _ in groups:
        yield rows[indices]


def endpoint_sites(lane, ledger, scores, flags, clock, meta, config):
    if not len(scores):
        return [], []
    # Use the best raw template only to associate observations, never to delete
    # their other kind interpretations. Weak and zero-weight rows remain here.
    primary = scores[:, :3].argmax(axis=1)
    anchor = ledger['endpoint_q'][np.arange(len(scores)), primary]
    sites, endpoints = [], []
    frames = ledger['endpoint_index'][:, 0]
    clock_index = np.searchsorted(clock['frame_id'], frames)
    at = np.arange(len(scores))
    interior = ledger.get('endpoint_peak_interior', np.ones((len(scores), 3), bool))
    foreground = ledger.get('endpoint_foreground', np.ones((len(scores), 3)))
    association_usable = (((flags[at, primary] & 11) == 0) & (ledger['endpoint_weight'][at, primary] > 0)
                          & interior[at, primary])
    support_scores = scores[:, :3] * foreground
    for cluster, center, association in _associated_clusters(anchor, primary, support_scores, frames, association_usable, config, interior):
        # A tail and a distinct co-observed onset may touch. Tap/head are
        # competing onsets and stay exclusive, as do same-kind duplicate seeds.
        # Across frames, a changing winning kind alone does not split a site.
        coexist = any(2 in kinds and bool(kinds & {0, 1}) for kinds in (
            set(primary[cluster][frames[cluster] == f]) for f in np.unique(frames[cluster])))
        partitions = ([cluster[primary[cluster] != 2], cluster[primary[cluster] == 2]] if coexist else [cluster])
        for rows in partitions:
            if not len(rows):
                continue
            site_id = f'l{lane}:s{len(sites)}'
            strong_rows = rows[association_usable[rows] & (support_scores[rows, primary[rows]] >= config.association_similarity)]
            support_rows = strong_rows if len(strong_rows) else rows
            site = {'id': site_id, 'anchor_px': center, 'association': association,
                    'anchor_range_px': [float(anchor[support_rows].min()), float(anchor[support_rows].max())],
                    'observed_anchor_range_px': [float(anchor[rows].min()), float(anchor[rows].max())],
                    'primary_kind': KINDS[int(np.argmax(np.bincount(primary[rows], minlength=3)))],
                    'endpoint_ids': []}
            for kind in range(3):
                for indices in _kind_modes(rows, kind, ledger, support_scores, flags, frames, interior, config):
                    # One best contribution per displayed group, regardless of
                    # duplicate seeds or repeated observation frames.
                    order = sorted(indices, key=lambda i: (int(ledger['endpoint_group'][i]),
                                   -float(ledger['endpoint_weight'][i, kind]), -float(scores[i, kind]), int(i)))
                    representatives = []
                    seen = set()
                    for i in order:
                        group = int(ledger['endpoint_group'][i])
                        if group not in seen:
                            representatives.append(i); seen.add(group)
                    reps = np.array(representatives)
                    weights = np.minimum(ledger['endpoint_weight'][reps, kind], meta.get('repeat_group_cap', 1.))
                    total = float(weights.sum())
                    score = float(np.dot(weights, scores[reps, kind]) / total) if total else float(scores[reps, kind].max())
                    supported_score = (float(np.clip(np.dot(weights, scores[reps, kind] * foreground[reps, kind]) / total, 0., 1.))
                                       if total else 0.)
                    usable = ((flags[indices, kind] & 11) == 0) & (ledger['endpoint_weight'][indices, kind] > 0)
                    localized = indices[usable]
                    # A finite search-window maximum can point toward a peak
                    # outside that window. Keep its observed coordinate and
                    # support, but require an interior witness before claiming
                    # that this endpoint has a usable localization.
                    has_localization = bool(np.any(usable & interior[indices, kind]))
                    coords = ledger['endpoint_q'][localized if len(localized) else indices, kind]
                    location_reps, location_groups = [], set()
                    for i in order:
                        group = int(ledger['endpoint_group'][i])
                        if (flags[i, kind] & 11) == 0 and ledger['endpoint_weight'][i, kind] > 0 and group not in location_groups:
                            location_reps.append(i); location_groups.add(group)
                    if not location_reps:
                        location_reps = representatives
                    ordered_locations = sorted(location_reps, key=lambda i: (ledger['endpoint_q'][i, kind], int(i)))
                    middle = len(ordered_locations) // 2
                    coordinate_rows = (ordered_locations[middle:middle+1] if len(ordered_locations) % 2 else
                                       ordered_locations[middle-1:middle+1])
                    q = float(np.median(ledger['endpoint_q'][location_reps, kind]))
                    refs = [f'f{int(frames[i])}:l{lane}:y{int(ledger["endpoint_index"][i, 2])}' for i in indices]
                    shared = float(clock['error_px'][clock_index[indices]].max())
                    error = float(ledger['endpoint_error_px'][indices, kind].max())
                    ep = {'id': f'{site_id}:{KINDS[kind]}:{len(site["endpoint_ids"])}', 'site_id': site_id,
                          'kind': KINDS[kind], 'q_px': q if has_localization else None,
                          'observed_q_px': q, 'observed_range_px': [float(coords.min()), float(coords.max())],
                          'score': score, 'weight': total, 'group_count': len(seen),
                          'foreground_supported_score': supported_score,
                          'utility': supported_score * min(1., total / config.support_weight),
                          'shared_motion_error_px': shared, 'fusion_error_px': error,
                          'localization_error_px': meta.get('observation_position_error_px'),
                          'calibration_error_px': meta['endpoint_timing_error_px'][kind],
                          'motion_ambiguous': bool(np.any(clock['status'][clock_index[indices]] == STATUS['ambiguous'])),
                          'motion_frame_ids': sorted(set(int(frames[i]) for i in indices)),
                          'coordinate_observations': [
                              {'observation_id': f'f{int(frames[i])}:l{lane}:y{int(ledger["endpoint_index"][i, 2])}',
                               'frame_id': int(frames[i]), 'q_px': float(ledger['endpoint_q'][i, kind]),
                               'weight': [1, len(coordinate_rows)]} for i in coordinate_rows],
                          'observation_ids': refs,
                          'search_boundary_observation_ids': [ref for ref, i in zip(refs, indices)
                                                              if not interior[i, kind]],
                          'evidence_refs': [f'l{lane}:q{b}:k{kind}' for b in sorted({b for i in indices
                              if ledger['endpoint_weight'][i, kind] > 0
                              for b in (int(np.floor(ledger['endpoint_q'][i, kind] / meta['grid_px'])),
                                        int(np.ceil(ledger['endpoint_q'][i, kind] / meta['grid_px'])))})],
                          'reason': None if has_localization else 'no unclipped, interior endpoint localization'}
                    endpoints.append(ep); site['endpoint_ids'].append(ep['id'])
            sites.append(site)
    return sites, endpoints


def union_ranges(intervals):
    """Union without double counting touching or overlapping pixel intervals."""
    merged = []
    for a, b in sorted(intervals):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([float(a), float(b)])
    return merged


def endpoint_footprints(endpoints, ledger, flags, heights, offsets):
    """The template area shared by all usable observations of an endpoint.

    q runs opposite to screen y. Restore the detection reference offset before
    projecting the downward template rectangle. Unknown or incompatible
    geometry gives no mask; it does not manufacture observed absence.
    """
    lookup = {f'f{int(f)}:l{int(l)}:y{int(y)}': i
              for i, (f, l, y) in enumerate(ledger['endpoint_index'])}
    for endpoint in endpoints:
        k = KINDS.index(endpoint['kind'])
        rows = [lookup[ref] for ref in endpoint['observation_ids']]
        usable = [i for i in rows if flags[i, k] & 11 == 0 and ledger['endpoint_weight'][i, k] > 0]
        endpoint['template_footprint_px'] = None
        if usable and heights[k] is not None:
            tops = ledger['endpoint_q'][usable, k] + offsets[k]
            lower, upper = float(tops.max() - heights[k]), float(tops.min())
            if lower < upper:
                endpoint['template_footprint_px'] = [lower, upper]


def shared_body_footprints(sites, endpoints, config):
    """Candidate-independent overlap, common to supported site interpretations."""
    by_id = {e['id']: e for e in endpoints}
    ranges = []
    for site in sites:
        eligible = [by_id[e] for e in site['endpoint_ids'] if by_id[e]['q_px'] is not None
                    and by_id[e]['score'] >= config.association_similarity
                    and by_id[e]['weight'] >= config.support_weight]
        if not eligible or any(e.get('template_footprint_px') is None for e in eligible):
            continue
        lower = max(e['template_footprint_px'][0] for e in eligible)
        upper = min(e['template_footprint_px'][1] for e in eligible)
        if lower < upper:
            ranges.append([lower, upper])
    return union_ranges(ranges)


def interval_cost(endpoints, missing, body, config):
    """Auditable additive costs; neither similarities nor costs are probabilities."""
    reward = config.body_reward if endpoints else config.body_only_reward
    terms = {'note': config.note_cost,
            'endpoint_support': -config.endpoint_reward * sum(e['utility'] for e in endpoints),
            'additional_endpoint': config.additional_endpoint_cost * max(0, len(endpoints) - 1),
            'missing_endpoint': config.missing_endpoint_cost * missing,
            'body_support': -reward * body['support_fraction'] if body else 0.,
            'body_support_shortfall': reward * (body['support_fraction'] - body['supported_fraction']) if body else 0.,
            'body_absence': config.absence_cost_per_px * body['absent_px'] if body else 0.,
            'body_unavailable': config.unavailable_cost_per_px * body['unavailable_px'] if body else 0.}
    if body is not None and 'endpoint_overlap_supported_px' in body:
        terms['endpoint_body_overlap'] = reward * body['endpoint_overlap_supported_px'] / body['support_scale_px']
    if body is not None:
        # A shorter hypothesis may consume only its share of a continuous run.
        # Use the same independent brightness after template-overlap accounting;
        # absence, unavailable coverage and endpoint evidence keep their costs.
        independent = body['supported_bright_px'] - body.get('endpoint_overlap_supported_px', 0.)
        terms['body_run_allocation'] = reward * independent * (
            1 / body['support_scale_px'] - 1 / body['reward_scale_px'])
    return terms


def interval_candidates(lane, sites, endpoints, body, config):
    by_id = {e['id']: e for e in endpoints}
    by_site = {s['id']: s for s in sites}
    tokens = []
    for s in sites:
        es = [by_id[e] for e in s['endpoint_ids']]
        # Full alternative footprint is conservative: competing interpretations
        # cannot reuse a physical endpoint even when their template tops differ.
        lower = min(e['observed_range_px'][0] for e in es)
        upper = max(e['observed_range_px'][1] for e in es)
        phase = {'tail': 0, 'tap': 1, 'head': 2}[s['primary_kind']]
        s['tokens'] = [(lower, phase, s['id'], 0), (upper, phase, s['id'], 1)]
        tokens.extend(s['tokens'])
    # Missing ends reserve body extents too, instead of inventing endpoints.
    for e in endpoints:
        e['extent_px'] = body.extent(e['observed_q_px'], -1 if e['kind'] == 'tail' else 1,
                                    config.endpoint_radius_px * 2)
        if e['kind'] != 'tap':
            tokens.append((e['extent_px'], 1, e['id'], 0))
    for i, (a, b) in enumerate(body.runs):
        tokens.extend([(float(a), -1, f'body{i}', 0), (float(b), 3, f'body{i}', 1)])
    tokens = sorted(set(tokens)); positions = {v: i for i, v in enumerate(tokens)}
    for s in sites:
        s['lo'], s['hi'] = [positions[v] for v in s.pop('tokens')]
    candidates = []
    token_coordinates = np.array([t[0] for t in tokens])

    def add(kind, start, end, bounds=None):
        es = [e for e in (start, end) if e is not None]
        if kind == 'longnote' and start and end:
            if start['site_id'] == end['site_id'] or end['observed_q_px'] <= start['observed_q_px']:
                return
        q0 = start['observed_q_px'] if start else bounds[0]
        q1 = end['observed_q_px'] if end else bounds[1] if bounds else q0
        if kind == 'longnote' and q1 <= q0:
            # A point-like incomplete hypothesis is still retained.
            if start and end:
                return
            q0, q1 = min(q0, q1), max(q0, q1)
        summary = body.summary(q0, q1) if kind == 'longnote' else None
        missing = (int(start is None or start['q_px'] is None)
                   + (int(end is None or end['q_px'] is None) if kind == 'longnote' else 0))
        terms = interval_cost(es, missing, summary, config)
        footprint = [x for e in es for x in (by_site[e['site_id']]['lo'], by_site[e['site_id']]['hi'])]
        if bounds:
            footprint.extend((int(np.searchsorted(token_coordinates, q0, side='left')),
                              int(np.searchsorted(token_coordinates, q1, side='right')) - 1))
        c = {'id': f'l{lane}:c{len(candidates)}', 'lane': lane, 'type': kind,
             'start': start['id'] if start else None, 'end': end['id'] if end else None,
             'start_px': start['q_px'] if start else None, 'end_px': end['q_px'] if end else None,
             'lo': min(footprint), 'hi': max(footprint), 'cost': math.fsum(terms.values()),
             'cost_terms': terms,
             'missing_endpoints': missing, 'body': summary,
             'endpoint_site_ids': [e['site_id'] for e in es]}
        for name, endpoint, at_edge in (
                ('start', start, bool(len(body.q) and q0 <= body.q[0])),
                ('end', end, bool(len(body.q) and q1 >= body.q[-1] + body.grid))):
            c[f'missing_{name}_reason'] = (
                None if name == 'end' and kind == 'tap' or endpoint and endpoint['q_px'] is not None
                else 'clipped_or_unusable_observation' if endpoint
                else 'coverage_boundary' if at_edge else 'missing_endpoint')
        candidates.append(c)

    heads = [e for e in endpoints if e['kind'] == 'head']
    tails = sorted([e for e in endpoints if e['kind'] == 'tail'], key=lambda e: e['observed_q_px'])
    for e in endpoints:
        if e['kind'] == 'tap':
            add('tap', e, None)
        elif e['kind'] == 'head':
            add('longnote', e, None, (e['observed_q_px'], max(e['observed_q_px'], e['extent_px'])))
        else:
            add('longnote', None, e, (min(e['extent_px'], e['observed_q_px']), e['observed_q_px']))
    omitted = 0
    # Nonnegative integrated penalties give a safe lower bound independent of
    # future endpoint quality. Once above the bound, no later pair can lie in
    # the ambiguity band. All original endpoints remain available for re-decode.
    budget = (2 * config.endpoint_reward + config.body_reward + config.ambiguity_cost
              - config.note_cost - config.additional_endpoint_cost)
    tail_q = np.array([e['observed_q_px'] for e in tails])
    for head in heads:
        first = int(np.searchsorted(tail_q, head['observed_q_px'], side='right'))
        for j in range(first, len(tails)):
            tail = tails[j]
            summary = body.summary(head['observed_q_px'], tail['observed_q_px'])
            penalty = (config.absence_cost_per_px * summary['absent_px']
                       + config.unavailable_cost_per_px * summary['unavailable_px'])
            if penalty > budget + 1e-10:
                omitted += len(tails) - j
                break
            add('longnote', head, tail)
    for a, b in body.runs:
        if b - a >= body.grid:
            add('longnote', None, None, (a, b))
    return tokens, candidates, omitted


def review_events(lane, candidates, endpoints, config, best):
    by_id = {e['id']: e for e in endpoints}
    active = sorted((c for c in candidates if c['competitive']), key=lambda c: (c['lo'], c['hi'], c['id']))
    components = []
    for c in active:
        if not components or c['lo'] > components[-1][0]:
            components.append([c['hi'], [c]])
        else:
            components[-1][0] = max(components[-1][0], c['hi'])
            components[-1][1].append(c)
    events = []
    for _, group in components:
        selected = [c for c in group if c['selected']]
        alternatives = [c for c in group if not c['selected']]
        alt_cost = min([c['path_cost'] for c in alternatives] +
                       [c['exclusion_cost'] for c in selected], default=best)
        missing = any(c['missing_endpoints'] for c in selected)
        motion_ambiguous = any(by_id[e]['motion_ambiguous'] for c in group for e in (c['start'], c['end']) if e)
        motion_ambiguous |= any(c['body'] and c['body']['motion_ambiguous'] for c in group)
        ambiguous = bool(alternatives or alt_cost - best <= config.ambiguity_cost + 1e-10 or motion_ambiguous)
        status = 'incomplete' if missing else 'ambiguous' if ambiguous else 'resolved'
        # These are inspectable properties of competing hypotheses, not labels
        # claiming that a physical note is missing or that an occlusion occurred.
        causes = set()
        if len({c['type'] for c in group}) > 1:
            causes.add('kind_conflict')
        for name in ('start', 'end'):
            for kind in KINDS:
                eps = {c[name]: by_id[c[name]] for c in group if c[name] and by_id[c[name]]['kind'] == kind}
                if len({e['observed_q_px'] for e in eps.values()}) > 1:
                    causes.add('coordinate_alternatives')
                if len({e['site_id'] for e in eps.values()}) > 1:
                    causes.add('site_alternatives')
            causes.update(c[f'missing_{name}_reason'] for c in group if c[f'missing_{name}_reason'])
        if any(c['body'] and c['body']['bright_px'] < config.body_support_px for c in group):
            causes.add('limited_body_support')
        if any(c['body'] and c['body']['bright_px'] - c['body']['supported_bright_px'] > 1e-9 for c in group):
            causes.add('limited_body_observation_support')
        if any(c['start'] is None and c['end'] is None for c in group):
            causes.add('body_only_hypothesis')
        if not selected or any(c['exclusion_cost'] - best <= config.ambiguity_cost + 1e-10 for c in selected):
            causes.add('selection_alternative')
        if motion_ambiguous:
            causes.add('motion_alternatives')
        if any(by_id[e].get('search_boundary_observation_ids') for c in group for e in (c['start'], c['end']) if e):
            causes.add('search_window_boundary')
        events.append({'id': f'l{lane}:e{len(events)}', 'lane': lane, 'status': status,
                       'selected_candidate_ids': [c['id'] for c in selected],
                       'candidate_ids': [c['id'] for c in group],
                       'selected_cost': best, 'alternative_cost': float(alt_cost),
                       'cost_scope': 'whole lane, including noise',
                       'review_causes': sorted(causes),
                       'has_alternatives': ambiguous, 'motion_ambiguous': bool(motion_ambiguous),
                       'reason': ('missing or clipped endpoint' if missing else
                                  'near-cost interpretations, noise, or motion alternatives' if ambiguous else
                                  'pixel interval selected; timing is separate')})
    return events


def summarize_lane(lane, sites, endpoints, body, config):
    tokens, candidates, omitted = interval_candidates(lane, sites, endpoints, body, config)
    best = ordered_dp(len(tokens), candidates, config.ambiguity_cost)
    events = review_events(lane, candidates, endpoints, config, best)
    return {'format': 'ez2rc.decode-lane', 'schema_version': 1, 'lane': lane,
            'body_support_runs_px': [[float(a), float(b)] for a, b in body.runs],
            **({'shared_body_footprints_px': body.endpoint_footprints} if body.endpoint_footprints is not None else {}),
            'sites': sites, 'endpoints': endpoints, 'candidates': candidates, 'events': events,
            'best_cost': best, 'token_count': len(tokens), 'noise_dominated_pair_count': omitted,
            'boundary': {'finalized': True, 'open_candidate_ids': [c['id'] for c in candidates if c['missing_endpoints']],
                         'policy': 'all coordinate tiles retained until lane inference; no storage-edge endpoints'}}


def _lane_input(observations, fusion, lane):
    # Only scalar endpoint records and one lane of cells remain resident. Dense
    # per-frame images/projections are discarded as each source chunk is read.
    obs = {k: [] for k in ('candidate_index', 'scores', 'flags', 'positions')}
    foreground = []
    parameters = observations.manifest['key']['parameters']
    shapes = template_shapes(parameters)
    algorithm = parameters.get('algorithm', {})
    profile_lane = observations.manifest['metadata']['profile']['lanes'][lane]
    for chunk in observations.chunks():
        keep = chunk['candidate_index'][:, 1] == lane
        for k in obs:
            obs[k].append(chunk[k][keep])
        if shapes is not None and 'brightness_threshold' in algorithm:
            foreground.append(endpoint_foreground(chunk, lane, [shapes[lane][k][0] for k in KINDS],
                                                   profile_lane['roi'][1], algorithm['brightness_threshold']))
    obs = {k: np.concatenate(v) for k, v in obs.items()}
    ledger, cells = {}, {}
    for chunk in fusion.chunks():
        for k, value in chunk.items():
            target = ledger if k.startswith('endpoint') else cells
            keep = chunk['endpoint_index'][:, 1] == lane if target is ledger else chunk['index'][:, 0] == lane
            target.setdefault(k, []).append(value[keep])
    ledger = {k: np.concatenate(v) for k, v in ledger.items()}
    cells = {k: np.concatenate(v) for k, v in cells.items()}
    if not np.array_equal(obs['candidate_index'], ledger['endpoint_index']):
        raise ValueError('decode endpoint ledger does not match original observations')
    if foreground:
        ledger['endpoint_foreground'] = np.concatenate(foreground)
    # A maximum at the edge of a seed's search window is not evidence of a
    # second local peak: another seed may see the higher adjacent maximum.
    # Preserve it, but do not let it split a supported site. This uses raw
    # screen coordinates, independently of kind-specific timing corrections.
    radius = observations.manifest['key']['parameters'].get('algorithm', {}).get('search_radius_px')
    if radius is not None:
        top = observations.manifest['metadata']['profile']['lanes'][lane]['roi'][1]
        seeds = obs['candidate_index'][:, 2] + top
        ledger['endpoint_peak_interior'] = np.abs(obs['positions'][:, :, 1] - seeds[:, None]) < radius
    order = np.argsort(cells['index'][:, 1], kind='stable')
    return ledger, obs['scores'], obs['flags'], {k: v[order] for k, v in cells.items()}


def _infer_lane(observations, fusion, lane, clock, meta, config):
    ledger, scores, flags, cells = _lane_input(observations, fusion, lane)
    sites, endpoints = endpoint_sites(lane, ledger, scores, flags, clock, meta, config)
    shapes = template_shapes(observations.manifest['key']['parameters'])
    footprints = None
    if shapes is not None:
        heights = [shapes[lane][kind][0] for kind in KINDS]
        offsets = [meta['calibration'][kind].get('offset_px', 0.)
                   if meta['calibration'][kind]['status'] == 'verified' else 0. for kind in KINDS]
        endpoint_footprints(endpoints, ledger, flags, heights, offsets)
        footprints = shared_body_footprints(sites, endpoints, config)
    return summarize_lane(lane, sites, endpoints,
                          BodyEvidence(cells, meta['grid_px'], config, clock, footprints), config)


def infer(observations: Cache, motion: Cache, fusion: Cache, config: DecodeConfig, root: Path):
    if (motion.manifest['key']['parents'] != [observations.id]
            or fusion.manifest['key']['parents'] != [observations.id, motion.id]):
        raise ValueError('decode parent lineage mismatch')
    declaration = observations.manifest['metadata'].get('position_calibration')
    if fusion.manifest['metadata'].get('position_calibration') != declaration:
        raise ValueError('decode position declaration differs between parents')
    key = cache_key(observations, motion, fusion, config)
    try:
        cached = open_decode(root / 'decode' / digest(key), key)
        if cached.manifest['metadata'].get('position_calibration') != declaration:
            raise CacheError('decode position declaration differs from its observation parent')
        return cached, True
    except CacheError:
        pass
    clock = timeline(motion)
    meta = {**fusion.manifest['metadata'],
            'repeat_group_cap': fusion.manifest['key']['parameters']['repeat_group_cap']}
    counts = dict.fromkeys(('resolved', 'ambiguous', 'incomplete'), 0)
    candidate_count = endpoint_count = 0
    with CacheWriter(root, key) as writer:
        for lane in range(meta['lane_count']):
            result = _infer_lane(observations, fusion, lane, clock, meta, config)
            for e in result['events']:
                counts[e['status']] += 1
            candidate_count += len(result['candidates']); endpoint_count += len(result['endpoints'])
            writer.append(payload=np.frombuffer(canonical(result), dtype=np.uint8))
            # Release this lane before constructing the next or validating the
            # published payload. No inference boundary is introduced within it.
            del result
        return writer.commit({'observation_cache_id': observations.id, 'motion_cache_id': motion.id,
                              'fusion_cache_id': fusion.id, 'lane_count': meta['lane_count'],
                              'event_status_counts': counts, 'candidate_count': candidate_count,
                              'endpoint_count': endpoint_count,
                              **({'position_calibration': dict(meta['position_calibration'])} if 'position_calibration' in meta else {}),
                              'grid_px': meta['grid_px'],
                              'coordinate': 'relative scroll pixels; never milliseconds or beats',
                              'cost_semantics': 'heuristic additive costs, not probabilities',
                              'endpoint_support_policy': 'correlation_with_matched_patch_brightness_support',
                              'body_overlap_policy': ('shared_supported_template_footprints'
                                  if template_shapes(observations.manifest['key']['parameters']) is not None
                                  else 'unavailable_template_geometry'),
                              'body_reward_policy': 'shared_contiguous_run_scale',
                              'motion_policy': 'representative fusion trajectory; original alternatives in motion parent',
                              'boundary_policy': 'lane-wide retained evidence; storage edges never create endpoints',
                              'timing_status': 'pending'}, validate=open_decode), False


def _lane_payload(chunk):
    data = chunk.get('payload')
    if data is None or data.dtype != np.uint8 or data.ndim != 1:
        raise CacheError('invalid decode payload encoding')
    def reject(value):
        raise ValueError(f'non-finite decode value: {value}')
    # Decode the array buffer directly, without a second full bytes payload.
    return json.loads(codecs.decode(data, 'utf-8'), parse_constant=reject)


def lanes(cache):
    for chunk in cache.chunks():
        data = _lane_payload(chunk)
        del chunk
        yield data
        del data


def _validate_lane(lane, data, config, grid):
    """Keep all candidate indexes and validation state inside this lane's scope."""
    counts = dict.fromkeys(('resolved', 'ambiguous', 'incomplete'), 0)
    if data['format'] != 'ez2rc.decode-lane' or data['schema_version'] != 1 or data['lane'] != lane:
        raise CacheError('invalid decoded lane')
    runs = data.get('body_support_runs_px')
    if (not isinstance(runs, list) or any(not isinstance(r, list) or len(r) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in r)
            or r[0] >= r[1] for r in runs)
            or any(a[1] >= b[0] for a, b in zip(runs, runs[1:]))):
        raise CacheError('invalid body support runs')
    reward_scale = BodyRunScale(runs)
    eps = {e['id']: e for e in data['endpoints']}
    cs = {c['id']: c for c in data['candidates']}
    sites = {s['id']: s for s in data['sites']}
    if runs != sorted(c['body']['range_px'] for c in cs.values()
                      if c['start'] is None and c['end'] is None):
        raise CacheError('body support runs differ from retained body hypotheses')
    if (len(eps) != len(data['endpoints']) or len(cs) != len(data['candidates'])
            or len(sites) != len(data['sites'])):
        raise CacheError('duplicate decode identities')
    owners = {}
    memberships = {sid: set() for sid in sites}
    for e in eps.values():
        if e['site_id'] not in memberships:
            raise CacheError('dangling endpoint site')
        memberships[e['site_id']].add(e['id'])
        if 'shared_body_footprints_px' in data:
            span = e['template_footprint_px']
            if span is not None and (not isinstance(span, list) or len(span) != 2
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in span) or span[0] >= span[1]):
                raise CacheError('invalid endpoint template footprint')
    if ('shared_body_footprints_px' in data
            and data['shared_body_footprints_px'] != shared_body_footprints(data['sites'], data['endpoints'], config)):
        raise CacheError('invalid shared body footprints')
    for s in sites.values():
        if (not s['endpoint_ids'] or len(set(s['endpoint_ids'])) != len(s['endpoint_ids'])
                or set(s['endpoint_ids']) != memberships[s['id']]
                or not 0 <= s['anchor_range_px'][1] - s['anchor_range_px'][0] <= 2 * config.endpoint_radius_px):
            raise CacheError('invalid endpoint site')
    for e in eps.values():
        if (e['kind'] not in KINDS or e['site_id'] not in sites
                or e['id'] not in sites[e['site_id']]['endpoint_ids']
                or e['observed_range_px'][0] > e['observed_range_px'][1]
                or not 0 <= e['utility'] <= 1 or e['weight'] < 0
                or not 0 <= e['foreground_supported_score'] <= 1
                or e['utility'] != e['foreground_supported_score'] * min(1., e['weight']/config.support_weight)
                or e['shared_motion_error_px'] < 0 or e['fusion_error_px'] < 0
                or not e['observed_range_px'][0] <= e['observed_q_px'] <= e['observed_range_px'][1]
                or not set(e['search_boundary_observation_ids']) <= set(e['observation_ids'])):
            raise CacheError('invalid endpoint hypothesis')
        for ref in e['observation_ids']:
            if ref in owners and owners[ref] != e['site_id']:
                raise CacheError('observation shared between endpoint sites')
            owners[ref] = e['site_id']
        support = e['coordinate_observations']
        if (not isinstance(support, list) or len(support) not in (1, 2)
                or len({p['observation_id'] for p in support}) != len(support)
                or any(p['observation_id'] not in e['observation_ids']
                       or type(p['frame_id']) is not int or p['frame_id'] not in e['motion_frame_ids']
                       or not p['observation_id'].startswith(f"f{p['frame_id']}:l{lane}:")
                       or p['weight'] != [1, len(support)]
                       or type(p['q_px']) not in (int, float) or not math.isfinite(p['q_px']) for p in support)
                or sum(p['q_px'] for p in support) / len(support) != e['observed_q_px']):
            raise CacheError('invalid endpoint coordinate observations')
    used, last = set(), -1
    for c in sorted(cs.values(), key=lambda c: c['lo']):
        if (not np.isfinite(c['cost']) or not 0 <= c['lo'] <= c['hi'] < data['token_count']
                or c['type'] not in ('tap', 'longnote') or c['lane'] != lane):
            raise CacheError('invalid interval candidate')
        for name in ('start', 'end'):
            if c[name] is not None and c[name] not in eps:
                raise CacheError('dangling endpoint reference')
            if c[f'{name}_px'] != (eps[c[name]]['q_px'] if c[name] is not None else None):
                raise CacheError('interval coordinate disagrees with endpoint')
        endpoint_sites = [eps[e]['site_id'] for e in (c['start'], c['end']) if e]
        if (c['endpoint_site_ids'] != endpoint_sites or len(set(endpoint_sites)) != len(endpoint_sites)
                or c['start'] and eps[c['start']]['kind'] != ('tap' if c['type'] == 'tap' else 'head')
                or c['end'] and eps[c['end']]['kind'] != 'tail'):
            raise CacheError('invalid endpoint resources or kinds')
        if c['type'] == 'tap' and (c['end'] is not None or c['end_px'] is not None):
            raise CacheError('tap with end')
        if c['start_px'] is not None and c['end_px'] is not None and c['end_px'] <= c['start_px']:
            raise CacheError('reversed longnote')
        if (c['body'] is None) != (c['type'] == 'tap'):
            raise CacheError('invalid interval body evidence')
        if c['body'] is not None:
            if ('endpoint_overlap_supported_px' in c['body']) != ('shared_body_footprints_px' in data):
                raise CacheError('inconsistent body overlap policy')
            _validate_body(c['body'], config, grid)
            a, b = c['body']['range_px']
            if c['body']['reward_scale_px'] != reward_scale.scale(a, b, c['body']['support_scale_px']):
                raise CacheError('body reward scale differs from shared runs')
            if 'shared_body_footprints_px' in data:
                a, b = c['body']['range_px']
                masked_length = sum(max(0., min(b, hi) - max(a, lo))
                                    for lo, hi in data['shared_body_footprints_px'])
                if c['body']['endpoint_overlap_supported_px'] > masked_length + 1e-8:
                    raise CacheError('body overlap exceeds shared footprint coverage')
        terms = interval_cost([eps[e] for e in (c['start'], c['end']) if e],
                              c['missing_endpoints'], c['body'], config)
        if c['cost_terms'] != terms or c['cost'] != math.fsum(terms.values()):
            raise CacheError('invalid interval cost terms')
        if c['selected']:
            resources = set(c['endpoint_site_ids'])
            if c['lo'] <= last or resources & used:
                raise CacheError('conflicting selected intervals')
            last = c['hi']; used |= resources
    if ordered_dp(data['token_count'], data['candidates'], config.ambiguity_cost, verify=True) != data['best_cost']:
        raise CacheError('invalid optimum cost')
    emitted = []
    for event in data['events']:
        counts[event['status']] += 1
        if (not set(event['candidate_ids']) <= cs.keys()
                or not set(event['selected_candidate_ids']) <= set(event['candidate_ids'])
                or any(not cs[c]['competitive'] for c in event['candidate_ids'])
                or any(not cs[c]['selected'] for c in event['selected_candidate_ids'])):
            raise CacheError('dangling interval reference')
        emitted.extend(event['candidate_ids'])
        if event['status'] == 'resolved' and (len(event['selected_candidate_ids']) != 1 or
                any(cs[c]['missing_endpoints'] for c in event['selected_candidate_ids'])):
            raise CacheError('resolved pixel event lacks endpoints')
    if len(emitted) != len(set(emitted)) or set(emitted) != {c['id'] for c in cs.values() if c['competitive']}:
        raise CacheError('competitive intervals hidden or duplicated')
    if data['events'] != review_events(lane, data['candidates'], data['endpoints'], config, data['best_cost']):
        raise CacheError('invalid pixel review diagnostics')
    return counts, len(cs), len(eps)


def _validate_body(body, config, grid):
    """Check derived support separately from darkness and unavailable coverage."""
    start, end = body['range_px']
    length = end - start
    bright, supported = body['bright_px'], body['supported_bright_px']
    if 'endpoint_overlap_supported_px' in body:
        overlap = body['endpoint_overlap_supported_px']
        if not math.isfinite(overlap) or not 0 <= overlap <= supported:
            raise CacheError('invalid endpoint body overlap')
    dark, unavailable, scale = body['absent_px'], body['unavailable_px'], body['support_scale_px']
    values = (start, end, bright, supported, dark, unavailable, scale,
              body['support_fraction'], body['supported_fraction'], body['cost'], body['reward_scale_px'])
    if (not all(math.isfinite(v) for v in values) or length < 0
            or min(bright, supported, dark, unavailable) < 0 or supported > bright
            or bright + dark + unavailable > length + 1e-8
            or scale != max(length, grid, config.body_support_px)
            or body['reward_scale_px'] < scale
            or body['support_fraction'] != bright / scale
            or body['supported_fraction'] != supported / scale
            or body['cost'] != float(config.absence_cost_per_px * dark
                                    + config.unavailable_cost_per_px * unavailable
                                    - config.body_reward * body['supported_fraction'])):
        raise CacheError('invalid body observation support')


def open_decode(path, expected_key=None):
    try:
        counts = dict.fromkeys(('resolved', 'ambiguous', 'incomplete'), 0)
        candidate_count = endpoint_count = lane_count = 0

        def check_chunk(manifest, lane, chunk):
            nonlocal candidate_count, endpoint_count, lane_count
            key, meta = manifest['key'], manifest['metadata']
            if (key['stage'] != 'decode' or key['algorithm_version'] != VERSION
                    or key['parents'] != [meta[f'{s}_cache_id'] for s in ('observation', 'motion', 'fusion')]):
                raise CacheError('unsupported decode cache/lineage')
            if meta.get('body_reward_policy') != 'shared_contiguous_run_scale':
                raise CacheError('unsupported body reward policy')
            if meta.get('endpoint_support_policy') != 'correlation_with_matched_patch_brightness_support':
                raise CacheError('unsupported endpoint support policy')
            grid = meta['grid_px']
            if not math.isfinite(grid) or grid <= 0:
                raise CacheError('invalid decode grid')
            payload = _lane_payload(chunk)
            policy = meta.get('body_overlap_policy')
            if (policy not in ('shared_supported_template_footprints', 'unavailable_template_geometry')
                    or ('shared_body_footprints_px' in payload) != (policy == 'shared_supported_template_footprints')):
                raise CacheError('decode body overlap policy differs from its lane evidence')
            if 'position_calibration' in meta:
                declared = read_position_calibration(meta['position_calibration'])
                if any(canonical(e['localization_error_px']) != canonical(declared.error_px) for e in payload['endpoints']):
                    raise CacheError('decoded localization error differs from its declaration')
            statuses, candidates, endpoints = _validate_lane(lane, payload, DecodeConfig(**key['parameters']), grid)
            for status, count in statuses.items():
                counts[status] += count
            candidate_count += candidates; endpoint_count += endpoints; lane_count += 1

        # Integrity, shape and lane semantics share one decompression. The
        # callback's parsed lane and indexes die before the next payload read.
        cache = Cache(path, expected_key, validate_chunk=check_chunk)
        meta = cache.manifest['metadata']
        if (lane_count != meta['lane_count'] or counts != meta['event_status_counts']
                or candidate_count != meta['candidate_count'] or endpoint_count != meta['endpoint_count']):
            raise CacheError('decode count mismatch')
        return cache
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        raise CacheError(f'invalid decode: {exc}') from exc


def _lane_output_events(data):
    events, pixels = [], []
    cs = {c['id']: c for c in data['candidates']}
    eps = {e['id']: e for e in data['endpoints']}
    for p in data['events']:
        selected = [cs[c] for c in p['selected_candidate_ids']]
        endpoint_ids = {e for c in p['candidate_ids'] for e in (cs[c]['start'], cs[c]['end']) if e}
        observations = sorted({o for e in endpoint_ids for o in eps[e]['observation_ids']})
        reason = p['reason'] + '; video timing, independent calibration and musical origin pending'
        events.append(Event(p['id'], p['lane'], 'ambiguous' if p['status'] == 'ambiguous' else 'incomplete',
                            selected[0]['type'] if len(selected) == 1 else None,
                            observation_ids=tuple(observations), candidate_ids=tuple(p['candidate_ids']),
                            selected_cost=p['selected_cost'], alternative_cost=p['alternative_cost'], reason=reason))
        pixels.append({**p, 'selected_intervals': selected,
                       'alternatives': [cs[c] for c in p['candidate_ids'] if not cs[c]['selected']],
                       'endpoints': [eps[e] for e in sorted(endpoint_ids)]})
    return events, pixels


def output_events(cache):
    """Bridge pixel results to the existing, deliberately unresolved time API."""
    events, pixels = [], []
    for data in lanes(cache):
        lane_events, lane_pixels = _lane_output_events(data)
        events.extend(lane_events); pixels.extend(lane_pixels)
        del data
    return tuple(events), pixels
