"""Procedural evidence in the same units and array layout as fusion caches."""
import numpy as np

from ez2rc.v2.config import DecodeConfig
from ez2rc.v2.decode import BodyEvidence, endpoint_sites, summarize_lane


def scene(points, bodies=(), *, unavailable=(), occluded=(), absent=(), bounds=(0, 200),
          flags=None, errors=(2., 4., 6.), config=None, ambiguous_motion=False,
          calibration_errors=(None, None, None)):
    config = config or DecodeConfig()
    n = len(points)
    index = np.array([(f, 0, i) for f in range(3) for i in range(n)], np.int64).reshape(-1, 3)
    q = np.array([p[0] if isinstance(p[0], (tuple, list)) else [p[0]] * 3 for p in points], float).reshape(-1, 3)
    scores = np.array([([.05 if k != p[1] else .99 for k in range(3)]
                        if isinstance(p[1], int) else p[1]) + [.0] for p in points], float).reshape(-1, 4)
    marks = np.zeros((n, 3), np.uint8) if flags is None else np.asarray(flags, np.uint8)
    weights = np.where((marks & 1) != 0, .25, .5)
    ledger = {'endpoint_index': index, 'endpoint_q': np.tile(q, (3, 1)),
              'endpoint_weight': np.tile(weights, (3, 1)),
              'endpoint_error_px': np.repeat(np.array(errors), n)[:, None] * np.ones((n * 3, 3)),
              'endpoint_group': index[:, 0]}
    clock = {'frame_id': np.arange(3), 'error_px': np.array(errors),
             'status': np.full(3, 3 if ambiguous_motion else 1)}
    meta = {'grid_px': 1., 'endpoint_timing_error_px': list(calibration_errors),
            'observation_position_error_px': None}
    bins = np.arange(*bounds)
    cells = {'index': np.column_stack((np.zeros(len(bins), int), bins)),
             'score_sum': np.zeros((len(bins), 4)), 'weight_sum': np.zeros((len(bins), 4)),
             'visibility_weight': np.zeros((len(bins), 4, 5))}
    cells['weight_sum'][:, 3] = 1
    cells['visibility_weight'][:, 3, 1] = 1
    for a, b in bodies:
        cells['score_sum'][(bins >= a) & (bins < b), 3] = .6
    for a, b in absent:
        cells['score_sum'][(bins >= a) & (bins < b), 3] = 0
    for a, b in unavailable:
        at = (bins >= a) & (bins < b)
        cells['score_sum'][at, 3] = 0
        cells['weight_sum'][at, 3] = 0
        cells['visibility_weight'][at, 3] = 0
    for a, b in occluded:
        at = (bins >= a) & (bins < b)
        cells['score_sum'][at, 3] = 0
        cells['visibility_weight'][at, 3, 1] = 0
        cells['visibility_weight'][at, 3, 4] = 1
    sites, eps = endpoint_sites(0, ledger, np.tile(scores, (3, 1)), np.tile(marks, (3, 1)), clock, meta, config)
    return summarize_lane(0, sites, eps, BodyEvidence(cells, 1., config), config)


def selected(data):
    return [(c['type'], c['start_px'], c['end_px']) for c in data['candidates'] if c['selected']]
