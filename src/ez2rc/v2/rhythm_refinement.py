"""Monotone grid inference from shared geometry, without nearest-grid snapping.

Only a label already unique over the entire current geometric domain may
become an additional constraint. Physical alternatives need an explicit bound
identity review before they can provide anchors.
The resulting constraints must remain jointly feasible for one common scale.
"""
from collections import defaultdict

from .contracts import ratio, read_ratio
from .endpoint_ranges import closed
from .rhythm import grid_candidates
from .spatial_constraints import ConstantSpacingMap

VERSION = 'shared-geometry-grid-refinement-4'


def refine_grid(model, endpoints, grid, *, reviewed_candidate_ids=()):
    reviewed = frozenset(reviewed_candidate_ids)
    if not reviewed <= {p['candidate_id'] for p in endpoints}:
        raise ValueError('reviewed candidate has no geometric endpoints')
    def eligible(point):
        return point['pixel_status']=='resolved' or point['candidate_id'] in reviewed
    added, iterations, conflicts = set(),[],[]
    while model.status == 'conditional':
        possibilities, groups = {},defaultdict(list)
        for point in endpoints:
            identity, node = point['id'],point['node']
            domain = model.project(node,point['offset'],point['beat_shift'])
            candidates = grid_candidates(closed(*domain),grid) if domain else None
            possibilities[identity] = ([read_ratio(p['beat']) for p in candidates
                if model.scales_for_beat(node,read_ratio(p['beat']),point['offset'],point['beat_shift']) is not None]
                if candidates is not None else None)
            groups[point['candidate_id']].append(point)
        valid = set()
        for rows in groups.values():
            if len(rows)==1:
                if possibilities[rows[0]['id']]:valid.add(rows[0]['id'])
            else:
                points = {p['side']:p for p in rows}
                starts, ends = possibilities[points['start']['id']],possibilities[points['end']['id']]
                if starts and ends and any(end>start for start in starts for end in ends):
                    valid.update(p['id'] for p in rows)
            if all(eligible(p) for p in rows) and any(
                    possibilities[p['id']] == [] for p in rows):
                conflicts.append({'candidate_id':rows[0]['candidate_id'],
                                  'reason':'no_supported_grid_label',
                                  'endpoint_ids':[p['id'] for p in rows if possibilities[p['id']]==[]]})
            elif len(rows)>1 and all(possibilities[p['id']] for p in rows) and not any(
                    p['id'] in valid for p in rows) and all(eligible(p) for p in rows):
                conflicts.append({'candidate_id':rows[0]['candidate_id'],
                                  'reason':'no_positive_grid_duration','endpoint_ids':[p['id'] for p in rows]})
        if conflicts:
            break
        seeds = [p for p in endpoints if p['id'] not in added and p['id'] in valid
                 and eligible(p) and possibilities[p['id']] is not None
                 and len(possibilities[p['id']])==1]
        if not seeds:
            break
        proof, additions = [], []
        previous_scale = model.scale
        for point in seeds:
            identity, node = point['id'],point['node']
            virtual = 'grid:'+identity
            if virtual in model.graph.edges:
                raise ValueError('grid endpoint coordinate identity collision')
            beat = possibilities[identity][0]
            lower, upper = point['offset']
            additions.append((virtual,node,lower,upper,beat-point['beat_shift'],'unique-grid:'+identity))
            proof.append({'endpoint_id':identity,'candidate_id':point['candidate_id'],
                          'automatic_pixel_status':point['pixel_status'],
                          'physical_basis':'manual_review' if point['candidate_id'] in reviewed else 'automatic_selection',
                          'beat':ratio(beat),'outer_domain_before':closed(*model.project(node,point['offset'],point['beat_shift']))})
            added.add(identity)
        model = model.with_leaf_anchors(additions)
        iterations.append({'scale_before':closed(*previous_scale),'added_unique_labels':proof,
                           'status_after':model.status,
                           'scale_bounds_after':[ratio(v) for v in model.scale] if model.scale else None})
    return model,{'version':VERSION,'status':'conflict' if conflicts else model.status,
                  'conflicts':conflicts,'iterations':iterations,'anchored_endpoint_ids':sorted(added),
                  'accepted_note_count':0,
                  'semantics':'Only previously unique labels from resolved or explicitly reviewed physical selections add constraints. Automatic states are preserved. All labels share the original observation graph and one positive spatial scale. Calibration validity remains separate.'}
