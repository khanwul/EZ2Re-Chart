"""Explicit physical identity reviews bound to immutable visual evidence.

A review confirms an existing selected lane, kind and endpoint connection.
It supplies no coordinates, calibration, musical labels or clock information.
Witnesses are a subset of the endpoint observations, not a certification of
every member of a fused cluster. Automatic decode states remain unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .config import string, table
from .decode import lanes
from .storage import canonical, reject_constant

VERSION = 'physical-identity-review-1'
PARENTS = ('observation_cache_id', 'motion_cache_id', 'decode_cache_id')


def checksum(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('physical review requires lowercase SHA-256 identities')
    return value


def identifiers(value):
    if not isinstance(value, list) or not value:
        raise ValueError('physical review requires a nonempty identifier list')
    for item in value:
        string(item, 'physical review identifier')
    if len(value) != len(set(value)):
        raise ValueError('duplicate physical review identifier')
    return value


@dataclass(frozen=True)
class PhysicalReview:
    """Validated review bytes; copies returned to callers cannot alter a review."""
    document: bytes
    input_sha256: str
    parents: tuple[str, str, str]
    video_sha256: str
    selected: tuple[tuple, ...]

    def snapshot(self):
        return json.loads(self.document)

    def receipt(self):
        value = self.snapshot()
        return {'version': VERSION, 'input_sha256': self.input_sha256,
                **dict(zip(PARENTS, self.parents)), 'video_sha256': self.video_sha256,
                'reviewer': value['reviewer'], 'review_context': value['review_context'],
                'candidate_ids': [d[0] for d in self.selected]}

    def check_parents(self, observations, movement, decoded):
        if (self.parents != (observations.id, movement.id, decoded.id)
                or self.video_sha256 != observations.manifest['metadata']['video_sha256']):
            raise ValueError('physical review parent lineage mismatch')

    def check_relative(self, relative):
        if (tuple(relative['reference'][k] for k in PARENTS) != self.parents
                or relative.get('physical_review') != self.receipt()):
            raise ValueError('relative projection is not bound to the physical review')
        found = {r['candidate_id'] for r in relative['candidates'] if self.basis(r) is not None}
        if found != {d[0] for d in self.selected}:
            raise ValueError('physical review selected candidate missing from projection')

    def basis(self, row):
        for cid, event, lane, kind, start, end in self.selected:
            if cid != row['candidate_id']:
                continue
            if ((row['event_id'], row['lane'], row['type']) != (event, lane, kind)
                    or row['endpoints']['start']['endpoint_id'] != start
                    or row['endpoints'].get('end', {}).get('endpoint_id') != end):
                raise ValueError('physical review candidate correspondence mismatch')
            value = self.snapshot()
            decision = next(d for d in value['decisions'] if d['candidate_id'] == cid)
            return {'method': 'manual_review', 'input_sha256': self.input_sha256,
                    'reviewer': value['reviewer'], 'decision': decision}
        return None


def load_physical_review(request, observations, movement, decoded):
    data = Path(request['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum(request['sha256']):
        raise ValueError('physical review checksum mismatch')
    value = json.loads(data, parse_constant=reject_constant)
    keys = {'format', 'schema_version', 'version', *PARENTS, 'video_sha256',
            'reviewer', 'review_context', 'evidence', 'decisions'}
    table(value, keys, keys)
    if (value['format'] != 'ez2rc.physical-review' or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['version'] != VERSION):
        raise ValueError('unsupported physical review')
    for key in (*PARENTS, 'video_sha256'):
        checksum(value[key])
    table(value['reviewer'], {'id', 'kind'}, {'id', 'kind'})
    string(value['reviewer']['id'], 'reviewer.id')
    if value['reviewer']['kind'] not in ('human', 'assistant'):
        raise ValueError('reviewer.kind must be human or assistant')
    string(value['review_context'], 'review_context')
    if not isinstance(value['evidence'], list) or not value['evidence']:
        raise ValueError('physical review requires evidence records')
    evidence_ids = set()
    for evidence in value['evidence']:
        table(evidence, {'id', 'sha256'}, {'id', 'sha256'})
        string(evidence['id'], 'evidence.id'); checksum(evidence['sha256'])
        if evidence['id'] in evidence_ids:
            raise ValueError('duplicate physical review evidence')
        evidence_ids.add(evidence['id'])
    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]):
        raise ValueError('physical review visual parent lineage mismatch')
    result = PhysicalReview(canonical(value), request['sha256'], tuple(value[k] for k in PARENTS),
                            value['video_sha256'], ())
    result.check_parents(observations, movement, decoded)
    if not isinstance(value['decisions'], list) or not value['decisions']:
        raise ValueError('physical review requires decisions')
    requested = {}
    for decision in value['decisions']:
        keys = {'candidate_id', 'event_id', 'lane', 'type', 'start', 'end', 'evidence_ids'}
        table(decision, keys, keys)
        cid = string(decision['candidate_id'], 'candidate_id')
        string(decision['event_id'], 'event_id')
        if cid in requested:
            raise ValueError('duplicate physical review candidate')
        if type(decision['lane']) is not int or decision['lane'] < 0 or decision['type'] not in ('tap', 'longnote'):
            raise ValueError('invalid reviewed lane or kind')
        if not set(identifiers(decision['evidence_ids'])) <= evidence_ids:
            raise ValueError('unknown physical review evidence')
        for side in ('start', 'end'):
            point = decision[side]
            if side == 'end' and decision['type'] == 'tap':
                if point is not None:
                    raise ValueError('reviewed tap must not have an end')
                continue
            table(point, {'endpoint_id', 'witness_observation_ids'}, {'endpoint_id', 'witness_observation_ids'})
            string(point['endpoint_id'], 'endpoint_id')
            identifiers(point['witness_observation_ids'])
        requested[cid] = decision
    selected = []
    for lane in lanes(decoded):
        endpoints = {e['id']: e for e in lane['endpoints']}
        candidates = {c['id']: c for c in lane['candidates']}
        for event in lane['events']:
            for cid in set(event['selected_candidate_ids']) & requested.keys():
                decision, candidate = requested[cid], candidates[cid]
                if (len(event['selected_candidate_ids']) != 1
                        or (decision['event_id'], decision['lane'], decision['type']) !=
                           (event['id'], lane['lane'], candidate['type'])):
                    raise ValueError('physical review requires one matching selected candidate')
                for side in ('start', 'end'):
                    point = decision[side]
                    if point is None:
                        continue
                    endpoint = endpoints.get(candidate[side])
                    if (endpoint is None or point['endpoint_id'] != endpoint['id']
                            or not set(point['witness_observation_ids']) <= set(endpoint['observation_ids'])):
                        raise ValueError('physical review endpoint or witness mismatch')
                selected.append((cid, event['id'], lane['lane'], candidate['type'], candidate['start'], candidate['end']))
    if {s[0] for s in selected} != requested.keys():
        raise ValueError('physical review candidate is unknown or unselected')
    return PhysicalReview(result.document, result.input_sha256, result.parents,
                          result.video_sha256, tuple(sorted(selected)))
