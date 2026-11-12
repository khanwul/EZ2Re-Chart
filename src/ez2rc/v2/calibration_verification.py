"""Caller-supplied measurement verification with an explicit operating scope.

The product checks immutable evidence references, code and observation settings,
and uniform lane/kind measurement definitions. It does not independently prove
the assessor's external measurements. A successful geometric fit alone cannot
create this input or turn a conditional calibration into a verified one.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .config import number, string, table
from .contracts import exact_ms, ratio
from .physical_review import LOCALIZATION_VERSION, checksum
from .storage import canonical, digest, file_hash, reject_constant

VERSION = 'scoped-calibration-verification-2'


def reviewed_localization_policy(relative):
    """Automatic visibility exclusions and explicit pixel reviews differ in scope."""
    return (LOCALIZATION_VERSION if relative.get('observed_geometry', {}).get('reviewed_localizations')
            else 'automatic_only')


def reviewed_localization_scope(relative):
    points = {p['endpoint_id']:(row['lane'],p['kind']) for row in relative['candidates']
              for p in row['endpoints'].values() if 'kind' in p}
    used = {points[r['endpoint_id']] for r in relative.get('observed_geometry', {}).get('reviewed_localizations', [])}
    return [{'lane':lane,'kind':kind} for lane,kind in sorted(used)]


def code_fingerprint():
    return digest({p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def observation_scope(observations):
    # Includes template hashes/shapes, channel, geometry and detector settings.
    # All values come from the observation cache during reconstruction.
    return {'observation_parameters_sha256':digest(observations.manifest['key']['parameters']),
            'alignment_offset_px':observations.manifest['metadata']['alignment_offset_px']}


def measurement_definitions(relative):
    """No candidate IDs, coordinates or note labels may enter a reusable scope."""
    definitions = {}
    for row in relative['candidates']:
        for point in row['endpoints'].values():
            calibration = point.get('calibration', {})
            if not calibration or calibration.get('source') == 'profile':
                raise ValueError('verification requires explicit scoped measurement definitions')
            definition = {k:v for k,v in calibration.items() if k not in ('endpoint_id','candidate_id')}
            key = row['lane'],point['kind']
            if key in definitions and definitions[key] != definition:
                raise ValueError('verification cannot generalize nonuniform lane/kind calibration')
            definitions[key] = definition
    return [{'lane':lane,'kind':kind,'definition':definition}
            for (lane,kind),definition in sorted(definitions.items())]


def measurement_scope(observations, relative):
    request = relative['reference'].get('observed_geometry')
    if request is None:
        raise ValueError('calibration verification requires observed geometry')
    return {**observation_scope(observations), 'coordinate':'first_barline_zero',
            'geometry_version':request['version'], 'scale':request['scale'],
            'endpoint_position_relation':request['endpoint_position_relation'],
            'localization_error_px':ratio(exact_ms(request['localization_error_px'])),
            'reviewed_localization_policy':reviewed_localization_policy(relative),
            'reviewed_localization_scope':reviewed_localization_scope(relative),
            'endpoint_definitions':measurement_definitions(relative)}


@dataclass(frozen=True)
class CalibrationVerification:
    document: bytes
    input_sha256: str
    observed_scope: bytes

    def snapshot(self):
        return json.loads(self.document)

    def receipt(self):
        value = self.snapshot()
        return {'version':VERSION,'input_sha256':self.input_sha256,
                'code_sha256':value['code_sha256'],'scope_sha256':digest(value['scope']),
                'assessor':value['assessor'],'method':value['method'],
                'criteria_sha256':value['criteria_sha256'],'evidence':value['evidence'],
                'limitations':value['limitations']}

    def check_relative(self, relative):
        value = self.snapshot();scope=value['scope']
        request = relative['reference'].get('observed_geometry')
        if request is None:
            return ['verified_measurement_geometry_unavailable']
        actual = {**json.loads(self.observed_scope),'coordinate':'first_barline_zero',
                  'geometry_version':request['version'],'scale':request['scale'],
                  'endpoint_position_relation':request['endpoint_position_relation'],
                  'localization_error_px':ratio(exact_ms(request['localization_error_px'])),
                  'reviewed_localization_policy':reviewed_localization_policy(relative),
                  'reviewed_localization_scope':reviewed_localization_scope(relative)}
        if actual != {k:v for k,v in scope.items() if k!='endpoint_definitions'}:
            return ['calibration_verification_geometry_scope_mismatch']
        declared={(r['lane'],r['kind']):r['definition'] for r in scope['endpoint_definitions']}
        try:
            used=measurement_definitions(relative)
        except ValueError:
            return ['calibration_verification_definition_scope_mismatch']
        if any(declared.get((r['lane'],r['kind'])) != r['definition'] for r in used):
            return ['calibration_verification_definition_scope_mismatch']
        return []


def load_calibration_verification(request, observations):
    data=Path(request['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=checksum(request['sha256']):
        raise ValueError('calibration verification checksum mismatch')
    value=json.loads(data,parse_constant=reject_constant)
    keys={'format','schema_version','version','status','code_sha256','scope','assessor',
          'method','criteria_sha256','evidence','limitations'}
    table(value,keys,keys)
    if (value['format']!='ez2rc.calibration-verification' or type(value['schema_version']) is not int
            or value['schema_version']!=1 or value['version']!=VERSION or value['status']!='verified'):
        raise ValueError('unsupported or unpassed calibration verification')
    if checksum(value['code_sha256'])!=code_fingerprint():
        raise ValueError('calibration verification code changed')
    checksum(value['criteria_sha256']);string(value['assessor'],'calibration assessor')
    string(value['limitations'],'calibration verification limitations')
    if value['method'] not in ('procedural_measurement','empirical_validation'):
        raise ValueError('unsupported calibration verification method')
    if not isinstance(value['evidence'],list) or not value['evidence']:
        raise ValueError('calibration verification requires external evidence references')
    seen=set()
    for item in value['evidence']:
        table(item,{'id','sha256'},{'id','sha256'})
        string(item['id'],'calibration evidence ID');checksum(item['sha256'])
        if item['id'] in seen:raise ValueError('duplicate calibration evidence ID')
        seen.add(item['id'])
    scope=value['scope']
    keys={'observation_parameters_sha256','alignment_offset_px','coordinate','geometry_version','scale',
          'endpoint_position_relation','localization_error_px','reviewed_localization_policy',
          'reviewed_localization_scope','endpoint_definitions'}
    table(scope,keys,keys)
    observed=observation_scope(observations)
    if any(scope[k]!=v for k,v in observed.items()):
        raise ValueError('calibration verification observation scope mismatch')
    if (scope['coordinate']!='first_barline_zero' or scope['scale']!='constant'
            or scope['endpoint_position_relation']!='same_raw_localization'):
        raise ValueError('unsupported verified measurement geometry')
    from .observed_geometry import VERSION as GEOMETRY_VERSION
    if scope['geometry_version']!=GEOMETRY_VERSION:
        raise ValueError('verified geometry version changed')
    if scope['reviewed_localization_policy'] not in ('automatic_only',LOCALIZATION_VERSION):
        raise ValueError('unsupported verified localization review policy')
    from .contracts import read_ratio
    if read_ratio(scope['localization_error_px'])<0:
        raise ValueError('invalid verified localization error')
    if not isinstance(scope['endpoint_definitions'],list) or not scope['endpoint_definitions']:
        raise ValueError('verification requires covered endpoint definitions')
    seen=set();lane_count=len(observations.manifest['metadata']['profile']['lanes'])
    reviewed_scope=scope['reviewed_localization_scope']
    if (not isinstance(reviewed_scope,list) or
            bool(reviewed_scope)!=(scope['reviewed_localization_policy']==LOCALIZATION_VERSION)):
        raise ValueError('verified localization review scope does not match its policy')
    for item in reviewed_scope:
        table(item,{'lane','kind'},{'lane','kind'})
        if type(item['lane']) is not int or not 0<=item['lane']<lane_count or item['kind'] not in ('tap','head','tail'):
            raise ValueError('invalid verified localization review scope')
        identity=item['lane'],item['kind']
        if identity in seen:raise ValueError('duplicate verified localization review scope')
        seen.add(identity)
    seen=set()
    for item in scope['endpoint_definitions']:
        table(item,{'lane','kind','definition'},{'lane','kind','definition'})
        if type(item['lane']) is not int or not 0<=item['lane']<lane_count or item['kind'] not in ('tap','head','tail'):
            raise ValueError('invalid verified endpoint scope')
        identity=item['lane'],item['kind']
        if identity in seen:raise ValueError('duplicate verified endpoint scope')
        seen.add(identity)
        # Use the same scalar and optical/registration validators as a spatial
        # reference. Candidate IDs never enter the reusable measurement scope.
        definition=item['definition']
        allowed={'version','offset_px','position_error_px','beat_shift','optical_reference','template_registration'}
        table(definition,allowed,{'version','offset_px','position_error_px'})
        string(definition['version'],'verified endpoint convention')
        from .relative_beats import bounds
        bounds(definition['offset_px'])
        number(definition['position_error_px'],'verified endpoint localization bound',0)
        if 'beat_shift' in definition:read_ratio(definition['beat_shift'])
        from .projection_edges import validate_spec,validate_registration
        if 'optical_reference' in definition:validate_spec(definition['optical_reference'])
        if 'template_registration' in definition:validate_registration(definition['template_registration'])
    return CalibrationVerification(canonical(value),request['sha256'],canonical(observed))
