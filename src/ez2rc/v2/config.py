"""Strict v2 song, recording-profile and common algorithm configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
import math
import tomllib

import cv2
import numpy as np

from .contracts import decimal_bpm, origin_payload, ratio, read_ratio
from .storage import file_hash
from .rhythm import RhythmGrid


class ConfigError(ValueError):
    pass


def table(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ConfigError("expected a TOML table")
    if extra := set(value) - set(allowed):
        raise ConfigError(f"unknown settings: {sorted(extra)}")
    if missing := set(required) - set(value):
        raise ConfigError(f"missing settings: {sorted(missing)}")
    return value


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ConfigError(f"unsupported v2 schema_version in {path}; expected integer 1")
    return data


def string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def number(value, name, minimum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or minimum is not None and value < minimum):
        raise ConfigError(f"invalid {name}")
    return float(value)


def rectangle(value, width, height):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) is not int for v in value)
            or not 0 <= value[0] < value[2] <= width
            or not 0 <= value[1] < value[3] <= height):
        raise ConfigError("ROI must be [left, top, right, bottom] inside the frame")
    return tuple(value)


@dataclass(frozen=True)
class JudgmentCalibration:
    status: str
    version: str
    offset_ms: float | None = None
    error_ms: float | None = None

    def __post_init__(self):
        string(self.version, "judgment_calibration.version")
        if self.status == "verified":
            number(self.offset_ms, "judgment_calibration.offset_ms")
            number(self.error_ms, "judgment_calibration.error_ms", 0)
        elif self.status != "unverified" or self.offset_ms is not None or self.error_ms is not None:
            raise ConfigError("unverified judgment calibration cannot supply assumed corrections")


@dataclass(frozen=True)
class MarkerCalibration:
    status: str
    version: str
    meaning: str
    offset_ms: float | None = None
    error_ms: float | None = None

    def __post_init__(self):
        string(self.version, 'marker_calibration.version')
        if self.meaning not in ('integer_beat', 'measure_boundary'):
            raise ConfigError('unknown marker musical meaning')
        if self.status == 'verified':
            number(self.offset_ms, 'marker_calibration.offset_ms')
            number(self.error_ms, 'marker_calibration.error_ms', 0)
        elif self.status != 'unverified' or self.offset_ms is not None or self.error_ms is not None:
            raise ConfigError('unverified marker calibration cannot supply assumed corrections')


@dataclass(frozen=True)
class BeatAnchor:
    id: str
    beat: Fraction
    timestamp_ms: float
    error_ms: float
    status: str
    version: str

    def __post_init__(self):
        string(self.id, 'beat_anchor.id')
        string(self.version, 'beat_anchor.version')
        if not isinstance(self.beat, Fraction):
            raise ConfigError('beat anchor requires a reduced fraction')
        number(self.timestamp_ms, 'beat_anchor.timestamp_ms')
        number(self.error_ms, 'beat_anchor.error_ms', 0)
        if self.status not in ('verified', 'unverified'):
            raise ConfigError('invalid beat anchor status')

    def snapshot(self):
        return {**asdict(self), 'beat': ratio(self.beat)}


def validate_reference_inputs(calibrations, anchors):
    meanings = {'beat_signal': 'integer_beat', 'barlines': 'measure_boundary'}
    if not isinstance(calibrations, dict) or set(calibrations) - meanings.keys():
        raise ConfigError('unknown marker calibration channel')
    for channel, calibration in calibrations.items():
        if not isinstance(calibration, MarkerCalibration) or calibration.meaning != meanings[channel]:
            raise ConfigError('marker meaning does not match channel')
    if not isinstance(anchors, (tuple, list)) or any(not isinstance(a, BeatAnchor) for a in anchors):
        raise ConfigError('beat anchors must be a sequence of BeatAnchor values')
    if len({a.id for a in anchors}) != len(anchors):
        raise ConfigError('duplicate beat anchor ID')


@dataclass(frozen=True)
class OriginAdoption:
    policy: str
    version: str

    def __post_init__(self):
        if self.policy not in ('exact_anchor_intersection', 'anchor_interval'):
            raise ConfigError('unsupported origin adoption policy')
        string(self.version, 'origin_adoption.version')


@dataclass(frozen=True)
class RangeUncertainty:
    version: str
    judgment_scope: str
    origin_relation: str
    origin_reference_ms: Fraction | None = None
    endpoint_component: str | None = None

    def __post_init__(self):
        string(self.version, 'range_uncertainty.version')
        if self.judgment_scope != 'shared_all_events':
            raise ConfigError('judgment uncertainty requires an explicit shared_all_events scope')
        if self.origin_relation == 'same_clock_shift':
            if not isinstance(self.origin_reference_ms, Fraction):
                raise ConfigError('same_clock_shift requires an exact origin reference')
        elif self.origin_relation != 'unknown' or self.origin_reference_ms is not None:
            raise ConfigError('unknown origin relation must not supply an origin reference')
        if self.endpoint_component not in (None, 'calibration_and_localization'):
            raise ConfigError('endpoint ranges require the complete calibration_and_localization component')

    def snapshot(self):
        return {'version': self.version, 'judgment_scope': self.judgment_scope,
                'origin_relation': self.origin_relation,
                'origin_reference_ms_exact': ratio(self.origin_reference_ms) if self.origin_reference_ms is not None else None,
                **({'endpoint_component': self.endpoint_component} if self.endpoint_component is not None else {})}


@dataclass(frozen=True)
class SongConfig:
    source: Path
    video: Path
    profile: Path
    title: str
    difficulty: str
    bpm: str
    start_ms: float = 0.0
    end_ms: float | None = None
    algorithm: Path | None = None
    beat_zero_ms: float | Fraction | None = None
    measure_beats: Fraction | None = None
    judgment_calibration: JudgmentCalibration | None = None
    marker_calibrations: dict[str, MarkerCalibration] = field(default_factory=dict)
    beat_anchors: tuple[BeatAnchor, ...] = ()
    origin_adoption: OriginAdoption | None = None
    range_uncertainty: RangeUncertainty | None = None
    relative_beats: dict | None = None
    context_before_ms: float = 0.0
    context_after_ms: float = 0.0
    rhythm_grid: RhythmGrid = field(default_factory=RhythmGrid)
    spatial_calibration: dict | None = None
    physical_review: dict | None = None
    calibration_verification: dict | None = None

    def __post_init__(self):
        number(self.context_before_ms, 'context_before_ms', 0)
        number(self.context_after_ms, 'context_after_ms', 0)
        if not isinstance(self.rhythm_grid, RhythmGrid):
            raise ConfigError('rhythm_grid must be a RhythmGrid')

    def snapshot(self):
        values = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        values.update(origin_payload(self.beat_zero_ms))
        values['measure_beats'] = ratio(self.measure_beats) if self.measure_beats is not None else None
        values['beat_anchors'] = [a.snapshot() for a in self.beat_anchors]
        values['range_uncertainty'] = self.range_uncertainty.snapshot() if self.range_uncertainty is not None else None
        values['rhythm_grid'] = self.rhythm_grid.snapshot()
        return values


@dataclass(frozen=True)
class MotionConfig:
    min_similarity: float = 0.70
    match_tolerance_px: float = 2.0
    max_speed_px_per_ms: float = 6.0
    displacement_slack_px: float = 4.0
    min_lanes: int = 2
    history_frames: int = 8
    ambiguity_ratio: float = 0.90
    repeat_difference: float = 0.75

    def __post_init__(self):
        for key in ("min_similarity", "ambiguity_ratio"):
            if not 0 < number(getattr(self, key), key) <= 1:
                raise ConfigError(f"{key} must be in (0, 1]")
        for key in ("match_tolerance_px", "max_speed_px_per_ms", "displacement_slack_px"):
            number(getattr(self, key), key, 0.001)
        number(self.repeat_difference, "repeat_difference", 0)
        integer(self.min_lanes, "min_lanes", 2)
        integer(self.history_frames, "history_frames", 1)


@dataclass(frozen=True)
class FusionConfig:
    grid_px: float = 1.0
    tile_bins: int = 2048
    unknown_weight: float = 0.5
    occluded_weight: float = 0.1
    clipped_weight: float = 0.25
    effect_weight: float = 0.0
    repeat_group_cap: float = 1.0
    inferred_motion_weight: float = 0.25

    def __post_init__(self):
        number(self.grid_px, "grid_px", 0.001)
        integer(self.tile_bins, "tile_bins", 16)
        for key in ("unknown_weight", "occluded_weight", "clipped_weight", "effect_weight", "inferred_motion_weight"):
            if not 0 <= number(getattr(self, key), key) <= 1:
                raise ConfigError(f"{key} must be in [0, 1]")
        number(self.repeat_group_cap, "repeat_group_cap", 0.001)


@dataclass(frozen=True)
class DecodeConfig:
    endpoint_radius_px: float = 2.0
    association_similarity: float = 0.70
    support_weight: float = 1.0
    note_cost: float = 0.8
    endpoint_reward: float = 2.0
    additional_endpoint_cost: float = 1.0
    body_reward: float = 1.0
    body_only_reward: float = 3.0
    body_support_px: float = 24.0
    body_bright: float = 0.20
    body_dark: float = 0.08
    absence_cost_per_px: float = 0.12
    unavailable_cost_per_px: float = 0.002
    missing_endpoint_cost: float = 0.8
    ambiguity_cost: float = 0.35

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            number(getattr(self, key), key, 0)
        for key in ("endpoint_radius_px", "support_weight", "body_support_px", "absence_cost_per_px",
                    "unavailable_cost_per_px"):
            number(getattr(self, key), key, 0.001)
        if not 0 < self.association_similarity <= 1:
            raise ConfigError("decode association_similarity must be in (0, 1]")
        if not 0 <= self.body_dark < self.body_bright <= 1:
            raise ConfigError("decode body thresholds must satisfy 0 <= dark < bright <= 1")


@dataclass(frozen=True)
class AlgorithmConfig:
    chunk_frames: int = 120
    timestamp_policy: str = "strict"
    brightness_threshold: float = 40.0
    edge_threshold: float = 12.0
    search_radius_px: int = 8
    alignment_probe_ms: float = 2000.0
    barline_min_brightness: float = 30.0
    barline_max_brightness: float = 180.0
    barline_max_thickness: int = 8
    motion: MotionConfig = field(default_factory=MotionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    def __post_init__(self):
        if (not isinstance(self.motion, MotionConfig) or not isinstance(self.fusion, FusionConfig)
                or not isinstance(self.decode, DecodeConfig)):
            raise ConfigError("motion, fusion and decode must be parameter tables")
        integer(self.chunk_frames, "chunk_frames", 1)
        integer(self.search_radius_px, "search_radius_px")
        integer(self.barline_max_thickness, "barline_max_thickness", 1)
        if not isinstance(self.timestamp_policy, str) or self.timestamp_policy not in {"strict", "fps_fallback"}:
            raise ConfigError("timestamp_policy must be strict or fps_fallback")
        for key in ("brightness_threshold", "edge_threshold", "barline_min_brightness",
                    "barline_max_brightness"):
            if not 0 <= number(getattr(self, key), key) <= 255:
                raise ConfigError(f"{key} must be in [0, 255]")
        if self.barline_max_brightness < self.barline_min_brightness:
            raise ConfigError("invalid barline brightness interval")
        number(self.alignment_probe_ms, "alignment_probe_ms", 0.001)

    def observation_parameters(self):
        return {k: v for k, v in asdict(self).items() if k not in {"timestamp_policy", "motion", "fusion", "decode"}}


@dataclass(frozen=True)
class Calibration:
    detection_reference: str
    status: str
    version: str
    offset_px: float | None = None
    error_px: float | None = None


@dataclass(frozen=True)
class PositionCalibration:
    """Caller-declared vertical localization bound for this recording profile."""
    status: str
    version: str
    scope: str
    error_px: float | None = None

    def __post_init__(self):
        string(self.version, 'position_calibration.version')
        if self.scope != 'profile_endpoints':
            raise ConfigError('position calibration scope must be profile_endpoints')
        if self.status == 'verified':
            number(self.error_px, 'position_calibration.error_px', 0)
        elif self.status != 'unverified' or self.error_px is not None:
            raise ConfigError('unverified position calibration cannot supply an error bound')

    def snapshot(self):
        return {'status': self.status, 'version': self.version, 'scope': self.scope,
                **({'error_px': self.error_px} if self.error_px is not None else {})}


def read_position_calibration(value):
    value = table(value, {'status', 'version', 'scope', 'error_px'}, {'status', 'version', 'scope'})
    if value['status'] == 'unverified' and 'error_px' in value:
        raise ConfigError('unverified position calibration must omit error_px')
    return PositionCalibration(**value)


@dataclass
class Lane:
    roi: tuple[int, int, int, int]
    match_roi: tuple[int, int, int, int]
    channel: str
    templates: dict[str, np.ndarray]


@dataclass
class Profile:
    id: str
    game: str
    mode: str
    resolution: tuple[int, int]
    judge_y: float
    effect_top: float
    beat_roi: tuple[int, int, int, int]
    alignment_band: tuple[int, int, int, int]
    alignment_max_shift: int
    calibration: dict[str, Calibration]
    lanes: list[Lane]
    snapshot: dict
    asset_hashes: dict[str, str]
    position_calibration: PositionCalibration | None = None

    @property
    def lane_count(self):
        return len(self.lanes)


def load_song(path: str | Path) -> SongConfig:
    source = Path(path).resolve()
    data = table(load_toml(source), {"schema_version", "input", "metadata", "timing", "algorithm"},
                 {"input", "metadata", "timing"})
    inp = table(data["input"], {"video", "profile", "start_ms", "end_ms",
                                "context_before_ms", "context_after_ms"}, {"video", "profile"})
    meta = table(data["metadata"], {"title", "difficulty"}, {"title", "difficulty"})
    timing = table(data["timing"], {"mode", "bpm", "beat_zero_ms", "beat_zero_ms_exact", "measure_beats", "judgment_calibration",
                                     "marker_calibrations", "beat_anchors", "origin_adoption", "range_uncertainty", "relative_beats", "rhythm_grid", "spatial_calibration", "physical_review", "calibration_verification"}, {"mode", "bpm"})
    if timing["mode"] != "fixed":
        raise ConfigError("only explicitly supplied fixed BPM is implemented")
    decimal_bpm(timing["bpm"])
    origin = number(timing['beat_zero_ms'], 'timing.beat_zero_ms') if 'beat_zero_ms' in timing else None
    if 'beat_zero_ms_exact' in timing:
        if 'beat_zero_ms' in timing:
            raise ConfigError('supply only one beat origin representation')
        try:
            origin = read_ratio(timing['beat_zero_ms_exact'])
        except ValueError as exc:
            raise ConfigError(f'invalid timing.beat_zero_ms_exact: {exc}') from exc
    measure = None
    if 'measure_beats' in timing:
        try:
            measure = read_ratio(timing['measure_beats'])
            if measure <= 0:
                raise ValueError('measure length must be positive')
        except ValueError as exc:
            raise ConfigError(f'invalid timing.measure_beats: {exc}') from exc
    judgment = None
    if 'judgment_calibration' in timing:
        value = table(timing['judgment_calibration'], {'status', 'version', 'offset_ms', 'error_ms'}, {'status', 'version'})
        judgment = JudgmentCalibration(**value)
    calibrations = {}
    for channel, value in table(timing.get('marker_calibrations', {}), {'beat_signal', 'barlines'}).items():
        calibrations[channel] = MarkerCalibration(**table(value, MarkerCalibration.__dataclass_fields__,
                                                          {'status', 'version', 'meaning'}))
    anchors = []
    if not isinstance(timing.get('beat_anchors', []), list):
        raise ConfigError('timing.beat_anchors must be an array of tables')
    for value in timing.get('beat_anchors', []):
        value = dict(table(value, BeatAnchor.__dataclass_fields__, BeatAnchor.__dataclass_fields__))
        try:
            value['beat'] = read_ratio(value['beat'])
        except ValueError as exc:
            raise ConfigError(f'invalid beat anchor: {exc}') from exc
        anchors.append(BeatAnchor(**value))
    validate_reference_inputs(calibrations, anchors)
    adoption = None
    if 'origin_adoption' in timing:
        adoption = OriginAdoption(**table(timing['origin_adoption'], {'policy', 'version'}, {'policy', 'version'}))
    uncertainty = None
    if 'range_uncertainty' in timing:
        value = dict(table(timing['range_uncertainty'],
                           {'version', 'judgment_scope', 'origin_relation', 'origin_reference_ms_exact', 'endpoint_component'},
                           {'version', 'judgment_scope', 'origin_relation'}))
        if 'origin_reference_ms_exact' in value:
            try:
                value['origin_reference_ms'] = read_ratio(value.pop('origin_reference_ms_exact'))
            except ValueError as exc:
                raise ConfigError(f'invalid uncertainty origin reference: {exc}') from exc
        uncertainty = RangeUncertainty(**value)
    video = (source.parent / string(inp["video"], "input.video")).resolve()
    profile_name = string(inp["profile"], "input.profile")
    # IDs are resolved in a profiles/ directory beside the song config.
    profile_path = (Path(profile_name) if Path(profile_name).suffix
                    else Path("profiles") / f"{profile_name}.toml")
    profile = (source.parent / profile_path).resolve()
    title = string(meta["title"], "metadata.title")
    difficulty = string(meta["difficulty"], "metadata.difficulty")
    if difficulty not in {"EZ", "NM", "HD", "SHD"}:
        raise ConfigError("difficulty must be EZ, NM, HD or SHD")
    start = number(inp.get("start_ms", 0.0), "start_ms", 0)
    end = number(inp["end_ms"], "end_ms", 0) if "end_ms" in inp else None
    if end is not None and end <= start:
        raise ConfigError("end_ms must follow start_ms")
    algorithm = None
    if "algorithm" in data:
        settings = table(data["algorithm"], {"config"}, {"config"})
        algorithm = (source.parent / string(settings["config"], "algorithm.config")).resolve()
    relative, spatial_calibration, physical_review, calibration_verification = None, None, None, None
    try:
        grid = RhythmGrid(**table(timing.get('rhythm_grid', {}), {'subdivisions'}))
    except ValueError as exc:
        raise ConfigError(f'invalid rhythm_grid: {exc}') from exc
    if 'relative_beats' in timing:
        value = table(timing['relative_beats'], {'path', 'sha256'}, {'path', 'sha256'})
        checksum = string(value['sha256'], 'relative_beats.sha256')
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ConfigError('relative_beats.sha256 must be a lowercase SHA-256')
        relative = {'path': str((source.parent / string(value['path'], 'relative_beats.path')).resolve()),
                    'sha256': checksum}
    if 'spatial_calibration' in timing:
        if relative is not None:
            raise ConfigError('supply manual relative_beats or automatic spatial_calibration')
        value = table(timing['spatial_calibration'], {'path', 'sha256'}, {'path', 'sha256'})
        checksum = string(value['sha256'], 'spatial_calibration.sha256')
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ConfigError('spatial_calibration.sha256 must be a lowercase SHA-256')
        spatial_calibration = {'path': str((source.parent / string(value['path'], 'spatial_calibration.path')).resolve()),
                               'sha256': checksum}
    if 'physical_review' in timing:
        value = table(timing['physical_review'], {'path', 'sha256'}, {'path', 'sha256'})
        checksum = string(value['sha256'], 'physical_review.sha256')
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ConfigError('physical_review.sha256 must be a lowercase SHA-256')
        physical_review = {'path': str((source.parent / string(value['path'], 'physical_review.path')).resolve()),
                           'sha256': checksum}
    if 'calibration_verification' in timing:
        value = table(timing['calibration_verification'], {'path', 'sha256'}, {'path', 'sha256'})
        checksum = string(value['sha256'], 'calibration_verification.sha256')
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ConfigError('calibration_verification.sha256 must be a lowercase SHA-256')
        calibration_verification = {
            'path': str((source.parent / string(value['path'], 'calibration_verification.path')).resolve()),
            'sha256': checksum}
    return SongConfig(source, video, profile, title, difficulty, timing["bpm"], start, end, algorithm,
                      origin, measure, judgment, calibrations, tuple(anchors), adoption, uncertainty, relative,
                      number(inp.get('context_before_ms', 0), 'context_before_ms', 0),
                      number(inp.get('context_after_ms', 0), 'context_after_ms', 0), grid, spatial_calibration, physical_review,
                      calibration_verification)


def load_algorithm(path: Path | None) -> AlgorithmConfig:
    if path is None:
        return AlgorithmConfig()
    data = load_toml(path)
    table(data, {"schema_version", *AlgorithmConfig.__dataclass_fields__})
    for name, cls in (("motion", MotionConfig), ("fusion", FusionConfig), ("decode", DecodeConfig)):
        if name in data:
            table(data[name], cls.__dataclass_fields__)
            data[name] = cls(**data[name])
    return AlgorithmConfig(**{k: v for k, v in data.items() if k != "schema_version"})


def load_profile(path: Path) -> Profile:
    data = table(load_toml(path), {"schema_version", "id", "game", "mode", "resolution",
                 "judge_y", "effect_top", "beat_roi", "alignment", "calibration", "position_calibration", "lanes"},
                 {"id", "game", "mode", "resolution", "judge_y", "effect_top", "beat_roi",
                  "alignment", "calibration", "lanes"})
    for key in ("id", "game", "mode"):
        string(data[key], key)
    resolution = data["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ConfigError("resolution requires width and height")
    width, height = [integer(v, "resolution", 1) for v in resolution]
    judge = number(data["judge_y"], "judge_y", 0)
    effect = number(data["effect_top"], "effect_top", 0)
    if not effect <= judge < height:
        raise ConfigError("invalid judgment/effect geometry")
    beat = rectangle(data["beat_roi"], width, height)
    alignment = table(data["alignment"], {"band", "max_shift_px"}, {"band", "max_shift_px"})
    band = rectangle(alignment["band"], width, height)
    shift = integer(alignment["max_shift_px"], "max_shift_px")
    calibrations = {}
    table(data["calibration"], {"tap", "head", "tail"}, {"tap", "head", "tail"})
    for kind, value in data["calibration"].items():
        table(value, {"detection_reference", "status", "version", "offset_px", "error_px"},
              {"detection_reference", "status", "version"})
        if value["detection_reference"] != "template_top":
            raise ConfigError("only template_top detection references are implemented")
        string(value["version"], "calibration.version")
        if value["status"] == "verified":
            if "offset_px" not in value or "error_px" not in value:
                raise ConfigError("verified calibration needs offset_px and error_px")
            number(value["offset_px"], "offset_px")
            number(value["error_px"], "error_px", 0)
        elif value["status"] != "unverified" or "offset_px" in value or "error_px" in value:
            raise ConfigError("unverified calibration cannot supply assumed corrections")
        calibrations[kind] = Calibration(**value)
    position = read_position_calibration(data['position_calibration']) if 'position_calibration' in data else None
    if not isinstance(data["lanes"], list) or not data["lanes"]:
        raise ConfigError("profile requires lanes")
    lanes, hashes, loaded = [], {}, {}
    previous_right = -1
    for i, lane in enumerate(data["lanes"]):
        table(lane, {"roi", "match_roi", "channel", "templates"}, {"roi", "match_roi", "channel", "templates"})
        roi = rectangle(lane["roi"], width, height)
        match = rectangle(lane["match_roi"], width, height)
        if roi[0] < previous_right or match[1::2] != roi[1::2] or not match[0] <= roi[0] < roi[2] <= match[2]:
            raise ConfigError("lanes must be ordered and nonoverlapping with matching ROI enclosing each lane")
        previous_right = roi[2]
        if not isinstance(lane["channel"], str) or lane["channel"] not in {"gray", "blue", "green", "red", "max_bg"}:
            raise ConfigError("unsupported lane channel")
        if lanes and roi[1::2] != lanes[0].roi[1::2]:
            raise ConfigError("lanes must share vertical coordinates")
        table(lane["templates"], {"tap", "head", "tail"}, {"tap", "head", "tail"})
        templates = {}
        for kind, filename in lane["templates"].items():
            asset = (path.parent / string(filename, "template path")).resolve()
            if not asset.is_file():
                raise ConfigError(f"missing template asset: {asset}")
            if asset not in loaded:
                image = cv2.imread(str(asset), cv2.IMREAD_COLOR)
                if image is None or image.std() == 0:
                    raise ConfigError(f"invalid template asset: {asset}")
                loaded[asset] = image
            image = loaded[asset]
            if image.shape[0] > roi[3] - roi[1] or image.shape[1] > match[2] - match[0]:
                raise ConfigError("template does not fit its matching ROI")
            templates[kind] = image
            hashes[f"lane{i}.{kind}"] = file_hash(asset)
        lanes.append(Lane(roi, match, lane["channel"], templates))
    if data["mode"] != f"{len(lanes)}k":
        raise ConfigError("profile mode does not match its lane count")
    return Profile(data["id"], data["game"], data["mode"], (width, height), judge, effect,
                   beat, band, shift, calibrations, lanes, data, hashes, position)
