"""Strict v2 song, recording-profile and common algorithm configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import tomllib

import cv2
import numpy as np

from .contracts import decimal_bpm
from .storage import file_hash


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

    def snapshot(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


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

    @property
    def lane_count(self):
        return len(self.lanes)


def load_song(path: str | Path) -> SongConfig:
    source = Path(path).resolve()
    data = table(load_toml(source), {"schema_version", "input", "metadata", "timing", "algorithm"},
                 {"input", "metadata", "timing"})
    inp = table(data["input"], {"video", "profile", "start_ms", "end_ms"}, {"video", "profile"})
    meta = table(data["metadata"], {"title", "difficulty"}, {"title", "difficulty"})
    timing = table(data["timing"], {"mode", "bpm"}, {"mode", "bpm"})
    if timing["mode"] != "fixed":
        raise ConfigError("only explicitly supplied fixed BPM is implemented")
    decimal_bpm(timing["bpm"])
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
    return SongConfig(source, video, profile, title, difficulty, timing["bpm"], start, end, algorithm)


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
                 "judge_y", "effect_top", "beat_roi", "alignment", "calibration", "lanes"},
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
                   beat, band, shift, calibrations, lanes, data, hashes)
