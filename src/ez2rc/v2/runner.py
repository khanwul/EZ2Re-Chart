"""Run lifecycle. Visual inference without interval/timing decode stays partial."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from . import __version__
from .capture import CaptureError, VERSION as CAPTURE_VERSION, validate_timeline
from .config import load_algorithm, load_profile, load_song
from .contracts import Chart, Metadata, STAGES, Timing
from .export import write_outputs
from .observation import collect, open_observations
from .motion import estimate
from .fusion import accumulate
from .storage import Cache, digest, file_hash, safe_component, write_json

EXIT_COMPLETE, EXIT_PARTIAL, EXIT_FAILED = 0, 2, 1


def code_identity():
    root = Path(__file__).parent
    sources = {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        commit = None
    return {"package_version": __version__, "git_commit": commit, "source_sha256": digest(sources)}


def attach_cache(cache: Cache, destination: Path):
    destination.mkdir(parents=True)
    shutil.copyfile(cache.path / "manifest.json", destination / "manifest.json")
    for chunk in cache.manifest["chunks"]:
        source, target = cache.path / chunk["file"], destination / chunk["file"]
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def execute(config_path: Path, *, output_root: Path = Path("out"),
            cache_root: Path | None = None, observation_cache: Path | None = None) -> tuple[int, Path]:
    song = load_song(config_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    root = output_root / safe_component(song.title) / safe_component(song.difficulty) / run_id
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run = {"format": "ez2rc.run", "schema_version": 1, "run_id": run_id, "status": "running",
           "command": "reconstruct" if observation_cache is not None else "extract",
           "code": code_identity(), "settings": song.snapshot(), "algorithm": None,
           "input": {}, "stages": {s: {"status": "pending"} for s in STAGES},
           "cost": {}, "failure": None}
    write_json(root / "run.json", run)
    active = "preflight"
    try:
        algorithm = load_algorithm(song.algorithm)
        run["algorithm"] = asdict(algorithm)
        active = "observation"
        run["stages"]["observation"]["status"] = "running"
        write_json(root / "run.json", run)
        if observation_cache is None:
            active = "preflight"
            profile = load_profile(song.profile)
            run["input"]["asset_sha256"] = profile.asset_hashes
            active = "observation"
            capture, observations, reused = collect(song, profile, algorithm, cache_root or output_root / ".cache")
        else:
            observations = open_observations(observation_cache)
            meta = observations.manifest["metadata"]
            params = observations.manifest["key"]["parameters"]
            if params["algorithm"] != algorithm.observation_parameters():
                raise ValueError("observation settings changed; run extract to regenerate observations")
            # Reconstruct needs no video or image assets. If a profile file is
            # present, do not silently accept geometry edits against old evidence.
            if song.profile.is_file() and file_hash(song.profile) != meta["profile_sha256"]:
                raise ValueError("recording profile changed; run extract again")
            if song.video.is_file() and file_hash(song.video) != meta["video_sha256"]:
                raise ValueError("video content changed; run extract again")
            for lane_id, lane in enumerate(meta["profile"]["lanes"]):
                for kind, filename in lane["templates"].items():
                    asset = (song.profile.parent / filename).resolve()
                    if asset.is_file() and file_hash(asset) != meta["asset_sha256"][f"lane{lane_id}.{kind}"]:
                        raise ValueError("template content changed; run extract again")
            capture_id = meta["capture_cache_id"]
            capture_path = observation_cache.resolve().parent.parent / "capture" / capture_id
            # Run artifacts contain an adjacent capture manifest/cache too.
            if not capture_path.exists():
                capture_path = observation_cache.parent / "capture"
            capture = Cache(capture_path)
            if (capture.manifest["key"]["stage"] != "capture"
                    or capture.manifest["key"]["algorithm_version"] != CAPTURE_VERSION):
                raise ValueError("unsupported capture cache version")
            for chunk in capture.chunks():
                validate_timeline(chunk, capture.manifest["metadata"])
            capture_parameters = capture.manifest["key"]["parameters"]
            if (capture.id != capture_id
                    or observations.manifest["key"]["parents"] != [capture.id]
                    or capture_parameters["start_ms"] != song.start_ms
                    or capture_parameters["end_ms"] != song.end_ms
                    or capture_parameters["timestamp_policy"] != algorithm.timestamp_policy):
                raise ValueError("capture lineage or analysis interval differs from the configuration")
            reused = True
        meta = observations.manifest["metadata"]
        run["input"] = {"video_sha256": meta["video_sha256"], "asset_sha256": meta["asset_sha256"],
                        "capture": meta["capture"], "profile": meta["profile"]}
        run["cost"]["capture_observation_seconds"] = time.monotonic() - started
        for stage, cache in (("capture", capture), ("observation", observations)):
            run["stages"][stage] = {"status": "complete", "cache_id": cache.id, "reused": reused,
                                     "algorithm_version": cache.manifest["key"]["algorithm_version"]}
            attach_cache(cache, root / "artifacts" / ("observations" if stage == "observation" else stage))
        motion = None
        for active in ("motion", "fusion"):
            run["stages"][active]["status"] = "running"
            write_json(root / "run.json", run)
            stage_started = time.monotonic()
            if active == "motion":
                cache, reused = estimate(observations, algorithm.motion, cache_root or output_root / ".cache")
                motion = cache
            else:
                cache, reused = accumulate(observations, motion, algorithm.fusion, cache_root or output_root / ".cache")
            run["cost"][f"{active}_seconds"] = time.monotonic() - stage_started
            run["stages"][active] = {"status": "complete", "cache_id": cache.id, "reused": reused,
                                      "algorithm_version": cache.manifest["key"]["algorithm_version"],
                                      "parents": cache.manifest["key"]["parents"],
                                      "artifact_ref": f"artifacts/{active}/manifest.json"}
            attach_cache(cache, root / "artifacts" / active)
        active = "export"
        run["stages"][active]["status"] = "running"
        write_json(root / "run.json", run)
        profile_data = meta["profile"]
        chart = Chart(
            Metadata(profile_data["game"], profile_data["mode"], len(profile_data["lanes"]),
                     song.title, song.difficulty), Timing(song.bpm),
            pending_stages=("decode", "timing"))
        write_outputs(root, chart, (), observation_ref="artifacts/observations/manifest.json",
                      motion_ref="artifacts/motion/manifest.json", fusion_ref="artifacts/fusion/manifest.json",
                      diagnostics={"motion": motion.manifest["metadata"]["status_counts"],
                                   "endpoint_calibration": "unverified" if any(c["status"] != "verified" for c in profile_data["calibration"].values()) else "verified"})
        run["status"] = chart.status
        run["stages"]["export"] = {"status": "complete", "schema_version": 1}
        result = EXIT_PARTIAL
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CaptureError):
            active = "capture"
        run["status"] = "failed"
        run["failure"] = {"stage": active, "type": type(exc).__name__, "message": str(exc)}
        if active in run["stages"]:
            run["stages"][active] = {"status": "failed"}
        for stage in run["stages"].values():
            if stage["status"] == "running":
                stage["status"] = "interrupted"
        result = EXIT_FAILED
    run["cost"]["elapsed_seconds"] = time.monotonic() - started
    write_json(root / "run.json", run)
    return result, root
