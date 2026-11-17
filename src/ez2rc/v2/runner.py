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
from .contracts import Metadata, STAGES, origin_schema, read_origin
from .export import write_outputs, chart_schema
from .observation import collect, open_observations
from .motion import estimate
from .fusion import accumulate
from .decode import infer, output_events
from .storage import Cache, digest, file_hash, safe_component, write_json
from .timing import PROJECTION_VERSION, RESOLUTION_VERSION, project_video_times, resolve_pixel_timing
from .reference import VERSION as REFERENCE_VERSION, reference_candidates
from .reference_link import VERSION as REFERENCE_LINK_VERSION, link_reference_origin
from .origin import VERSION as ORIGIN_VERSION, adopt_reference_origin
from .beat_ranges import project_beat_ranges
from .endpoint_ranges import VERSION as ENDPOINT_RANGES_VERSION, project_endpoint_time_ranges
from .relative_beats import load_reference, load_spatial_calibration, project_relative_beats, screen_end_reference
from .rhythm import rhythm_candidates
from .barline_grid import infer_barline_grid, spatial_reference
from .spatial_resolution import resolve_spatial_chart
from .physical_review import load_physical_review
from .calibration_verification import load_calibration_verification
from .window import song_window, has_context, analysis_scope, scope_events
from .draft import create_draft, save_draft
from .draft_projection import project_draft
from .review import write_review

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
    run = {"format": "ez2rc.run", "schema_version": origin_schema(song.beat_zero_ms), "run_id": run_id, "status": "running",
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
            window = song_window(song)
            declared_window = window if has_context(window) else None
            if (capture.id != capture_id
                    or observations.manifest["key"]["parents"] != [capture.id]
                    or capture_parameters["start_ms"] != song.start_ms
                    or capture_parameters["end_ms"] != song.end_ms
                    or capture_parameters["timestamp_policy"] != algorithm.timestamp_policy
                    or capture_parameters.get('analysis_window') != declared_window
                    or params.get('analysis_window') != declared_window
                    or capture.manifest['metadata'].get('analysis_window') != declared_window):
                raise ValueError("capture lineage or analysis interval differs from the configuration")
            reused = True
        meta = observations.manifest["metadata"]
        run["input"] = {"video_sha256": meta["video_sha256"], "asset_sha256": meta["asset_sha256"],
                        "capture": meta["capture"], "profile": meta["profile"],
                        "profile_sha256": meta["profile_sha256"]}
        if has_context(song_window(song)):
            run['input']['analysis_window'] = song_window(song)
        run["cost"]["capture_observation_seconds"] = time.monotonic() - started
        for stage, cache in (("capture", capture), ("observation", observations)):
            run["stages"][stage] = {"status": "complete", "cache_id": cache.id, "reused": reused,
                                     "algorithm_version": cache.manifest["key"]["algorithm_version"]}
            attach_cache(cache, root / "artifacts" / ("observations" if stage == "observation" else stage))
        motion = None
        for active in ("motion", "fusion", "decode"):
            run["stages"][active]["status"] = "running"
            write_json(root / "run.json", run)
            stage_started = time.monotonic()
            if active == "motion":
                cache, reused = estimate(observations, algorithm.motion, cache_root or output_root / ".cache")
                motion = cache
            elif active == "fusion":
                cache, reused = accumulate(observations, motion, algorithm.fusion, cache_root or output_root / ".cache")
                fusion = cache
            else:
                cache, reused = infer(observations, motion, fusion, algorithm.decode, cache_root or output_root / ".cache")
                decoded = cache
            run["cost"][f"{active}_seconds"] = time.monotonic() - stage_started
            run["stages"][active] = {"status": "complete", "cache_id": cache.id, "reused": reused,
                                      "algorithm_version": cache.manifest["key"]["algorithm_version"],
                                      "parents": cache.manifest["key"]["parents"],
                                      "artifact_ref": f"artifacts/{active}/manifest.json"}
            attach_cache(cache, root / "artifacts" / active)
        active = "timing"
        scope_ref, scope = None, None
        if has_context(song_window(song)):
            scope = analysis_scope(observations, motion, decoded, song_window(song))
            scope_ref = 'artifacts/analysis-scope.json'
            write_json(root / scope_ref, scope)
            run['analysis_scope_ref'] = scope_ref
        projection = project_video_times(observations, motion, decoded)
        if scope_ref is not None:
            projection.update(analysis_scope_ref=scope_ref, observation_scope='includes_auxiliary')
        projection_ref = "artifacts/time-projection.json"
        write_json(root / projection_ref, projection)
        endpoint_ranges_ref = 'artifacts/endpoint-time-ranges.json'
        endpoint_ranges = project_endpoint_time_ranges(observations, motion, projection)
        endpoint_ranges.update(time_projection_ref=projection_ref, time_projection_sha256=file_hash(root / projection_ref),
                               motion_ref='artifacts/motion/manifest.json',
                               motion_sha256=file_hash(root / 'artifacts/motion/manifest.json'))
        write_json(root / endpoint_ranges_ref, endpoint_ranges)
        reference_ref = "artifacts/reference-candidates.json"
        references = reference_candidates(
            observations, motion, song.bpm, beat_zero_ms=song.beat_zero_ms, measure_beats=song.measure_beats)
        write_json(root / reference_ref, references)
        barline_ref = 'artifacts/barline-grid.json'
        barlines = infer_barline_grid(observations, motion, song.bpm, song.measure_beats)
        write_json(root / barline_ref, barlines)
        reference_link_ref = 'artifacts/reference-origin-link.json'
        linked = link_reference_origin(references, marker_calibrations=song.marker_calibrations,
                                       beat_anchors=song.beat_anchors)
        linked['reference_candidates_ref'] = reference_ref
        linked['reference_candidates_sha256'] = file_hash(root / reference_ref)
        write_json(root / reference_link_ref, linked)
        adoption_ref = 'artifacts/origin-adoption.json'
        adoption = adopt_reference_origin(linked, song.origin_adoption)
        adoption['reference_origin_link_ref'] = reference_link_ref
        adoption['reference_origin_link_sha256'] = file_hash(root / reference_link_ref)
        write_json(root / adoption_ref, adoption)
        profile_data = meta["profile"]
        calibration_verification, calibration_verification_ref = None, None
        if song.calibration_verification is not None:
            calibration_verification = load_calibration_verification(song.calibration_verification, observations)
            calibration_verification_ref = 'artifacts/calibration-verification.json'
            write_json(root / calibration_verification_ref, calibration_verification.snapshot())
        physical_review, physical_review_ref = None, None
        if song.physical_review is not None:
            physical_review = load_physical_review(song.physical_review, observations, motion, decoded)
            physical_review_ref = 'artifacts/physical-review.json'
            write_json(root / physical_review_ref, physical_review.snapshot())
        events, pixel_intervals = output_events(decoded)
        if scope is not None:
            events, pixel_intervals = scope_events(events, pixel_intervals, scope)
        relative_ref, relative = None, None
        relative_input, spatial_input_sha = None, None
        if song.relative_beats is not None:
            relative_input = load_reference(song.relative_beats)
            spatial_input_sha = song.relative_beats['sha256']
        elif song.spatial_calibration is not None:
            spatial_calibration = load_spatial_calibration(song.spatial_calibration)
            write_json(root/'artifacts/spatial-calibration.json',spatial_calibration)
            relative_input = spatial_reference(barlines,spatial_calibration)
            spatial_input_sha = song.spatial_calibration['sha256']
        if relative_input is not None:
            relative_input, end_policy = screen_end_reference(relative_input, decoded)
            run['end_policy'] = end_policy
            relative = project_relative_beats(observations, motion, decoded, relative_input,
                                              measure_beats=song.measure_beats, rhythm_grid=song.rhythm_grid,
                                              physical_review=physical_review)
            if scope_ref is not None:
                relative.update(analysis_scope_ref=scope_ref, observation_scope='includes_auxiliary')
            relative_input_ref = 'artifacts/spatial-beat-reference.json'
            write_json(root / relative_input_ref, relative_input)
            relative.update(input_sha256=spatial_input_sha,
                            reference_ref=relative_input_ref, reference_sha256=file_hash(root / relative_input_ref))
            if song.spatial_calibration is not None:
                relative.update(barline_grid_ref=barline_ref,barline_grid_sha256=file_hash(root/barline_ref),
                                calibration_ref='artifacts/spatial-calibration.json')
            relative_ref = 'artifacts/relative-beats.json'
            write_json(root / relative_ref, relative)
        rhythm_ref = 'artifacts/rhythm-candidates.json'
        rhythm = rhythm_candidates(relative, song.rhythm_grid)
        rhythm.update(relative_beats_ref=relative_ref,
                      relative_beats_sha256=file_hash(root / relative_ref) if relative_ref else None)
        rhythm['identity'] = digest({'version': rhythm['algorithm_version'], 'grid': rhythm['grid'],
                                     'relative_sha256': rhythm['relative_beats_sha256']})
        write_json(root / rhythm_ref, rhythm)
        chart, events, resolution = resolve_pixel_timing(
            events, pixel_intervals, projection,
            Metadata(profile_data["game"], profile_data["mode"], len(profile_data["lanes"]),
                     song.title, song.difficulty), song.bpm,
            beat_zero_ms=(None if adoption['status'] == 'bounded' else
                          read_origin(adoption, adoption['schema_version'], 'effective_beat_zero_ms')),
            measure_beats=song.measure_beats,
            judgment_calibration=song.judgment_calibration)
        resolution_ref = "artifacts/timing-resolution.json"
        if song.origin_adoption is not None:
            resolution['origin_adoption_ref'] = adoption_ref
            resolution['origin_adoption_sha256'] = file_hash(root / adoption_ref)
            resolution['origin_source'] = adoption['origin_source']
        write_json(root / resolution_ref, resolution)
        ranges_ref = 'artifacts/beat-ranges.json'
        endpoint_requested = song.range_uncertainty is not None and song.range_uncertainty.endpoint_component is not None
        endpoint_inputs = {'endpoint_ranges': endpoint_ranges, 'pixels': pixel_intervals,
                           'projection': projection} if endpoint_requested else {}
        ranges = project_beat_ranges(events, resolution, adoption, uncertainty_model=song.range_uncertainty, **endpoint_inputs)
        ranges.update(timing_resolution_ref=resolution_ref, timing_resolution_sha256=file_hash(root / resolution_ref),
                      origin_adoption_ref=adoption_ref, origin_adoption_sha256=file_hash(root / adoption_ref))
        if endpoint_requested:
            ranges.update(endpoint_time_ranges_ref=endpoint_ranges_ref,
                          endpoint_time_ranges_sha256=file_hash(root / endpoint_ranges_ref),
                          time_projection_ref=projection_ref, time_projection_sha256=file_hash(root / projection_ref))
        write_json(root / ranges_ref, ranges)
        chart,events,spatial_resolution=resolve_spatial_chart(chart,events,relative,rhythm,barlines,profile_data,
            physical_review=physical_review, calibration_verification=calibration_verification,
            analysis_scope=scope)
        spatial_resolution_ref='artifacts/spatial-resolution.json'
        spatial_resolution.update(relative_beats_ref=relative_ref,
            relative_beats_sha256=file_hash(root/relative_ref) if relative_ref else None,
            rhythm_candidates_ref=rhythm_ref,rhythm_candidates_sha256=file_hash(root/rhythm_ref),
            barline_grid_ref=barline_ref,barline_grid_sha256=file_hash(root/barline_ref))
        if physical_review_ref is not None:
            spatial_resolution.update(physical_review_ref=physical_review_ref,
                                      physical_review_sha256=file_hash(root / physical_review_ref))
        if calibration_verification_ref is not None:
            spatial_resolution.update(calibration_verification_ref=calibration_verification_ref,
                calibration_verification_sha256=file_hash(root / calibration_verification_ref))
        write_json(root/spatial_resolution_ref,spatial_resolution)
        run["stages"]["timing"] = {"status": "pending" if "timing" in chart.pending_stages else "complete",
                                    "algorithm_version": RESOLUTION_VERSION, "projection_ref": projection_ref,
                                    "projection_version": PROJECTION_VERSION,
                                    "endpoint_time_ranges_ref": endpoint_ranges_ref,
                                    "endpoint_time_ranges_version": ENDPOINT_RANGES_VERSION,
                                    "reference_candidates_ref": reference_ref,
                                    "reference_candidates_version": REFERENCE_VERSION,
                                    "reference_origin_link_ref": reference_link_ref,
                                    "reference_origin_link_version": REFERENCE_LINK_VERSION,
                                    "origin_adoption_ref": adoption_ref, "origin_adoption_version": ORIGIN_VERSION,
                                    "origin_adoption_status": adoption['status'],
                                    "beat_ranges_ref": ranges_ref, "beat_ranges_version": ranges['algorithm_version'],
                                    "bounded_events": ranges['bounded_event_count'],
                                    "artifact_ref": resolution_ref, "resolved_notes": len(chart.notes),
                                    "unresolved_events": len(chart.unresolved)}
        if relative is not None:
            run['stages']['timing'].update(relative_beats_ref=relative_ref,
                relative_beats_version=relative['algorithm_version'],
                relative_beat_candidates=relative['conditional_candidate_count'])
        run['stages']['timing'].update(rhythm_candidates_ref=rhythm_ref,
                                      rhythm_candidates_version=rhythm['algorithm_version'],
                                      rhythm_candidates_identity=rhythm['identity'])
        run['stages']['timing'].update(barline_grid_ref=barline_ref,barline_grid_status=barlines['status'],
                                      barline_grid_version=barlines['algorithm_version'])
        run['stages']['timing'].update(spatial_resolution_ref=spatial_resolution_ref,
                                      spatial_resolution_version=spatial_resolution['algorithm_version'])
        if physical_review_ref is not None:
            run['stages']['timing'].update(physical_review_ref=physical_review_ref,
                                           physical_review=physical_review.receipt())
        if calibration_verification_ref is not None:
            run['stages']['timing'].update(calibration_verification_ref=calibration_verification_ref,
                                          calibration_verification=calibration_verification.receipt())
        active = "export"
        run["stages"][active]["status"] = "running"
        write_json(root / "run.json", run)
        draft_projection = project_draft(observations, motion, decoded, barlines,
            allowed_candidates={c['id'] for p in pixel_intervals
                                for c in (*p['selected_intervals'], *p['alternatives'])},
            analysis_scope=scope)
        draft = create_draft(draft_projection, asdict(chart.metadata), song.bpm,
                            {'video': str(song.video), 'video_sha256': meta['video_sha256'],
                             'alignment_offset_px': meta.get('alignment_offset_px'),
                             'run_id': run_id, 'code': run['code'],
                             'observation_cache_id': observations.id, 'motion_cache_id': motion.id,
                             'decode_cache_id': decoded.id,
                             'analysis_window': song_window(song)})
        save_draft(root / 'draft.json', draft)
        write_review(root / 'review.html', draft)
        run['draft'] = {'path': 'draft.json', 'notes': len(draft['notes']),
                        'unresolved': sum(n['status'] == 'unresolved' for n in draft['notes']),
                        'document_sha256': draft['document_sha256']}
        write_outputs(root, chart, events, observation_ref="artifacts/observations/manifest.json",
                      motion_ref="artifacts/motion/manifest.json", fusion_ref="artifacts/fusion/manifest.json",
                      decode_ref="artifacts/decode/manifest.json", pixel_intervals=pixel_intervals,
                      time_projection_ref=projection_ref, timing_resolution_ref=resolution_ref,
                      endpoint_time_ranges_ref=endpoint_ranges_ref,
                      reference_candidates_ref=reference_ref,
                      reference_origin_link_ref=reference_link_ref,
                      origin_adoption_ref=adoption_ref,
                      beat_ranges_ref=ranges_ref,
                      relative_beats_ref=relative_ref, relative_beats=relative,
                      rhythm_candidates_ref=rhythm_ref,
                      barline_grid_ref=barline_ref,
                      spatial_resolution_ref=spatial_resolution_ref,
                      physical_review_ref=physical_review_ref,
                      calibration_verification_ref=calibration_verification_ref,
                      analysis_scope_ref=scope_ref, analysis_scope=scope,
                      candidates=[c for p in pixel_intervals for c in (*p['selected_intervals'], *p['alternatives'])],
                      diagnostics={"motion": motion.manifest["metadata"]["status_counts"],
                                   "decode": decoded.manifest["metadata"]["event_status_counts"],
                                   "position_calibration": meta.get('position_calibration'),
                                   "endpoint_calibration": "unverified" if any(c["status"] != "verified" for c in profile_data["calibration"].values()) else "verified"})
        run["status"] = chart.status
        run["stages"]["export"] = {"status": "complete", "schema_version": chart_schema(chart)}
        result = EXIT_COMPLETE if chart.status == "complete" else EXIT_PARTIAL
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CaptureError):
            active = "capture"
        run["status"] = "failed"
        run["failure"] = {"stage": active, "type": type(exc).__name__, "message": str(exc)}
        from .difference_constraints import ConstraintConflict
        if isinstance(exc,ConstraintConflict):
            from .contracts import ratio
            conflict_ref='artifacts/coordinate-conflict.json'
            write_json(root/conflict_ref,{'format':'ez2rc.coordinate-conflict','schema_version':1,
                'edges':[{**edge,'upper':ratio(edge['upper'])} for edge in exc.edges]})
            run['failure']['evidence_ref']=conflict_ref
        if active in run["stages"]:
            run["stages"][active] = {"status": "failed"}
        for stage in run["stages"].values():
            if stage["status"] == "running":
                stage["status"] = "interrupted"
        result = EXIT_FAILED
    run["cost"]["elapsed_seconds"] = time.monotonic() - started
    write_json(root / "run.json", run)
    return result, root
