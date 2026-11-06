# EZ2Re:Chart

EZ2Re:Chart (`ez2rc`) reconstructs note candidates from EZ2ON REBOOT : R
gameplay videos using PyAV and OpenCV.

**Game videos, charts, and note template images are not included. Supply your
own input files to run the pipeline.**

## Development status

This branch contains v2. Video observation, scroll estimation, evidence
accumulation, and tap/long-note interval decoding are implemented. Timing
calibration and musical origin inference are still pending.

Current successful runs produce **partial results**, exit with code **2**, and
leave `chart.json.notes` empty. Inspect the pixel candidates and unresolved
events in the report; an empty final note list does not mean the video has no
notes. Complete beat-based chart output is not available yet.

The previous implementation and its instructions are preserved under the `v1`
tag. v2 accepts its own versioned configuration and cache formats.

## Requirements and setup

- Python 3.14
- [uv](https://github.com/astral-sh/uv) for dependency management
- The bundled profile targets fixed-BPM, 1920×1080, 5K recordings.

```bash
uv sync --locked
```

### Recording setup

The bundled `ez2on_5k_1080p_speed8` profile expects:

- Resolution: 1920×1080; recording rate: 60 FPS
- Key mode: 5K
- Panel skin: PG-RESPECT; note skin: EZ2ON
- Note speed: 8.0; judge line: old; judge height: 700
- Judgement tracker: off
- Panel opacity: 100%; alignment: center; background: none
- Recording without player input; LIVE CTRL is recommended.

Video times use decoded presentation timestamps (PTS). Missing timestamps fail
by default; an explicit `timestamp_policy = "fps_fallback"` in the algorithm
config enables a recorded fallback. The profile permits panel translation up
to 32 pixels. Other geometry or recording conditions require a suitable profile
and separate calibration. Tap, head, and tail timing offsets remain unverified.

### Note templates

Crop clean templates at the recording resolution and place them under
`config/skins/ez2on/5k/`. The v2 profile refers to these six images:

| Filename | Description |
| --- | --- |
| `note_cyan.png` | Cyan tap |
| `note_cyan_lnhead.png` | Cyan long-note head |
| `note_cyan_lntail.png` | Cyan long-note tail |
| `note_white.png` | White tap |
| `note_white_lnhead.png` | White long-note head |
| `note_white_lntail.png` | White long-note tail |

## Configuration and usage

Copy `config/v2/song.toml` to another file in the same directory. Set the video
path, title, difficulty, and fixed BPM. BPM is a decimal string, such as
`"191.91"`. Optional `input.start_ms` and `input.end_ms` select a half-open
analysis interval in milliseconds.

Paths are relative to the configuration file. A profile ID resolves to
`profiles/<id>.toml` beside the song config; an explicit profile path is also
accepted. Template paths are relative to the profile file. The optional
`[algorithm]` section selects the common algorithm TOML; omitting it uses defaults.

```bash
uv run ez2rc extract config/v2/my-song.toml --output out
```

Reuse a saved observation cache without decoding the video again:

```bash
uv run ez2rc reconstruct config/v2/my-song.toml \
  --observations "out/<title>/<difficulty>/<run-id>/artifacts/observations" \
  --output out
```

Reconstruction checks the observation settings and cache lineage. If original
video, profile, or template files are present, their identities must still
match. Changes to observation settings or recording geometry require extraction
again. Motion, fusion, and decode caches are reused when their inputs match.

| File | Purpose |
| --- | --- |
| `config/v2/<song>.toml` | Video, metadata, fixed BPM, analysis interval |
| `config/v2/algorithm.toml` | Observation, motion, fusion, and decode settings |
| `config/v2/profiles/<id>.toml` | Geometry, template paths, and calibration status |
| `config/skins/ez2on/5k/*.png` | User-supplied note template images |

## Output

Each attempt creates a new `out/<title>/<difficulty>/<run-id>/` directory:

- `run.json`: input/code identities, settings, stage states, and errors.
- `chart.json`: chart metadata, timing status, and unresolved event references.
- `report.html`: pixel candidates, alternatives, and unresolved events.
- `artifacts/events.json`: event states and supporting evidence references.
- `artifacts/{capture,observations,motion,fusion,decode}/`: stage manifests and caches.

Reusable stage caches live in `out/.cache/` by default. `extract --cache-root`
selects another cache root. Exit codes are `0` for complete, `2` for partial,
and `1` for failed runs; current successful v2 runs return `2`.

See [the architecture overview](doc/ARCHITECTURE.md) for stage boundaries and
cache behavior.

## Tests

The tests generate synthetic inputs without requiring game videos or templates.

```bash
uv run python -m unittest discover -s tests -v
```

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) for personal,
non-commercial use. This project is not affiliated with or endorsed by
NEONOVICE. Game content belongs to its respective copyright holders.
