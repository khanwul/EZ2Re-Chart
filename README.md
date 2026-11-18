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
can still produce a draft picture. Its note positions are estimates for video
review. Numbered items identify choices about the draft.

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
  --observations "out/<title>/<difficulty>/<run-id>/.ez2rc/artifacts/observations" \
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

Open **채보.png** in the result directory printed by `extract` or `reconstruct`.
Read each column from bottom to top, then move right. Each column spans four
measures when the measure length is known, or sixteen beats otherwise. Short
bars mark taps and filled strips mark holds. Thin solid lines mark each beat,
with brighter lines at measure boundaries. Read the beat at the bottom edge
of each tap or hold endpoint. Fainter lines mark occupied subdivisions. If
automatic notes consistently fall slightly ahead of or behind the grid, the
PNG header reports the display shift. Manual positions and the saved draft
retain their original timing.
Numbers identify notes to check. A is the current note, and other letters show
competing note interpretations or a pair of possible beat positions found during extraction.
The key beneath the chart lists the lane and beat for each choice. A question
mark means that the beat is unknown.

Extraction also saves original video photos in `검토사진-r0000/`. Open the photos
for one item together. For example, `check-0003-01.png` and `check-0003-02.png`
show observed moments for item 3. A bright pink outline with a dark outer edge marks each tap. Hold heads are open at the top
and tails are open at the bottom, toward the body of the hold. Read the question at the top, compare the listed answers and use the
command beside your choice. When there are no numbered choices, the
`참고사진-r0000/` folder contains previews that require no input.

Each photo folder is a snapshot of the draft revision. After editing, or to
create photos for a reconstructed or older run, use `ez2rc photos --run <result-directory>`.
Supply `--video <path>` if the recording moved.

From that result directory, use the number and letter in the picture:

```bash
ez2rc choose 3                      # list choices for item 3
ez2rc choose 3 B --frame check.png  # inspect B's original video frame
ez2rc choose 3 B                    # apply B and update the PNG
```

X removes the note. For a missing beat, supply `--start` or `--end` with the
position you determined. Leaving an item unanswered keeps it for later.
Remaining items keep their numbers after each edit. The underlying draft and
analysis files live in `.ez2rc/`. User decisions update the draft. The analysis
record describes the original extraction.

Use `ez2rc chart --run <result-directory>` to regenerate the PNG, including for
older runs with a top-level `draft.json`. The same `--run` option works with
`choose` from outside the result directory.

To compare the current draft with the recording, run this from the result directory:

```bash
ez2rc overlay
```

Open `대조.mp4` in a video player. Pink outlines follow the estimated positions
of the current taps and holds. Solid gold lines mark measure boundaries
(`M1` starts at beat zero). Dashed blue lines mark whole beats, labeled with
their absolute beat number (`B1` is one beat after zero). Pause or slow playback to compare their lanes
and endpoints with the visible notes. Numbered barlines anchor the outlines,
and measured scroll links carry them between frames. Frames without a linked
position show “위치 대조 불가” (position unavailable). The command reports notes
omitted because a beat is unknown. The movie retains the recording's frame
times and audio. Progress on stderr shows elapsed time, the frame-processing
bar, and the final MP4 save stage. For partial extractions, the bar uses video
time when available. Use `--no-progress` to hide progress. Use `--video` if the recording moved, `--run` to select a run,
or `--output another.mp4` to save a new comparison after editing.

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
