# EZ2Re:Chart v2 usage

Development currently focuses on extracting charts from 5K recordings of fixed-BPM songs. Use the settings below to try the bundled profile.

## Recording setup

| Setting | Value |
| --- | --- |
| Video | 1920×1080, 60 FPS |
| Mode and speed | 5K, speed 8.0 |
| Skins | PG-RESPECT panel, EZ2ON notes |
| Judgment line | Old, height 700, judgment tracker off |
| Panel | Centered, 100% opacity, no background |
| Playback | Record without player input. LIVE CTRL is recommended. |

Keep the judgment line visible at the start. Crop each template at the original resolution as an **82 × 22 pixel PNG**. Place all six in one directory.

| Shape | White lanes | Cyan lanes |
| --- | --- | --- |
| Tap | `note_white.png` | `note_cyan.png` |
| Hold head | `note_white_lnhead.png` | `note_cyan_lnhead.png` |
| Hold tail | `note_white_lntail.png` | `note_cyan_lntail.png` |

For the default draft timing, crop tap and hold-head templates flush with the bottom of the note. Crop hold-tail templates flush with the top of the tail. A profile can specify `draft_offsets_px` for templates with padding. Each offset is measured downward from the first row of the matching template.

## First run

Follow the three commands in the [quick start](../README.md#quick-start). `init` creates a new project directory containing `song.toml`, the profile, and copies of the templates. The video stays at its original path. To label the result, add `--title "My Song" --difficulty HD` to `init`.

`extract` prints the result directory. Keep its hidden `.ez2rc/` folder when moving the draft. To hide per-stage progress on stderr, use `--no-progress`.

## Review the draft

Open `chart.png` in the result directory. Read four measures **from bottom to top** in each column, then scroll right. If the measure length is unknown, each column covers 16 beats.

Short bars are taps, and vertical strips are holds. Read a tap or hold start at its bottom edge and a hold end at its top edge. Grid lines mark beats and measures. Columns show measure numbers starting at 1. The HTML review uses the same reading direction.

For the commands below, run this from the repository root, replacing the path with the directory printed by `extract`.

```bash
RUN="path/to/result"
```

### Check a numbered item

The key to the right of your chart lists the choices for each numbered item.

1. List the choices with `uv run ez2rc choose 1 --run "$RUN"`, substituting the item number from your chart.
2. Open the matching photos in `review-photos-r0000/`. For item 1, start with `check-0001-01.png`. To distinguish a hold from a tap, look for a body behind the head and a tail in later frames.
3. Apply the choice that matches the footage. For example, if item 1 matches option B, run `uv run ez2rc choose 1 B --run "$RUN"`. This updates the draft and redraws `chart.png`.

A keeps the current choice. X removes the note. A `?` marks an unknown beat. Supply `--start` or `--end` if you can establish that position, or leave the item for later.

### Compare with the recording

Run `uv run ez2rc overlay --run "$RUN"` and play `overlay.mp4` slowly. The pink outlines show where the draft places notes. Check unnumbered passages too, since a missed note can leave no review item. If the recording moved, pass its new path with `--video`.

Use `edit` to correct an existing note or `add` for a missed note. This example adds a tap at beat 8 in lane 2.

```bash
uv run ez2rc add "$RUN/.ez2rc/draft.json" --lane 2 --type tap --start 8 --confirm
```

To refresh the review photos after editing or reconstructing a run, use `uv run ez2rc photos --run "$RUN"` with the result directory you want to review.

## Rerun after a change

- Run **`extract`** again if the video, templates, profile, or analysis interval changes. Set `start_ms` and `end_ms` under `[input]` in `song.toml` to analyze part of the video. The end time is exclusive.
- Use **`reconstruct`** if you change only the BPM or downstream analysis settings. Set `RUN` to the previous result directory.

```bash
uv run ez2rc reconstruct my-song/song.toml \
  --observations "$RUN/.ez2rc/artifacts/observations" --output out
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing template or alignment failure | Check the PNG names and dimensions above, then check the panel and judgment line in the opening frames. |
| Irregular frame interval warning | The extractor keeps the original timestamps. Inspect notes near gaps against the recording. |
| `Draft partial` (exit `2`) | Check the printed reasons and numbered choices. The first barline may be ambiguous, a note may lack a beat position, or the interval may have no detected notes. |
| Error (exit `1`) | Read the terminal error. If a run directory exists, `.ez2rc/run.json` records the failed stage and reason. |

For a command's options, append `--help`, for example `uv run ez2rc extract --help`. `Draft complete` (exit `0`) means the draft places the detected notes relative to the first barline and has no outstanding review items. This works without game timing calibration. Check the recording for missed notes.
