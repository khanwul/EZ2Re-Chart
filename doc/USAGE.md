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

Crop taps and hold heads flush with the bottom edge of the note. Crop hold tails flush with the top edge.

## First run

Follow the three commands in the [quick start](../README.md#quick-start). `init` creates a new project directory containing `song.toml`, the profile, and copies of the templates. The video stays at its original path. To label the result, add `--title "My Song" --difficulty HD` to `init`.

The first identified barline marks beat zero. The draft reads measure lengths from the recording, so meter changes need no extra setting. If the frame-to-frame motion path breaks, a line can still be placed when its passage through the screen agrees with both neighboring numbered lines at one beat position.

`extract` prints the result directory. Keep its hidden `.ez2rc/` folder when moving the draft. To hide per-stage progress on stderr, use `--no-progress`.

A tap can also be placed through a neighboring tap visible with a numbered line. The connected measurements must support one start at a common scroll scale.

## Review the draft

Open `chart.png` in the result directory. Read each column **from bottom to top**, then move right. Short bars are taps, and vertical strips are holds. Read the start beat at the bottom edge of either shape and the end of a hold at its top edge. Brighter grid lines mark measures. Columns continue through the last observed measure line, including rests after the final note.

For the commands below, run this from the repository root, replacing the path with the directory printed by `extract`.

```bash
RUN="path/to/result"
```

### Check a numbered item

The key to the right of your chart lists the choices for each numbered item.

1. List the choices with `uv run ez2rc choose 1 --run "$RUN"`, substituting the item number from your chart.
2. Open the matching photos in `review-photos-r0000/`. For item 1, start with `check-0001-01.png`. To distinguish a hold from a tap, look for a body behind the head and a tail in later frames.
3. Apply the choice that matches the footage. For example, if item 1 matches option B, run `uv run ez2rc choose 1 B --run "$RUN"`. This updates the draft and redraws `chart.png`.

For an individual note, A keeps the current choice and X removes the note. A `?` marks an unknown beat. If you can place it from the footage, supply `--start` or `--end`, for example `--start '2 + 3/16'`. When several taps share one item, A or B places all of them at the beats shown for that option. Leave the item open if you are unsure.

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
| `Draft partial` (exit `2`) | Open the saved draft. Check the printed timing issues and any numbered choices against the recording. |
| Error (exit `1`) | Read the terminal error. If a run directory exists, `.ez2rc/run.json` records the failed stage and reason. |

For a command's options, append `--help`, for example `uv run ez2rc extract --help`. `Draft complete` (exit `0`) means the draft has resolved timing and review choices for the detected notes. Check the recording for missed notes.
