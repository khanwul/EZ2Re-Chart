# EZ2Re:Chart

EZ2Re:Chart v2 (`ez2rc`) is a **work in progress (WIP)** that drafts editable note charts from EZ2ON REBOOT: R gameplay videos. Current work focuses on improving note detection and beat estimates for **5K recordings of fixed-BPM songs**. Check the draft against the recording for missed notes and timing errors.

> **Warning**
>
> This project does not include any data (gameplay video, chart, note skins, etc.) from EZ2ON REBOOT: R. If you wish to run this project, please obtain the data yourself.

## Quick start

Prepare your recording and six note templates using the [recording setup guide](doc/USAGE.md#recording-setup). With Python 3.14 and uv installed, run these commands from the repository root, substituting your paths and song BPM.

```bash
uv sync --locked
uv run ez2rc init my-song --video "input/play.mkv" --templates "input/templates" --bpm 120
uv run ez2rc extract my-song/song.toml --output out
```

Open `chart.png` in the printed result directory. A `partial` run (exit code `2`) still saves a draft for review.

For recording setup, draft review, and troubleshooting, see the [v2 user guide](doc/USAGE.md).

## License

The software is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

**Disclaimer**:

This project is not affiliated with, endorsed by, or sponsored by NEONOVICE or any related parties. All copyrights to EZ2ON REBOOT: R game content belong to NEONOVICE and the original creators. This tool is intended for personal, non-commercial use only. All responsibility arising from the use of this tool lies entirely with the user.
