"""Display-only progress estimates; never used for video timing."""

import time


def frame_progress(count, timestamp_ms, start_ms, end_ms, *, complete=False):
    detail = f"{count} frames; video {timestamp_ms / 1000:.1f}s"
    if end_ms is None or end_ms <= start_ms:
        return f"Observation: {detail}"
    fraction = 1.0 if complete else max(0.0, min(.99, (timestamp_ms - start_ms) / (end_ms - start_ms)))
    filled = int(fraction * 24)
    bar = "#" * filled + "-" * (24 - filled)
    return f"Observation: [{bar}] {fraction:6.1%} | {detail}"


class ProgressDisplay:
    def __init__(self, stream):
        self.stream = stream
        self.started = time.monotonic()
        self.width = 0
        self.terminal = stream.isatty()

    def __call__(self, message):
        line = f"[{time.monotonic() - self.started:.0f}s] {message}"
        if self.terminal and message.startswith('Observation: ['):
            self.stream.write('\r' + line.ljust(self.width))
            self.width = len(line)
        else:
            self.close()
            self.stream.write(line + '\n')
        self.stream.flush()

    def close(self):
        if self.width:
            self.stream.write('\n')
            self.stream.flush()
            self.width = 0
