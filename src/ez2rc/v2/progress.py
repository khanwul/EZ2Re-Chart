"""Display-only progress estimates; never used for video timing."""

from contextlib import contextmanager
import time
import threading


def frame_progress(count, timestamp_ms, start_ms, end_ms, *, complete=False,
                   label="Observation", total_frames=None):
    frames = str(count) if total_frames is None else f"{count}/{total_frames}"
    detail = f"{frames} frames; video {timestamp_ms / 1000:.1f}s"
    if total_frames:
        fraction = 1. if complete else min(.999, count / total_frames)
    elif end_ms is not None and end_ms > start_ms:
        fraction = 1. if complete else max(0., min(.99, (timestamp_ms - start_ms) / (end_ms - start_ms)))
    else:
        return f"{label}: {detail}"
    filled = int(fraction * 24)
    bar = "#" * filled + "-" * (24 - filled)
    return f"{label}: [{bar}] {fraction:6.1%} | {detail}"


@contextmanager
def progress_display(stream, *, enabled):
    if not enabled:
        yield None
        return
    display = ProgressDisplay(stream)
    display.start()
    try:
        yield display
    finally:
        display.finish()


class WorkProgress:
    """A throttled count of completed work units within one subtask."""
    def __init__(self, report, label, total, unit):
        self.report, self.label, self.total, self.unit = report, label, total, unit
        self.last = float('-inf')
        self.update(0)

    def update(self, done):
        if self.report is None:
            return
        now = time.monotonic()
        if done != self.total and now - self.last < 2:
            return
        self.last = now
        if self.total is None:
            self.report(f"{self.label}: {done} {self.unit}")
            return
        fraction = min(1., max(0., done / self.total)) if self.total else 1.
        n = int(fraction * 24)
        self.report(f"{self.label}: [{'#' * n}{'-' * (24 - n)}] {fraction:6.1%} | "
                    f"{done}/{self.total} {self.unit}")


class ProgressDisplay:
    def __init__(self, stream):
        self.stream = stream
        self.started = time.monotonic()
        self.width = 0
        self.terminal = stream.isatty()
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.worker = None
        self.message = None
        self.updated = time.monotonic()

    def start(self):
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def _heartbeat(self):
        interval = 2 if self.terminal else 15
        while not self.stopped.wait(interval):
            with self.lock:
                if self.message and time.monotonic() - self.updated >= interval:
                    self._render(self.message + ' | working')

    def finish(self):
        self.stopped.set()
        if self.worker is not None:
            self.worker.join()
        self.close()

    def __call__(self, message):
        with self.lock:
            self.message = message
            self.updated = time.monotonic()
            self._render(message)

    def _render(self, message):
        line = f"[{time.monotonic() - self.started:.0f}s] {message}"
        if self.terminal:
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
