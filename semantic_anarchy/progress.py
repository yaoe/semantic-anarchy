"""A one-line progress bar that reads well in BOTH places this repo shows logs.

The dashboard streams a job's stdout over SSE and splits it into lines. Python's
universal-newline translation makes a bare ``\\r`` a line terminator too, so a
classic redrawing bar would arrive as thousands of separate log lines. So:

* attached to a terminal -> redraw in place with ``\\r`` (a real bar);
* piped (the webui's subprocess) -> emit a *new* line, but only every
  ``interval`` seconds, so a long encode costs a handful of lines.

Either way the final 100% line is always emitted. Torch-free.
"""

from __future__ import annotations

import sys
import time


def fmt_duration(seconds: float) -> str:
    """``45s`` / ``1m02s`` / ``1h04m`` -- short enough to sit inside the bar."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def bar(fraction: float, width: int = 24) -> str:
    """``[############------------]`` for a fraction in [0, 1]."""
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


class Progress:
    """Report progress through ``total`` items, throttled to one line per
    ``interval`` seconds.

    ``clock`` is injectable so the throttling can be tested without sleeping.
    """

    def __init__(self, total: int, label: str = "working", unit: str = "item",
                 prefix: str = "", interval: float = 1.5, width: int = 24,
                 stream=None, clock=time.monotonic,
                 draw_first: bool = True) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.unit = unit
        self.prefix = f"{prefix} " if prefix else ""
        self.interval = interval
        self.width = width
        self.stream = sys.stdout if stream is None else stream
        self._clock = clock
        self.done = 0
        self._start = clock()
        # None = nothing shown yet, so the first update draws. draw_first=False
        # starts the throttle now instead: work faster than one interval prints
        # only its 100% line.
        self._last_emit = None if draw_first else self._start
        self._closed = False

    # ------------------------------------------------------------------ text
    def render(self) -> str:
        """The whole status as one line (no terminator)."""
        elapsed = self._clock() - self._start
        frac = self.done / self.total if self.total else 1.0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        parts = [
            f"{self.prefix}{self.label} {bar(frac, self.width)} {frac * 100:3.0f}%",
            f"{self.done}/{self.total}",
        ]
        if rate > 0:
            # Below 1/s the rate reads better inverted: 0.1 step/s hides 7s vs 9s.
            parts.append(f"{rate:.1f} {self.unit}/s" if rate >= 1
                         else f"{1 / rate:.1f}s/{self.unit}")
            if self.done < self.total:
                parts.append(f"eta {fmt_duration((self.total - self.done) / rate)}")
        if self.done >= self.total:
            parts.append(f"in {fmt_duration(elapsed)}")
        return " · ".join(parts)

    # ---------------------------------------------------------------- output
    @property
    def _tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:            # a StringIO in tests, a closed pipe, ...
            return False

    def _emit(self) -> None:
        line = self.render()
        self.stream.write(f"\r{line}" if self._tty else f"{line}\n")
        try:
            self.stream.flush()
        except Exception:
            pass
        self._last_emit = self._clock()

    def _due(self) -> bool:
        """First call always draws; after that, only once per interval."""
        if self._last_emit is None:
            return True
        return (self._clock() - self._last_emit) >= self.interval

    # ----------------------------------------------------------------- verbs
    def update(self, done: int) -> None:
        """Set the absolute count of finished items."""
        self.done = min(self.total, max(0, int(done)))
        # The last item always draws, so the log ends on a 100% line rather
        # than wherever the throttle happened to fall.
        if self._due() or self.done >= self.total:
            self._emit()

    def advance(self, n: int = 1) -> None:
        self.update(self.done + n)

    def close(self) -> None:
        """Finish the line. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._last_emit is None or self.done < self.total:
            self._emit()
        if self._tty:
            self.stream.write("\n")
            try:
                self.stream.flush()
            except Exception:
                pass

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def step_callback(index: int, n: int, steps: int, interval: float = 10.0,
                  stream=None, clock=time.monotonic):
    """A diffusers ``callback_on_step_end`` that reports denoising steps.

    Build one per image, right before the pipe call: the rate clock starts here,
    so the first step's duration counts. The real step count comes from the
    pipeline's ``num_timesteps`` once it is running, because img2img only runs
    ``strength`` of ``steps``.
    """
    progress = Progress(steps, label=f"image {index + 1}/{n}", unit="step",
                        prefix="[gen]", interval=interval, stream=stream,
                        clock=clock, draw_first=False)

    def on_step_end(pipe, step, timestep, callback_kwargs):
        progress.total = getattr(pipe, "num_timesteps", None) or progress.total
        progress.update(step + 1)
        if progress.done >= progress.total:
            progress.close()
        # diffusers pops latents etc. back out of this, so it must be returned.
        return callback_kwargs

    return on_step_end
