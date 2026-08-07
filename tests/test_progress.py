"""Torch-free tests for the encode progress line (semantic_anarchy/progress.py).

The throttling is the load-bearing part: the dashboard turns every ``\\r`` into a
log line, so an unthrottled bar would bury a mine job's real output.
"""

import io

from semantic_anarchy.progress import Progress, bar, fmt_duration


class FakeClock:
    """A clock the test advances by hand, so nothing sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def _pipe(total, clock, **kw):
    """A Progress writing to a non-tty stream (what the webui subprocess gets)."""
    out = io.StringIO()
    return Progress(total, stream=out, clock=clock, **kw), out


def test_bar_fills_with_the_fraction():
    assert bar(0.0, 10) == "[----------]"
    assert bar(1.0, 10) == "[##########]"
    assert bar(0.5, 10) == "[#####-----]"
    # out-of-range input is clamped, never a ragged string
    assert len(bar(-1, 8)) == len(bar(2, 8)) == 10


def test_fmt_duration_units():
    assert fmt_duration(0) == "0s"
    assert fmt_duration(45) == "45s"
    assert fmt_duration(62) == "1m02s"
    assert fmt_duration(3900) == "1h05m"


def test_piped_output_is_throttled_to_one_line_per_interval():
    clock = FakeClock()
    p, out = _pipe(100, clock, interval=1.5)
    p.update(1)                       # first update always draws
    for i in range(2, 51):            # 49 more updates inside the same second
        clock.tick(0.01)
        p.update(i)
    lines = out.getvalue().splitlines()
    assert len(lines) == 1, lines

    clock.tick(2.0)                   # past the interval -> one more line
    p.update(60)
    assert len(out.getvalue().splitlines()) == 2


def test_final_update_always_draws():
    """However the throttle falls, the log ends on a 100% line."""
    clock = FakeClock()
    p, out = _pipe(10, clock, interval=999)
    p.update(1)
    clock.tick(0.1)
    p.update(10)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert "100%" in lines[-1] and "10/10" in lines[-1]


def test_close_emits_when_nothing_was_shown():
    clock = FakeClock()
    p, out = _pipe(10, clock, interval=999)
    p.close()
    assert out.getvalue().splitlines()          # something was written
    p.close()                                   # idempotent
    assert len(out.getvalue().splitlines()) == 1


def test_rate_and_eta_appear_once_time_has_passed():
    clock = FakeClock()
    p, out = _pipe(100, clock, interval=0)
    clock.tick(10.0)
    p.update(25)                                # 25 in 10s -> 2.5/s, 30s left
    line = out.getvalue().splitlines()[-1]
    assert "2.5 item/s" in line
    assert "eta 30s" in line
    assert " 25%" in line


def test_no_eta_on_the_final_line_but_a_total_time():
    clock = FakeClock()
    p, out = _pipe(10, clock, interval=0)
    clock.tick(5.0)
    p.update(10)
    line = out.getvalue().splitlines()[-1]
    assert "eta" not in line
    assert "in 5s" in line


def test_advance_accumulates_and_clamps():
    clock = FakeClock()
    p, _ = _pipe(3, clock, interval=0)
    p.advance()
    p.advance()
    assert p.done == 2
    p.advance(10)
    assert p.done == 3          # never overshoots the total


def test_zero_total_does_not_divide_by_zero():
    clock = FakeClock()
    p, out = _pipe(0, clock, interval=0)
    p.update(0)
    p.close()
    assert "100%" in out.getvalue()


def test_context_manager_closes_without_faking_completion():
    """Leaving the block early (an exception, a cancel) must not print 100%."""
    clock = FakeClock()
    out = io.StringIO()
    with Progress(4, stream=out, clock=clock, interval=999) as p:
        p.update(2)
    last = out.getvalue().splitlines()[-1]
    assert "2/4" in last and "100%" not in last
