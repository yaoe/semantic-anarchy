"""Torch-free tests for the encode progress line (semantic_anarchy/progress.py).

The throttling is the load-bearing part: the dashboard turns every ``\\r`` into a
log line, so an unthrottled bar would bury a mine job's real output.
"""

import io

from semantic_anarchy.progress import Progress, bar, fmt_duration, step_callback


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


def test_slow_rates_are_shown_as_seconds_per_unit():
    clock = FakeClock()
    p, out = _pipe(35, clock, interval=0, unit="step")
    clock.tick(14.0)
    p.update(2)
    assert "7.0s/step" in out.getvalue()


class FakePipe:
    def __init__(self, num_timesteps=None):
        if num_timesteps is not None:
            self.num_timesteps = num_timesteps


def _run_steps(pipe, count, seconds_per_step, steps=None, **kw):
    clock = FakeClock()
    out = io.StringIO()
    cb = step_callback(1, 4, steps or count, stream=out, clock=clock, **kw)
    for step in range(count):
        clock.tick(seconds_per_step)
        latents = {"latents": object()}
        assert cb(pipe, step, 999 - step, latents) is latents  # diffusers needs it back
    return out.getvalue().splitlines()


def test_step_callback_reports_slow_renders_as_it_goes():
    lines = _run_steps(FakePipe(35), 35, 7.0, interval=10.0)
    assert len(lines) > 10
    assert "[gen] image 2/4" in lines[0] and "7.0s/step" in lines[0] and "eta" in lines[0]
    assert "100%" in lines[-1] and "35/35" in lines[-1]


def test_step_callback_prints_one_line_for_a_fast_image():
    """A big sd15 batch must not gain several log lines per image."""
    lines = _run_steps(FakePipe(30), 30, 0.05, interval=10.0)
    assert len(lines) == 1 and "100%" in lines[0]


def test_step_callback_uses_the_pipelines_real_step_count():
    """img2img runs only `strength` of the steps it was asked for."""
    lines = _run_steps(FakePipe(9), 9, 1.0, steps=30, interval=10.0)
    assert "9/9" in lines[-1] and "100%" in lines[-1]


def test_step_callback_falls_back_to_requested_steps():
    lines = _run_steps(FakePipe(), 5, 1.0, interval=10.0)
    assert "5/5" in lines[-1]
